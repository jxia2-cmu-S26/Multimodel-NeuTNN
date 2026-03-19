"""
Temporal NeuTNN v3: Improved L2 supervised learning with stronger class separation.

Key changes:
1. L2 weights initialized with strong class-specific patterns
2. Supervised learning target is more aggressive (per-segment feedback)
3. Alternative voting based on both activation and weight alignment
"""

from __future__ import annotations

import itertools
import json
import math
import random
import time
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from read_imu_dat import channel_roles

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from encoders import DeltaModulationEncoder


cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

# Training controls
INC_LEARN = 1
INTERVAL_TRAIN = 100
INTERVAL_TEST = 100

# Paths
IMU_DATASET_ROOT = Path(__file__).resolve().parent / "imu_dataset_clean"
IMU_MANIFEST_PATH = IMU_DATASET_ROOT / "manifest.json"

# Problem definition
NUM_CLASSES = 4
CLASS_NAMES = ["stand", "slow", "fast", "run"]

# Search budget
SEARCH_SAMPLES = 16
SEARCH_SEED = 42
MAX_TRAIN_WINDOWS = 1500
MAX_TEST_WINDOWS = 600
FINAL_RETRAIN_EPOCHS = 4

# Update rule defaults
# STDP-like update controls for dendritic weights.
CAPTURE = 1.2
SEARCH = 0.0
W_BASE = 12
W_MAX = 16
THRESHOLD = 0
BEST_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "best_temporal_neutnn_v3_config.json"


class activeDendriteLayer(nn.Module):
    def __init__(
        self,
        numDistalInputs: int,
        numSegments: int,
        numDendrites: int,
        numNeurons: int,
        threshold: int,
        wMax: int = 8,
        wInit: str = "uniform",
        rewardEn: bool = True,
        device: str = "cpu",
    ):
        super(activeDendriteLayer, self).__init__()
        self.numDistalInputs = numDistalInputs
        self.numSegments = numSegments
        self.numDendrites = numDendrites
        self.numNeurons = numNeurons
        self.threshold = threshold
        self.wMax = wMax
        self.wInit = wInit
        self.rewardEn = rewardEn
        self.device = device
        self.totalSegments = self.numNeurons * self.numDendrites * self.numSegments

        if self.wInit == "zero":
            self.weights = nn.Parameter(
                torch.zeros(self.totalSegments, self.numDistalInputs),
                requires_grad=False,
            )
        elif self.wInit == "uniform":
            self.weights = nn.Parameter(
                torch.randint(
                    low=0,
                    high=self.wMax + 1,
                    size=(self.totalSegments, self.numDistalInputs),
                ).type(torch.FloatTensor),
                requires_grad=False,
            )
        elif self.wInit == "normal":
            self.weights = nn.Parameter(
                torch.round(
                    ((self.wMax + 1) / 2 + torch.randn(self.totalSegments, self.numDistalInputs)).clamp_(
                        0, self.wMax
                    )
                ),
                requires_grad=False,
            )
        else:
            raise ValueError(f"Unsupported wInit: {self.wInit}")

    def __call__(self, data: torch.Tensor):
        inputData = data.unsqueeze(1).repeat(1, self.numDendrites * self.numSegments, 1).reshape(
            self.totalSegments, self.numDistalInputs
        )
        weightedInput = inputData * self.weights
        summedOutput = torch.sum(weightedInput, dim=1)
        summedOutput[summedOutput < self.threshold] = 0

        reshaped = summedOutput.reshape(-1, self.numSegments)
        self.wtaOutput = torch.zeros(
            self.numNeurons * self.numDendrites,
            self.numSegments,
            dtype=reshaped.dtype,
            device=self.device,
        )
        maxval, indices = torch.max(reshaped, dim=1)
        self.wtaOutput.scatter_(1, indices.unsqueeze(1), maxval.unsqueeze(1))

        outputData = (self.wtaOutput > 0).reshape(-1).unsqueeze(1).repeat(1, self.numDistalInputs)
        return inputData, outputData

    def update(
        self,
        inputData,
        inputLabels,
        outputData,
        weights,
        capture,
        search,
        backoff,
        wBase,
        rewardEn,
        frac=0.25,
    ):
        if rewardEn:
            captureVal = (inputData == 1) * (outputData == 1) * (inputLabels == 1) * capture + (
                (inputData == 0)
                * (outputData == 1)
                * (inputLabels == 0)
                * capture
                * frac
            )
            backoffVal = (inputData == 0) * (outputData == 1) * (inputLabels == 1) * backoff + (
                (inputData == 1)
                * (outputData == 1)
                * (inputLabels == 0)
                * backoff
                * frac
            )
            searchCondition = (inputData == 1) * (outputData == 0) * (inputLabels == 1)
        else:
            captureVal = (inputData == 1) * (outputData == 1) * capture
            backoffVal = (inputData == 0) * (outputData == 1) * backoff
            searchCondition = (inputData == 1) * (outputData == 0)

        weights += captureVal - backoffVal
        weights[searchCondition] = torch.maximum(
            weights[searchCondition],
            torch.minimum(weights[searchCondition] + search, torch.tensor([wBase], device=self.device)),
        )
        return weights.clamp_(0, self.wMax)


def locomotion_label_from_source(source: str) -> int | None:
    # Map session IDs to class IDs. ses-01 is ignored by design.
    p = Path(source)
    ses_id = None
    for part in p.parts:
        if part.startswith("ses-"):
            ses_id = part.split("-", 1)[1]
            break
    if ses_id == "01":
        return None
    if ses_id == "02":
        return 0
    if ses_id == "03":
        return 1
    if ses_id == "04":
        return 2
    if ses_id == "05":
        return 3
    return None


class IMUSlidingWindowDataset(Dataset):
    def __init__(
        self,
        entries: list[dict],
        imu_dataset_root: Path,
        window_len: int,
        stride: int,
        n_thresholds: int,
        threshold_base: float,
        scale: float,
        max_windows: int | None = None,
    ):
        self.samples: list[tuple[np.ndarray, int]] = []
        self.n_units: int = 0
        imu_dataset_root = Path(imu_dataset_root).resolve()
        
        # Keep diagnostics to verify balanced sampling by class.
        win_per_class = {i: 0 for i in range(NUM_CLASSES)}
        entries_processed = {i: 0 for i in range(NUM_CLASSES)}

        # Group entries first so window sampling can be balanced per class.
        by_class = {i: [] for i in range(NUM_CLASSES)}
        for entry in entries:
            source = entry.get("source", "")
            if "_task-ERP_imu" not in source:
                continue
            label = locomotion_label_from_source(source)
            if label is None:
                continue
            by_class[label].append(entry)

        # Split the total window budget equally across classes.
        max_per_class = None
        if max_windows is not None:
            max_per_class = max_windows // NUM_CLASSES

        # Build windows class-by-class to prevent early classes from consuming all budget.
        for class_id in range(NUM_CLASSES):
            class_entries = by_class[class_id]
            
            for entry in class_entries:
                source = entry.get("source", "")
                npy_rel = entry.get("npy")
                ch_names = entry.get("ch_names", [])
                if not npy_rel:
                    continue
                data_path = imu_dataset_root / npy_rel
                if not data_path.exists():
                    continue

                data = np.load(data_path)  # (n_channels, n_times)
                roles = channel_roles(ch_names)
                head_acc = [i for i, n in roles["acc"] if n.upper().startswith("H")]
                head_indices = head_acc[:3]
                if len(head_indices) < 3:
                    continue

                entries_processed[class_id] += 1
                data_sel = data[head_indices].astype(np.float64)
                enc = DeltaModulationEncoder(
                    n_thresholds=n_thresholds,
                    threshold_base=threshold_base,
                    scale=scale,
                )
                spikes = enc.encode(data_sel)
                n_units_local, n_t = spikes.shape

                if n_t < window_len or stride <= 0:
                    continue

                start = 0
                while start + window_len <= n_t:
                    if max_per_class is not None and win_per_class[class_id] >= max_per_class:
                        break
                    end = start + window_len
                    self.samples.append((spikes[:, start:end], class_id))
                    win_per_class[class_id] += 1
                    start += stride

                if self.n_units == 0:
                    self.n_units = n_units_local
        
        # Print effective class balance for quick sanity checking.
        print(f"[Dataset] Loaded {len(self.samples)} windows | Per-class counts: {dict(win_per_class)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)


def load_manifest() -> list[dict]:
    if not IMU_MANIFEST_PATH.exists():
        print(f"Manifest not found: {IMU_MANIFEST_PATH}")
        return []
    with IMU_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_entries(manifest: list[dict], train_ratio=0.8):
    entries = [e for e in manifest if "_task-ERP_imu" in e.get("source", "")]
    if not entries:
        return [], []
    
    # Build (subject, class) buckets to avoid leakage and preserve class coverage.
    by_subject_class = {}
    for e in entries:
        source = e.get("source", "")
        subject = None
        for part in Path(source).parts:
            if part.startswith("sub-"):
                subject = part
                break
        label = locomotion_label_from_source(source)
        if subject and label is not None:
            key = (subject, label)
            if key not in by_subject_class:
                by_subject_class[key] = []
            by_subject_class[key].append(e)
    
    print(f"[Manifest] Found {len(by_subject_class)} (subject, class) pairs")
    
    # Split subjects independently inside each class, then merge.
    train_entries = []
    test_entries = []
    
    for class_id in range(NUM_CLASSES):
        class_subjects = {}
        for (subj, lbl), es in by_subject_class.items():
            if lbl == class_id:
                if subj not in class_subjects:
                    class_subjects[subj] = []
                class_subjects[subj].extend(es)
        
        if not class_subjects:
            print(f"  [Class {class_id}] No data found!")
            continue
        
        subj_list = sorted(class_subjects.keys())
        n_train = max(1, int(len(subj_list) * train_ratio))
        train_subjs = subj_list[:n_train]
        test_subjs = subj_list[n_train:] or [subj_list[-1]]
        
        class_train_count = sum(len(class_subjects[s]) for s in train_subjs)
        class_test_count = sum(len(class_subjects[s]) for s in test_subjs)
        print(f"  [Class {class_id}] {len(subj_list)} subjects | Train: {class_train_count} entries | Test: {class_test_count} entries")
        
        for subj in train_subjs:
            train_entries.extend(class_subjects[subj])
        for subj in test_subjs:
            test_entries.extend(class_subjects[subj])
    
    return train_entries, test_entries


def temporal_unfold(data_bt: torch.Tensor, kernel_len: int, stride_t: int) -> torch.Tensor:
    # Convert (channels, time) into temporal patches for dendritic processing.
    data_4d = data_bt.unsqueeze(0).unsqueeze(2)
    unfold = nn.Unfold(kernel_size=(1, kernel_len), stride=(1, stride_t))
    patches = unfold(data_4d)
    return patches.squeeze(0).T


def dendrite_activity(layer: activeDendriteLayer, voting_thresh: int = 0) -> torch.Tensor:
    # Reduce segment-level WTA output to binary dendrite activity.
    w = layer.wtaOutput.reshape(layer.numNeurons, layer.numDendrites, layer.numSegments)
    act = (torch.max(w, dim=2)[0] > voting_thresh).float()
    return act


def build_supervised_labels_for_grouped_dendrites(
    label: int,
    num_neurons: int,
    num_classes: int,
    prototypes_per_class: int,
    num_segments: int,
    num_distal_inputs: int,
    device_: torch.device,
) -> torch.Tensor:
    # Activate only the prototype group that belongs to the target class.
    dendrite_targets = torch.zeros(num_classes * prototypes_per_class, device=device_)
    lo = label * prototypes_per_class
    hi = (label + 1) * prototypes_per_class
    dendrite_targets[lo:hi] = 1
    labels = dendrite_targets.unsqueeze(0).repeat(num_neurons, 1)
    labels = labels.unsqueeze(2).repeat(1, 1, num_segments * num_distal_inputs)
    return labels.reshape(num_neurons * num_classes * prototypes_per_class * num_segments, num_distal_inputs)


def grouped_votes_improved(
    layer: activeDendriteLayer,
    num_classes: int,
    prototypes_per_class: int,
    voting_thresh: int = 0,
) -> torch.Tensor:
    """Improved voting that uses both activation and weight magnitude."""
    act = dendrite_activity(layer, voting_thresh=voting_thresh)  # (numNeurons, numDendrites)
    
    # Weight-based voting: average weight per dendrite
    w_mag = torch.sqrt(torch.sum(layer.weights ** 2, dim=1)).reshape(-1, layer.numSegments)
    w_mag = torch.mean(w_mag, dim=1)  # (numNeurons * numDendrites,)
    w_mag = w_mag.reshape(layer.numNeurons, layer.numDendrites)
    
    # Combined evidence: current activity + long-term weight strength.
    combined = act + w_mag.float() / layer.wMax
    votes_per_dendrite = torch.sum(combined, dim=0)  # (numDendrites,)
    grouped = votes_per_dendrite.reshape(num_classes, prototypes_per_class)
    return torch.sum(grouped, dim=1)


@dataclass
class SearchConfig:
    n_thresholds: int
    threshold_base: float
    scale: float
    window_len: int
    window_stride: int
    k1: int
    s1: int
    seg1: int
    k2: int
    s2: int
    seg2: int
    proto2: int
    capture: float
    backoff: float
    frac: float
    epochs: int


def save_search_config(cfg: SearchConfig, out_path: Path) -> None:
    """Persist a config to JSON so the same run can be reproduced later."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_search_config(in_path: Path) -> SearchConfig | None:
    """Load a previously saved config JSON into SearchConfig."""
    if not in_path.exists():
        print(f"Config file not found: {in_path}")
        return None
    with in_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return SearchConfig(**payload)


def sample_quick_configs(seed: int = 42, n_samples: int = 16) -> list[SearchConfig]:
    # Small random search over architecture and STDP-related hyperparameters.
    grid = {
        "n_thresholds": [3],
        "threshold_base": [0.0112],
        "scale": [1.8, 2.0, 2.2],
        "window_len": [64, 96, 128],
        "k1": [5, 7],
        "seg1": [24, 36],
        "k2": [3],
        "seg2": [12, 24],
        "proto2": [2, 4],
        "capture": [0.8, 1.0, 1.2, 1.5],
        "backoff": [0.1, 0.3, 0.5, 0.8],
        "frac": [0.0, 0.15, 0.3],
        "epochs": [2, 3],
    }

    combos = []
    for vals in itertools.product(*grid.values()):
        combos.append(
            SearchConfig(
                n_thresholds=vals[0],
                threshold_base=vals[1],
                scale=vals[2],
                window_len=vals[3],
                window_stride=vals[3] // 2,
                k1=vals[4],
                s1=2,
                seg1=vals[5],
                k2=vals[6],
                s2=1,
                seg2=vals[7],
                proto2=vals[8],
                capture=vals[9],
                backoff=vals[10],
                frac=vals[11],
                epochs=vals[12],
            )
        )

    rng = random.Random(seed)
    rng.shuffle(combos)
    return combos[: min(n_samples, len(combos))]


def run_one_config(
    cfg: SearchConfig,
    train_entries: list[dict],
    test_entries: list[dict],
    max_train_windows: int = 1500,
    max_test_windows: int = 600,
):
    # Build balanced train/test window datasets from manifest entries.
    train_ds = IMUSlidingWindowDataset(
        entries=train_entries,
        imu_dataset_root=IMU_DATASET_ROOT,
        window_len=cfg.window_len,
        stride=cfg.window_stride,
        n_thresholds=cfg.n_thresholds,
        threshold_base=cfg.threshold_base,
        scale=cfg.scale,
        max_windows=max_train_windows,
    )
    test_ds = IMUSlidingWindowDataset(
        entries=test_entries,
        imu_dataset_root=IMU_DATASET_ROOT,
        window_len=cfg.window_len,
        stride=cfg.window_stride,
        n_thresholds=cfg.n_thresholds,
        threshold_base=cfg.threshold_base,
        scale=cfg.scale,
        max_windows=max_test_windows,
    )

    if len(train_ds) == 0 or len(test_ds) == 0:
        return None

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    n_units = train_ds.n_units
    n1 = (cfg.window_len - cfg.k1) // cfg.s1 + 1
    if n1 <= 2:
        return None
    d1 = n_units * cfg.k1

    # L2 consumes class-channel activity produced by L1 over pseudo-time.
    n2_input_channels = NUM_CLASSES
    d2 = n2_input_channels * cfg.k2
    n2 = (n1 - cfg.k2) // cfg.s2 + 1
    if n2 <= 0:
        return None

    l1 = activeDendriteLayer(
        numDistalInputs=d1,
        numSegments=cfg.seg1,
        numDendrites=NUM_CLASSES,
        numNeurons=n1,
        threshold=THRESHOLD,
        wMax=W_MAX,
        rewardEn=False,
        device=str(device),
    ).to(device)

    l2_dendrites = NUM_CLASSES * cfg.proto2
    l2 = activeDendriteLayer(
        numDistalInputs=d2,
        numSegments=cfg.seg2,
        numDendrites=l2_dendrites,
        numNeurons=n2,
        threshold=THRESHOLD,
        wMax=W_MAX,
        rewardEn=True,
        device=str(device),
    ).to(device)
    
    # Initialize class prototype groups with different biases to break symmetry.
    with torch.no_grad():
        for class_id in range(NUM_CLASSES):
            for proto_id in range(cfg.proto2):
                dendrite_idx = class_id * cfg.proto2 + proto_id
                segment_start = dendrite_idx * cfg.seg2
                segment_end = segment_start + cfg.seg2
                # Strong class bias: prefer inputs related to this class
                class_bias = 3 * float(class_id) / NUM_CLASSES
                l2.weights[segment_start:segment_end] = l2.weights[segment_start:segment_end] + class_bias

    def forward_two_layer(sample_win: torch.Tensor):
        # L1: temporal pattern extraction from encoded spikes.
        x1 = temporal_unfold(sample_win, cfg.k1, cfg.s1)
        in1, out1 = l1(x1)

        # L2 input: class-wise L1 activity sequence.
        a1 = dendrite_activity(l1, voting_thresh=0).T  # (NUM_CLASSES, n1)
        x2 = temporal_unfold(a1, cfg.k2, cfg.s2)
        in2, out2 = l2(x2)
        return in1, out1, in2, out2

    start = time.time()
    for _ in range(cfg.epochs):
        for idx, (data, target) in enumerate(train_loader):
            if cuda:
                data = data.to(device)
                target = target.to(device)

            label = int(target[0].item())
            sample = data[0]
            in1, out1, in2, out2 = forward_two_layer(sample)

            # L1 update
            zeros_label = torch.zeros_like(in1)
            l1.weights = l1.update(
                in1, zeros_label, out1, l1.weights,
                cfg.capture, SEARCH, cfg.backoff, W_BASE, False, cfg.frac,
            )

            # L2 update
            labels_l2 = build_supervised_labels_for_grouped_dendrites(
                label=label,
                num_neurons=l2.numNeurons,
                num_classes=NUM_CLASSES,
                prototypes_per_class=cfg.proto2,
                num_segments=l2.numSegments,
                num_distal_inputs=l2.numDistalInputs,
                device_=device,
            )
            l2.weights = l2.update(
                in2, labels_l2, out2, l2.weights,
                cfg.capture, SEARCH, cfg.backoff, W_BASE, True, cfg.frac,
            )

            if (idx + 1) % INTERVAL_TRAIN == 0:
                print(f"[cfg train] {idx+1}/{len(train_loader)}")

    # Evaluate with confusion matrix on the balanced test set.
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per_class_samples = {i: 0 for i in range(NUM_CLASSES)}
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            if cuda:
                data = data.to(device)
                target = target.to(device)
            label = int(target[0].item())
            per_class_samples[label] += 1
            sample = data[0]
            _, _, _, _ = forward_two_layer(sample)
            votes = grouped_votes_improved(l2, NUM_CLASSES, cfg.proto2, voting_thresh=0)
            pred = int(torch.argmax(votes).item())
            conf[label, pred] += 1

            if INC_LEARN == 1:
                in1, out1, in2, out2 = forward_two_layer(sample)
                zeros_label = torch.zeros_like(in1)
                l1.weights = l1.update(
                    in1, zeros_label, out1, l1.weights,
                    cfg.capture, SEARCH, cfg.backoff, W_BASE, False, cfg.frac,
                )
                labels_l2 = build_supervised_labels_for_grouped_dendrites(
                    label=label,
                    num_neurons=l2.numNeurons,
                    num_classes=NUM_CLASSES,
                    prototypes_per_class=cfg.proto2,
                    num_segments=l2.numSegments,
                    num_distal_inputs=l2.numDistalInputs,
                    device_=device,
                )
                l2.weights = l2.update(
                    in2, labels_l2, out2, l2.weights,
                    cfg.capture, SEARCH, cfg.backoff, W_BASE, True, cfg.frac,
                )

            if (idx + 1) % INTERVAL_TEST == 0:
                print(f"[cfg test] {idx+1}/{len(test_loader)}")

    total = int(conf.sum())
    acc = float(np.trace(conf) / total) if total > 0 else 0.0
    elapsed = time.time() - start
    
    return {
        "config": cfg,
        "acc": acc,
        "conf": conf,
        "elapsed_sec": elapsed,
        "train_windows": len(train_ds),
        "test_windows": len(test_ds),
        "test_class_dist": per_class_samples,
    }


def plot_confusion(conf: np.ndarray, out_path: Path):
    plt.figure(figsize=(4.2, 4.0))
    im = plt.imshow(conf, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_marks, CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Temporal NeuTNN v3 (IMU only)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def run_search(config_save_path: Path = BEST_CONFIG_PATH):
    # End-to-end experiment runner: split -> search -> report best config.
    manifest = load_manifest()
    if not manifest:
        print("No manifest found; aborting search.")
        return

    print(f"[Manifest] Total entries in manifest: {len(manifest)}")
    train_entries, test_entries = split_entries(manifest, train_ratio=0.8)
    if not train_entries or not test_entries:
        print("Insufficient entries for train/test split.")
        return

    configs = sample_quick_configs(seed=SEARCH_SEED, n_samples=SEARCH_SAMPLES)
    print(f"Running quick search with {len(configs)} configs on {device}.")
    print()

    results = []
    for i, cfg in enumerate(configs, start=1):
        print("=" * 80)
        print(f"Config {i}/{len(configs)} | capture={cfg.capture}, backoff={cfg.backoff}, epochs={cfg.epochs}")
        out = run_one_config(cfg, train_entries, test_entries, max_train_windows=MAX_TRAIN_WINDOWS, max_test_windows=MAX_TEST_WINDOWS)
        if out is None:
            print("Skipped.")
            continue
        test_dist_str = " | ".join([f"C{c}:{out['test_class_dist'][c]}" for c in range(NUM_CLASSES)])
        print(f"Acc={out['acc']*100:.2f}% | TestDist=[{test_dist_str}]")
        results.append(out)

    if not results:
        print("No valid results.")
        return

    results.sort(key=lambda x: x["acc"], reverse=True)
    best = results[0]
    print("=" * 80)
    print(f"BEST ACCURACY: {best['acc']*100:.2f}%")
    print(f"BEST CONFIG: {best['config']}")
    print(best["conf"])
    for c in range(NUM_CLASSES):
        row_sum = best["conf"][c, :].sum()
        if row_sum > 0:
            recall = best["conf"][c, c] / row_sum
            print(f"  Class {c}: {recall*100:.1f}%")

    # Save the best confusion matrix plot for quick inspection.
    out_dir = Path(__file__).resolve().parent / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "best_temporal_neutnn_v3_confusion.png"
    plot_confusion(best["conf"], out_path)
    print(f"Saved confusion plot to: {out_path}")

    # Save the best hyperparameter set for future runs.
    save_search_config(best["config"], config_save_path)
    print(f"Saved best config to: {config_save_path}")


def run_saved_config(config_path: Path):
    """Run one evaluation using a previously saved config JSON."""
    cfg = load_search_config(config_path)
    if cfg is None:
        return

    manifest = load_manifest()
    if not manifest:
        print("No manifest found; aborting run.")
        return

    print(f"Loaded config from: {config_path}")
    print(f"Config: {cfg}")
    train_entries, test_entries = split_entries(manifest, train_ratio=0.8)
    if not train_entries or not test_entries:
        print("Insufficient entries for train/test split.")
        return

    out = run_one_config(
        cfg,
        train_entries,
        test_entries,
        max_train_windows=MAX_TRAIN_WINDOWS,
        max_test_windows=MAX_TEST_WINDOWS,
    )
    if out is None:
        print("Saved config produced no valid run (shape/data issue).")
        return

    print("=" * 80)
    print(f"LOADED CONFIG ACCURACY: {out['acc']*100:.2f}%")
    print(out["conf"])
    for c in range(NUM_CLASSES):
        row_sum = out["conf"][c, :].sum()
        if row_sum > 0:
            recall = out["conf"][c, c] / row_sum
            print(f"  Class {c}: {recall*100:.1f}%")

    out_dir = Path(__file__).resolve().parent / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "loaded_temporal_neutnn_v3_confusion.png"
    plot_confusion(out["conf"], out_path)
    print(f"Saved confusion plot to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal NeuTNN v3 search and replay")
    parser.add_argument(
        "--load-config",
        action="store_true",
        help="Load and run a saved config instead of doing a new search.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=str(BEST_CONFIG_PATH),
        help="Path to save best config (search mode) or load config (load mode).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config_path)
    if args.load_config:
        run_saved_config(cfg_path)
    else:
        run_search(config_save_path=cfg_path)
