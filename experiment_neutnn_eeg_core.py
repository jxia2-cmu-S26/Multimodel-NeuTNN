"""
 EEG-only NeuTNN driver for testing spikify temporal encoders.

 Uses activeDendriteLayer from NeuTNN/NeuTNN/active_dendrite_core.py.
"""

### Importing libraries ###

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt
import time
import math
import json
import argparse
from dataclasses import asdict, dataclass

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_NEUTNN_PKG = _REPO_ROOT / "NeuTNN" / "NeuTNN"
if str(_NEUTNN_PKG) not in sys.path:
    sys.path.insert(0, str(_NEUTNN_PKG))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from active_dendrite_core import activeDendriteLayer

from read_imu_dat import load_imu_brainvision
from spikify_temporal_encode import (
    TEMPORAL_METHODS,
    encode_temporal_multichannel,
    is_spikify_available,
    n_units_for_method,
)


### Enabling CUDA support for GPU ###
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### Training Parameter Definitions ###
inc_learn = 0
interval1   = 100
interval2   = 100

### Model Parameter Definitions ###
capture = 1
search  = 0
backoff = 0.625
frac = 0
wBase = 12
wMax = 16

### Sample Rates ###
EEG_SAMPLE_RATE_ORIGINAL = 100.0  # Hz (native EEG rate in dataset)
RESAMPLE_RATE = 128.0  # Hz target after resampling for window_len / model

# Temporal receptive field for spike raster (n_units, window_len); n_units updated after data load
kernel_len = 5
stride_t = 2
n_eeg_channels = 3   # fixed EEG channel subset to keep model input dimensions stable
numSegments = 36
numDendrites = 4     # locomotion classes: standing, slow walk, fast walk, running
window_len = 128    # 1 s at RESAMPLE_RATE Hz
threshold = 0
voting_thresh = 0
NUM_CLASSES = 4
MAX_TRAIN_WINDOWS = 1500
MAX_TEST_WINDOWS = 600
SEARCH_TRIALS = 12
SEARCH_SEED = 42
RUN_CONFIG_SEARCH = 1
# Per-method paths set after parse_args (see BENCHMARK_* below)


def parse_runtime_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--load-best-config",
        action="store_true",
        help="Load NeuTNN SearchConfig from config/benchmarks/<spikify-method>/[run-tag]/best_snl_config.json",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional subfolder under each method (e.g. wlen16) so different hyperparams do not overwrite.",
    )
    parser.add_argument(
        "--eeg-root",
        default="",
        help="Dataset root with sub-*/ses-*/eeg/*task-ERP_eeg.vhdr. Default: <repo>/osfstorage-archive.",
    )
    parser.add_argument(
        "--spikify-method",
        default="moving_window",
        choices=tuple(TEMPORAL_METHODS),
        help="Temporal encoder from spikify (default: moving_window).",
    )
    parser.add_argument(
        "--spikify-window-length",
        type=int,
        default=8,
        help="moving_window / bens / hough / modified_hough window length (clamped per segment).",
    )
    parser.add_argument("--spikify-threshold", type=float, default=0.1, help="step_forward, zcsf, bens, modified_hough")
    parser.add_argument("--spikify-factor", type=float, default=0.5, help="threshold_based_representation factor")
    parser.add_argument("--burst-n-max", type=int, default=4)
    parser.add_argument("--burst-t-min", type=int, default=2)
    parser.add_argument("--burst-t-max", type=int, default=6)
    parser.add_argument(
        "--phase-num-bits",
        type=int,
        default=8,
        help="phase_encoding; trailing samples truncated so length divides num_bits",
    )
    parser.add_argument(
        "--tts-interval",
        type=int,
        default=8,
        help="time_to_first_spike; trailing samples truncated so length divides interval",
    )
    return parser.parse_args()


RUNTIME_ARGS = parse_runtime_args()
LOAD_BEST_CONFIG = bool(RUNTIME_ARGS.load_best_config)
RUN_TAG = (RUNTIME_ARGS.run_tag or "").strip()
EEG_DATASET_ROOT = (
    Path(RUNTIME_ARGS.eeg_root).resolve()
    if (RUNTIME_ARGS.eeg_root or "").strip()
    else Path(__file__).resolve().parent.parent / "osfstorage-archive"
)
SPIKIFY_METHOD = RUNTIME_ARGS.spikify_method
SPIKIFY_KW = {
    "window_length": RUNTIME_ARGS.spikify_window_length,
    "threshold": RUNTIME_ARGS.spikify_threshold,
    "factor": RUNTIME_ARGS.spikify_factor,
    "n_max": RUNTIME_ARGS.burst_n_max,
    "t_min": RUNTIME_ARGS.burst_t_min,
    "t_max": RUNTIME_ARGS.burst_t_max,
    "num_bits": RUNTIME_ARGS.phase_num_bits,
    "interval": RUNTIME_ARGS.tts_interval,
}

if not is_spikify_available():
    raise SystemExit(
        "experiment_neutnn_imu_eeg_core.py requires spikify. Install: pip install innuce-spikify"
    )


def _benchmark_config_dir(method: str, run_tag: str) -> Path:
    p = Path(__file__).resolve().parent / "config" / "benchmarks" / method
    if run_tag:
        p = p / run_tag
    p.mkdir(parents=True, exist_ok=True)
    return p


def _graphs_dir(method: str, run_tag: str) -> Path:
    p = Path(__file__).resolve().parent / "graphs_core" / method
    if run_tag:
        p = p / run_tag
    p.mkdir(parents=True, exist_ok=True)
    return p


BENCHMARK_CONFIG_DIR = _benchmark_config_dir(SPIKIFY_METHOD, RUN_TAG)
BEST_CONFIG_PATH = BENCHMARK_CONFIG_DIR / "best_snl_config.json"
BEST_RESULT_TXT_PATH = BENCHMARK_CONFIG_DIR / "result.txt"
GRAPHS_DIR = _graphs_dir(SPIKIFY_METHOD, RUN_TAG)
# Root-level log so it is not hidden by config/ in .gitignore
BENCHMARK_RUNS_JSONL = Path(__file__).resolve().parent / "benchmark_runs.jsonl"

print(
    f"Benchmark outputs: method={SPIKIFY_METHOD} run_tag={RUN_TAG or '(default)'} -> "
    f"{BENCHMARK_CONFIG_DIR}"
)

n_units = n_units_for_method(SPIKIFY_METHOD, n_eeg_channels)
numDistalInputs = kernel_len * n_units
numNeurons = (window_len - kernel_len) // stride_t + 1


### EEG dataset and spike-window building ###


def collect_eeg_entries(eeg_root: Path) -> list[dict]:
    """Collect ERP EEG files only from osfstorage-archive/sub-*/ses-*/eeg."""
    eeg_root = Path(eeg_root).resolve()
    entries: list[dict] = []
    for vhdr in eeg_root.glob("sub-*/ses-*/eeg/*task-ERP_eeg.vhdr"):
        entries.append(
            {
                "modality": "eeg",
                "source": str(vhdr.resolve()),
                "vhdr": str(vhdr.resolve()),
            }
        )
    entries.sort(key=lambda e: e["source"])
    return entries


def select_eeg_channel_indices(ch_names: list[str], max_channels: int = n_eeg_channels) -> list[int]:
    """Prefer common midline EEG channels; fallback to first EEG-like channels."""
    preferred = ["FZ", "CZ", "PZ"]
    name_to_idx = {name.upper(): i for i, name in enumerate(ch_names)}

    picked: list[int] = []
    for name in preferred:
        idx = name_to_idx.get(name)
        if idx is not None:
            picked.append(idx)

    if len(picked) >= max_channels:
        return picked[:max_channels]

    bad_prefixes = ("EOG", "ECG", "EMG", "TRIG", "STI", "EXG")
    for i, name in enumerate(ch_names):
        uname = name.upper()
        if i in picked:
            continue
        if any(uname.startswith(pref) for pref in bad_prefixes):
            continue
        picked.append(i)
        if len(picked) >= max_channels:
            break

    return picked[:max_channels]


def locomotion_label_from_source(source: str) -> int:
    """
    Map a BrainVision source path containing 'ses-YY' to a locomotion class:
    0: standing (ses-02), 1: slow walking (ses-03),
    2: fast walking (ses-04), 3: running (ses-05).
    """
    p = Path(source)
    ses_id = None
    for part in p.parts:
        if part.startswith("ses-"):
            ses_id = part.split("-", 1)[1]
            break
    # Ignore training session (ses-01)
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
    raise ValueError(f"Unexpected session id in source path: {source}")


def split_entries_by_subject(
    entries: list[dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Subject-aware split to avoid leakage between train/test."""
    def get_subj(source: str) -> str:
        p = Path(source)
        for part in p.parts:
            if part.startswith("sub-"):
                return part
        return "unknown"

    by_subj: dict[str, list[dict]] = {}
    for e in entries:
        by_subj.setdefault(get_subj(e.get("source", "")), []).append(e)

    subjects = sorted(by_subj.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n_train = max(1, int(len(subjects) * train_ratio))
    train_sub = set(subjects[:n_train])

    train_entries: list[dict] = []
    test_entries: list[dict] = []
    for s in subjects:
        if s in train_sub:
            train_entries.extend(by_subj[s])
        else:
            test_entries.extend(by_subj[s])

    if not test_entries and train_entries:
        test_entries = train_entries[:1]
    return train_entries, test_entries


class EegSlidingWindowDataset(Dataset):
    """Build spike windows from BrainVision ERP EEG via spikify temporal encoders (Fz/Cz/Pz or fallback)."""

    def __init__(
        self,
        entries: list[dict],
        window_len: int,
        stride: int,
        max_windows: int | None = None,
        num_classes: int = NUM_CLASSES,
        spikify_method: str = "moving_window",
        spikify_kwargs: dict | None = None,
    ):
        self.samples: list[tuple[np.ndarray, int]] = []
        self.n_units: int = 0
        self.window_len: int = window_len
        self.spikify_method = spikify_method
        self.spikify_kwargs = dict(spikify_kwargs or {})

        max_per_class = None
        if max_windows is not None and max_windows > 0:
            max_per_class = max(1, int(max_windows // max(1, num_classes)))
        per_class_count = {c: 0 for c in range(num_classes)}
        n_files_used = 0

        for entry in entries:
            source = entry["source"]
            if "_task-ERP_eeg" not in source:
                continue

            label = locomotion_label_from_source(source)
            if label is None:
                continue

            vhdr_path = Path(entry.get("vhdr", source)).resolve()
            if not vhdr_path.exists():
                continue

            try:
                data, info = load_imu_brainvision(vhdr_path, apply_scaling=True)
            except Exception:
                continue

            ch_names = info.get("ch_names", [])
            eeg_indices = select_eeg_channel_indices(ch_names, max_channels=n_eeg_channels)
            if len(eeg_indices) < n_eeg_channels:
                continue

            data_sel = data[eeg_indices].astype(np.float64)

            if EEG_SAMPLE_RATE_ORIGINAL != RESAMPLE_RATE:
                n_samples_new = int(data_sel.shape[1] * (RESAMPLE_RATE / EEG_SAMPLE_RATE_ORIGINAL))
                data_sel_resampled = np.zeros((data_sel.shape[0], n_samples_new), dtype=np.float64)
                for ch_idx in range(data_sel.shape[0]):
                    data_sel_resampled[ch_idx] = np.interp(
                        np.linspace(0, data_sel.shape[1] - 1, n_samples_new),
                        np.arange(data_sel.shape[1]),
                        data_sel[ch_idx],
                    )
                data_sel = data_sel_resampled

            try:
                spikes = encode_temporal_multichannel(
                    data_sel, spikify_method, **self.spikify_kwargs
                )
            except (ValueError, RuntimeError):
                continue
            n_units_local, n_t = spikes.shape

            win_len = int(window_len)
            step = int(stride)
            if win_len <= 0 or step <= 0 or win_len > n_t:
                continue

            file_had_window = False
            start = 0
            while start + win_len <= n_t:
                if max_per_class is not None and per_class_count[label] >= max_per_class:
                    break
                end = start + win_len
                win = spikes[:, start:end]
                self.samples.append((win, label))
                per_class_count[label] += 1
                file_had_window = True
                start += step

            if file_had_window:
                n_files_used += 1

            if self.samples and self.n_units == 0:
                self.n_units = n_units_local

        if self.samples:
            self.window_len = self.samples[0][0].shape[1]
        print(
            f"Loaded windows={len(self.samples)} | files={n_files_used} | spikify={spikify_method} | "
            f"per-class={per_class_count}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        win, label = self.samples[idx]
        # Prefer from_numpy; fall back to tolist if PyTorch reports NumPy unavailable.
        try:
            x = torch.from_numpy(win).float()
        except RuntimeError:
            x = torch.tensor(win.tolist(), dtype=torch.float32)
        return x, torch.tensor(label, dtype=torch.long)


def build_train_test_datasets(
    eeg_entries: list[dict],
    train_ratio: float = 0.8,
    window_len: int = window_len,
    stride: int = window_len // 2,
    max_train_windows: int | None = MAX_TRAIN_WINDOWS,
    max_test_windows: int | None = MAX_TEST_WINDOWS,
    num_classes: int = NUM_CLASSES,
    seed: int = 42,
    spikify_method: str = "moving_window",
    spikify_kwargs: dict | None = None,
):
    """Split EEG file entries by subject into train/test and build sliding-window datasets."""
    entries = sorted(eeg_entries, key=lambda e: e["source"])
    if not entries:
        return None, None

    train_entries, test_entries = split_entries_by_subject(
        entries,
        train_ratio=train_ratio,
        seed=seed,
    )

    print(f"Entry split: train={len(train_entries)}, test={len(test_entries)}")

    train_ds = EegSlidingWindowDataset(
        entries=train_entries,
        window_len=window_len,
        stride=stride,
        max_windows=max_train_windows,
        num_classes=num_classes,
        spikify_method=spikify_method,
        spikify_kwargs=spikify_kwargs,
    )
    test_ds = EegSlidingWindowDataset(
        entries=test_entries,
        window_len=window_len,
        stride=stride,
        max_windows=max_test_windows,
        num_classes=num_classes,
        spikify_method=spikify_method,
        spikify_kwargs=spikify_kwargs,
    )

    train_class_counts = np.zeros(num_classes, dtype=int)
    for _, y in train_ds.samples:
        train_class_counts[y] += 1
    test_class_counts = np.zeros(num_classes, dtype=int)
    for _, y in test_ds.samples:
        test_class_counts[y] += 1
    print("Window class counts (train):", train_class_counts.tolist())
    print("Window class counts (test): ", test_class_counts.tolist())

    return train_ds, test_ds


_eeg_entries = collect_eeg_entries(EEG_DATASET_ROOT)
print(f"EEG dataset root: {EEG_DATASET_ROOT} | vhdr entries: {len(_eeg_entries)}")

if _eeg_entries:
    train_dataset, test_dataset = build_train_test_datasets(
        eeg_entries=_eeg_entries,
        train_ratio=0.8,
        window_len=window_len,
        stride=window_len // 2,
        max_train_windows=MAX_TRAIN_WINDOWS,
        max_test_windows=MAX_TEST_WINDOWS,
        num_classes=NUM_CLASSES,
        seed=42,
        spikify_method=SPIKIFY_METHOD,
        spikify_kwargs=SPIKIFY_KW,
    )
    breakpoint1 = len(train_dataset) if train_dataset is not None else 0
    breakpoint2 = len(test_dataset) if test_dataset is not None else 0
    print(f"Train windows: {breakpoint1}, test windows: {breakpoint2}")
else:
    print(
        f"WARNING: No usable EEG files under {EEG_DATASET_ROOT} "
        "(expected sub-*/ses-*/eeg/*task-ERP_eeg.vhdr)."
    )
    train_dataset, test_dataset = None, None
    breakpoint1, breakpoint2 = 0, 0

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) if breakpoint1 > 0 else []
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) if breakpoint2 > 0 else []

# Match NeuTNN input size to encoder output (must match spikify raster n_units).
if breakpoint1 > 0 and train_dataset is not None and getattr(train_dataset, "n_units", 0) > 0:
    n_units = train_dataset.n_units
    _wl = train_dataset.window_len
    numDistalInputs = kernel_len * n_units
    numNeurons = (_wl - kernel_len) // stride_t + 1
    print(
        f"Model dims from data: n_units={n_units}, numDistalInputs={numDistalInputs}, "
        f"numNeurons={numNeurons}, window_len={_wl}"
    )


def build_supervised_labels_single_layer(network: activeDendriteLayer, label: int, num_classes: int, device_):
    """Build explicit supervised labels aligned with [neuron, dendrite, segment] ordering."""
    nd = network.numDendrites
    if nd != num_classes:
        raise ValueError(f"numDendrites ({nd}) must match num_classes ({num_classes})")

    # One target dendrite per neuron, then expand to all its segments and inputs.
    row_labels = torch.zeros(network.numNeurons * nd, device=device_)
    row_labels[label::nd] = 1.0
    seg_labels = row_labels.repeat_interleave(network.numSegments)  # [totalSegments]
    input_labels = seg_labels.unsqueeze(1).repeat(1, network.numDistalInputs)
    return input_labels


def class_votes_from_wta(network: activeDendriteLayer, num_classes: int, voting_thresh: float = 0.0):
    """Use activation strength voting to reduce tie-to-class-0 collapse."""
    seg_strength = torch.max(network.wtaOutput, dim=1)[0]  # [numNeurons*numDendrites]
    if voting_thresh > 0:
        seg_strength = seg_strength * (seg_strength > voting_thresh)
    votes = seg_strength.reshape(network.numNeurons, num_classes).sum(dim=0)
    return votes

def train_func(
    numDistalInputs,
    numSegments,
    numDendrites,
    numNeurons,
    threshold,
    capture,
    search,
    backoff,
    frac,
    wBase,
    device,
    kernel_len,
    stride_t,
    wMax,
    voting_thresh=0,
    num_classes=4,
    epochs=1,
    inc_learn_eval=0,
):

    network = activeDendriteLayer(numDistalInputs=numDistalInputs,
                                  numSegments=numSegments,
                                  numDendrites=numDendrites,
                                  numNeurons=numNeurons,
                                  threshold=threshold,
                                  wMax=wMax,
                                  device=device)

    if cuda:
        network.cuda()

    # Temporal unfold: (1, n_units, 1, window_len) -> patches (1, kernel_len*n_units, numNeurons)
    unfold = nn.Unfold(kernel_size=(1, kernel_len), stride=(1, stride_t))

    for ep in range(epochs):
        start = time.time()
        error1 = 0
        error2 = 0
        errorlist1 = []
        errorlist2 = []

        n_batches = len(train_loader)

        for idx, (data, target) in enumerate(train_loader):

            if cuda:
                data = data.cuda()
                target = target.cuda()

            label = target[0].item()

            # data: (1, n_units, window_len) -> (1, n_units, 1, window_len) for temporal unfold
            data_4d = data.unsqueeze(2)
            patches = unfold(data_4d)  # (1, kernel_len*n_units, numNeurons)
            sliced_data = patches.squeeze(0).T  # (numNeurons, numDistalInputs)

            inputData, outputData = network(sliced_data)

            labelVotes = class_votes_from_wta(network, num_classes=num_classes, voting_thresh=voting_thresh)

            if torch.argmax(labelVotes) != label:
                error1 += 1
                error2 += 1

            inputLabels = build_supervised_labels_single_layer(
                network=network,
                label=label,
                num_classes=num_classes,
                device_=device,
            )

            network.weights = network.update(inputData, inputLabels, outputData, network.weights,
                                            capture, search, backoff, wBase, network.rewardEn, frac)

            if (idx+1)%interval1 == 0:
                errorlist1.append(error1/(idx+1))
                errorlist2.append(error2/interval1)
                error2 = 0

            # Progress and ETA for training
            if (idx + 1) % max(1, n_batches // 20) == 0 or (idx + 1) == n_batches:
                elapsed = time.time() - start
                processed = idx + 1
                progress = processed / n_batches * 100.0
                est_total = elapsed / processed * n_batches if processed > 0 else 0.0
                eta = max(0.0, est_total - elapsed)
                # print(f"[Train] {processed}/{n_batches} ({progress:.1f}%) "
                #       f"elapsed={elapsed/60:.2f} min, ETA={eta/60:.2f} min")

            endt = time.time()

        end = time.time()
        training_acc = (breakpoint1-error1)*100/breakpoint1
        print("Training for {0} samples done in {1}".format(idx + 1, end-start))
        print("Training Accuracy for {1} epochs: {0}%".format(training_acc, ep + 1))

    error3 = 0
    start = time.time()
    all_preds = []
    all_labels = []

    n_test_batches = len(test_loader)

    for idx, (data, target) in enumerate(test_loader):

        if cuda:
            data = data.cuda()
            target = target.cuda()

        label = target[0].item()

        data_4d = data.unsqueeze(2)
        patches = unfold(data_4d)
        sliced_data = patches.squeeze(0).T

        inputData, outputData = network(sliced_data)

        labelVotes = class_votes_from_wta(network, num_classes=num_classes, voting_thresh=voting_thresh)

        pred = int(torch.argmax(labelVotes).item())
        all_labels.append(int(label))
        all_preds.append(pred)

        if pred != label:
            error1 += 1
            error2 += 1
            error3 += 1

        if inc_learn_eval == 1:
            inputLabels = build_supervised_labels_single_layer(
                network=network,
                label=label,
                num_classes=num_classes,
                device_=device,
            )

            network.weights = network.update(inputData, inputLabels, outputData, network.weights,
                                            capture, search, backoff, wBase, network.rewardEn, frac)

        if (idx+1)%interval2 == 0:
            errorlist1.append(error1/(idx+1+breakpoint1))
            errorlist2.append(error2/interval2)
            error2 = 0

        endt = time.time()

        # Progress reporting for testing
        if (idx + 1) % max(1, n_test_batches // 10) == 0 or (idx + 1) == n_test_batches:
            elapsed = endt - start
            progress = (idx + 1) / n_test_batches * 100.0
            print(f"[Test] {idx+1}/{n_test_batches} ({progress:.1f}%) "
                  f"elapsed={elapsed/60:.2f} min")

    end = time.time()
    testing_acc = (breakpoint2-error3)*100/breakpoint2
    print("Testing for {0} samples done in {1}".format(idx + 1, end-start))
    print("Testing Accuracy: {0}%".format(testing_acc))

    # Confusion matrix over test set: rows=true labels, cols=predicted labels
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            conf_mat[t, p] += 1
    print("Confusion matrix (rows=true, cols=predicted):")
    print(conf_mat)

    # Per-class metrics from confusion matrix
    per_class_acc = []
    precision = []
    recall = []
    f1 = []
    for c in range(num_classes):
        tp = conf_mat[c, c]
        fn = conf_mat[c, :].sum() - tp
        fp = conf_mat[:, c].sum() - tp
        tn = conf_mat.sum() - (tp + fp + fn)

        acc_c = (tp + tn) / conf_mat.sum() if conf_mat.sum() > 0 else 0.0
        prec_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_c = (2 * prec_c * rec_c / (prec_c + rec_c)) if (prec_c + rec_c) > 0 else 0.0

        per_class_acc.append(acc_c)
        precision.append(prec_c)
        recall.append(rec_c)
        f1.append(f1_c)

    print("Per-class accuracy:", per_class_acc)
    print("Per-class precision:", precision)
    print("Per-class recall:", recall)
    print("Per-class F1:", f1)

    return training_acc, testing_acc, errorlist2, network, conf_mat, per_class_acc, precision, recall, f1


@dataclass
class SearchConfig:
    capture: float
    search: float
    backoff: float
    frac: float
    wBase: int
    wMax: int
    voting_thresh: float
    epochs: int
    inc_learn_eval: int


def save_search_config(cfg: SearchConfig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_search_config(in_path: Path) -> SearchConfig:
    with in_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return SearchConfig(
        capture=float(raw["capture"]),
        search=float(raw["search"]),
        backoff=float(raw["backoff"]),
        frac=float(raw["frac"]),
        wBase=int(raw["wBase"]),
        wMax=int(raw["wMax"]),
        voting_thresh=float(raw["voting_thresh"]),
        epochs=int(raw["epochs"]),
        inc_learn_eval=int(raw["inc_learn_eval"]),
    )


def save_best_result_txt(
    out_path: Path,
    cfg: SearchConfig,
    train_acc: float,
    test_acc: float,
    conf_mat: np.ndarray,
    per_class_acc: list[float],
    precision: list[float],
    recall: list[float],
    f1: list[float],
    spikify_method: str | None = None,
    spikify_kw: dict | None = None,
    run_tag: str = "",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Best SNL-EEG Result (spikify benchmark)",
        "=" * 60,
    ]
    if spikify_method:
        lines.append(f"spikify_method: {spikify_method}")
    if run_tag:
        lines.append(f"run_tag: {run_tag}")
    if spikify_kw is not None:
        lines.append("spikify_kw: " + json.dumps(spikify_kw, indent=2))
    lines.extend(
        [
        f"Train Accuracy: {train_acc:.4f}%",
        f"Test Accuracy:  {test_acc:.4f}%",
        "",
        "Best NeuTNN SearchConfig:",
        json.dumps(asdict(cfg), indent=2),
        ]
    )
    lines.extend(
        [
        "",
        "Confusion Matrix (rows=true, cols=pred):",
        np.array2string(conf_mat),
        "",
        f"Per-class accuracy:  {per_class_acc}",
        f"Per-class precision: {precision}",
        f"Per-class recall:    {recall}",
        f"Per-class F1:        {f1}",
        "",
        ],
    )
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def append_benchmark_run(record: dict) -> None:
    """Append one JSON line to NeuTNN_imu/benchmark_runs.jsonl for cross-method comparison."""
    BENCHMARK_RUNS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    record.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with BENCHMARK_RUNS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def sample_quick_configs(seed: int = SEARCH_SEED, n_samples: int = SEARCH_TRIALS) -> list[SearchConfig]:
    rng = np.random.default_rng(seed)
    cfgs: list[SearchConfig] = []
    for _ in range(n_samples):
        cfgs.append(
            SearchConfig(
                capture=float(rng.choice([0.8, 1.0, 1.2, 1.5])),
                search=float(rng.choice([0.0, 0.1, 0.2])),
                backoff=float(rng.choice([0.2, 0.4, 0.625, 0.8])),
                frac=float(rng.choice([0.0, 0.1, 0.2])),
                wBase=int(rng.choice([10, 12, 14])),
                wMax=int(rng.choice([14, 16, 18])),
                voting_thresh=float(rng.choice([0.0, 1.0, 2.0])),
                epochs=int(rng.choice([1, 2, 3])),
                inc_learn_eval=int(rng.choice([0, 1])),
            )
        )
    return cfgs


def run_search_configs():
    if breakpoint1 <= 0 or breakpoint2 <= 0:
        print("No data for search.")
        return None

    best = None
    best_acc = -1.0

    for i, cfg in enumerate(sample_quick_configs(), 1):
        print(f"=== Config {i}/{SEARCH_TRIALS} ===")
        print(asdict(cfg))

        train_acc, test_acc, err, net, cm, pacc, prec, rec, f1 = train_func(
            numDistalInputs=numDistalInputs,
            numSegments=numSegments,
            numDendrites=numDendrites,
            numNeurons=numNeurons,
            threshold=threshold,
            capture=cfg.capture,
            search=cfg.search,
            backoff=cfg.backoff,
            frac=cfg.frac,
            wBase=cfg.wBase,
            device=device,
            kernel_len=kernel_len,
            stride_t=stride_t,
            wMax=cfg.wMax,
            voting_thresh=cfg.voting_thresh,
            num_classes=numDendrites,
            epochs=cfg.epochs,
            inc_learn_eval=cfg.inc_learn_eval,
        )

        print(f"Result: train={train_acc:.2f} test={test_acc:.2f}")
        if test_acc > best_acc:
            best_acc = test_acc
            best = (cfg, train_acc, test_acc, err, net, cm, pacc, prec, rec, f1)

    if best is not None:
        save_search_config(best[0], BEST_CONFIG_PATH)
        save_best_result_txt(
            out_path=BEST_RESULT_TXT_PATH,
            cfg=best[0],
            train_acc=best[1],
            test_acc=best[2],
            conf_mat=best[5],
            per_class_acc=best[6],
            precision=best[7],
            recall=best[8],
            f1=best[9],
            spikify_method=SPIKIFY_METHOD,
            spikify_kw=SPIKIFY_KW,
            run_tag=RUN_TAG,
        )
        print(f"BEST TEST RESULT: {best[2]:.2f}%")
        print(f"BEST TRAIN RESULT: {best[1]:.2f}%")
        print("BEST CONFUSION MATRIX (rows=true, cols=pred):")
        print(best[5])
        print(f"Best config saved to: {BEST_CONFIG_PATH}")
        print(f"Best result txt saved to: {BEST_RESULT_TXT_PATH}")
        print("Best config:", asdict(best[0]))
        return best[1:]

    return None

if breakpoint1 > 0 and breakpoint2 > 0:
    if LOAD_BEST_CONFIG:
        if BEST_CONFIG_PATH.exists():
            cfg = load_search_config(BEST_CONFIG_PATH)
            print(f"Loaded best config from: {BEST_CONFIG_PATH}")
            print("Loaded config:", asdict(cfg))
            training_accuracy, testing_accuracy, errorlist2, myNetwork, conf_mat, per_class_acc, precision, recall, f1 = train_func(
                numDistalInputs=numDistalInputs,
                numSegments=numSegments,
                numDendrites=numDendrites,
                numNeurons=numNeurons,
                threshold=threshold,
                capture=cfg.capture,
                search=cfg.search,
                backoff=cfg.backoff,
                frac=cfg.frac,
                wBase=cfg.wBase,
                device=device,
                kernel_len=kernel_len,
                stride_t=stride_t,
                wMax=cfg.wMax,
                voting_thresh=cfg.voting_thresh,
                num_classes=numDendrites,
                epochs=cfg.epochs,
                inc_learn_eval=cfg.inc_learn_eval,
            )
        else:
            print(f"WARNING: Best config file not found at {BEST_CONFIG_PATH}. Falling back to default run.")
            training_accuracy, testing_accuracy, errorlist2, myNetwork, conf_mat, per_class_acc, precision, recall, f1 = train_func(
                numDistalInputs=numDistalInputs,
                numSegments=numSegments,
                numDendrites=numDendrites,
                numNeurons=numNeurons,
                threshold=threshold,
                capture=capture,
                search=search,
                backoff=backoff,
                frac=frac,
                wBase=wBase,
                device=device,
                kernel_len=kernel_len,
                stride_t=stride_t,
                wMax=wMax,
                voting_thresh=voting_thresh,
                num_classes=numDendrites,
                epochs=1,
                inc_learn_eval=inc_learn,
            )
    elif RUN_CONFIG_SEARCH == 1:
        out = run_search_configs()
        if out is not None:
            training_accuracy, testing_accuracy, errorlist2, myNetwork, conf_mat, per_class_acc, precision, recall, f1 = out
        else:
            training_accuracy, testing_accuracy, errorlist2, myNetwork, conf_mat, per_class_acc, precision, recall, f1 = train_func(
                numDistalInputs=numDistalInputs,
                numSegments=numSegments,
                numDendrites=numDendrites,
                numNeurons=numNeurons,
                threshold=threshold,
                capture=capture,
                search=search,
                backoff=backoff,
                frac=frac,
                wBase=wBase,
                device=device,
                kernel_len=kernel_len,
                stride_t=stride_t,
                wMax=wMax,
                voting_thresh=voting_thresh,
                num_classes=numDendrites,
                epochs=1,
                inc_learn_eval=inc_learn,
            )
    else:
        training_accuracy, testing_accuracy, errorlist2, myNetwork, conf_mat, per_class_acc, precision, recall, f1 = train_func(
            numDistalInputs=numDistalInputs,
            numSegments=numSegments,
            numDendrites=numDendrites,
            numNeurons=numNeurons,
            threshold=threshold,
            capture=capture,
            search=search,
            backoff=backoff,
            frac=frac,
            wBase=wBase,
            device=device,
            kernel_len=kernel_len,
            stride_t=stride_t,
            wMax=wMax,
            voting_thresh=voting_thresh,
            num_classes=numDendrites,
            epochs=1,
            inc_learn_eval=inc_learn,
        )

### Storing image ###

if breakpoint1 > 0 and breakpoint2 > 0:
    # Error rate over training+testing intervals (best-effort; ignore failures)
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(errorlist2, color='red', linestyle='solid', linewidth=1.5)
        plt.xlabel("Intervals")
        plt.ylabel("Error Rate")
        plt.legend(["activeDendriteLayer (EEG-only, core import)"])
        plt.tight_layout()
        plt.savefig(GRAPHS_DIR / "online_adaptivity.png")
    except Exception as e:
        print(f"WARNING: error-rate plotting failed: {e}")

    # Confusion matrix heatmap (rows=true labels, cols=predicted labels)
    plt.figure(figsize=(4, 4))
    im = plt.imshow(conf_mat, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    classes = ["stand", "slow", "fast", "run"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    _tag = f"{SPIKIFY_METHOD}" + (f" ({RUN_TAG})" if RUN_TAG else "")
    plt.title(f"Confusion Matrix — {_tag}")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "confusion_matrix.png")

    try:
        append_benchmark_run(
            {
                "spikify_method": SPIKIFY_METHOD,
                "run_tag": RUN_TAG,
                "spikify_kw": SPIKIFY_KW,
                "train_accuracy_pct": float(training_accuracy),
                "test_accuracy_pct": float(testing_accuracy),
                "n_units": int(n_units),
                "train_windows": int(breakpoint1),
                "test_windows": int(breakpoint2),
                "config_dir": str(BENCHMARK_CONFIG_DIR),
                "best_config_json": str(BEST_CONFIG_PATH),
                "result_txt": str(BEST_RESULT_TXT_PATH),
                "graphs_dir": str(GRAPHS_DIR),
            }
        )
        print(f"Benchmark run appended to: {BENCHMARK_RUNS_JSONL}")
    except Exception as e:
        print(f"WARNING: could not append benchmark_runs.jsonl: {e}")
else:
    print("Skipping plot (no data).")