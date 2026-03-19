"""
 Author: Shanmuga Venkatachalam
 Adapted for IMU data with delta modulation encoding.
"""

### Importing libraries ###

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
from pathlib import Path

# IMU channel role helper
from read_imu_dat import channel_roles

# Delta modulation encoder (parent folder)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from encoders import DeltaModulationEncoder


### Enabling CUDA support for GPU ###
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### Training Parameter Definitions ###
inc_learn = 1
interval1   = 100
interval2   = 100

### IMU data paths: sourcedata/sub-**/ses-**/imu/ ###
SOURCEDATA_ROOT = Path(__file__).resolve().parent / "sourcedata"

### Model Parameter Definitions (IMU with delta modulation) ###
capture = 1
search  = 0
backoff = 0.625
frac = 0
wBase = 12
wMax = 16

# Temporal receptive field for spike raster (n_units, window_len)
kernel_len = 5
stride_t = 2
n_imu_channels = 6   # use head accelerometer + head gyroscope (HaccX/Y/Z + HgyroX/Y/Z)
n_thresholds = 3
THRESHOLD_BASE_ACC = 0.0112
THRESHOLD_BASE_GYRO = 0.17721262
n_units = n_imu_channels * n_thresholds * 2  # delta mod: pos+neg per threshold
numDistalInputs = kernel_len * n_units
numSegments = 36
numDendrites = 4     # locomotion classes: standing, slow walk, fast walk, running
window_len = 64    # 512hz = 4s window at ~128 Hz
numNeurons = (window_len - kernel_len) // stride_t + 1
threshold = 0
voting_thresh = 0
NUM_CLASSES = 4
MAX_TRAIN_WINDOWS = 1500
MAX_TEST_WINDOWS = 600
SEARCH_TRIALS = 12
SEARCH_SEED = 42
RUN_CONFIG_SEARCH = 1
BEST_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "best_snl_imu_config.json"
BEST_RESULT_TXT_PATH = Path(__file__).resolve().parent / "config" / "best_snl_imu_result.txt"


### Active Dendrite Layer ###

    # Implements a single NeuTNN layer with multiple neurons, dendrites, and segments.
    # Segments are assumed to be distal, while the proximal input is simply used as enable during weight update.
    # Capable of online continuous learning via STDP/R-STDP.
    
    # Args: numDistalInputs - Number of distal inputs to a single neuron. Within a neuron, the inputs are shared across all segments.
    #       numSegments     - Number of segments per dendrite per neuron.
    #       numDendrites    - Number of dendrites per neuron.
    #       numNeurons      - Number of neurons in the layer.
    #       threshold       - Excitation threshold for neuron. An output spike is generated when neuron's body potential reaches
    #                         this threshold.
    #       wMax            - Maximum value for synaptic weights.
    #       wInit           - Type of initialization for synaptic weights. Supports 'zero', 'uniform', and 'normal' initializations.
    #       rewardEn        - Enable reinforcement for STDP (R-STDP).
    #       device          - Device type (e.g. cpu, cuda).
    #
    # Input: data           - Input data to the layer of shape [numNeurons, numDistalInputs].
    #
    # __call__ is the forward function whereas "update" performs STDP update in either unsupervised or supervised fashion.
class activeDendriteLayer(nn.Module):
    def __init__(self,
                 numDistalInputs: int,
                 numSegments: int,
                 numDendrites: int,
                 numNeurons: int,
                 threshold: int,
                 wMax: int = 8,
                 wInit: str = "uniform",
                 rewardEn: bool = True,
                 device="cpu"
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

        self.totalSegments = self.numNeurons*self.numDendrites*self.numSegments

        if self.wInit       == "zero":
            self.weights   = nn.Parameter(torch.zeros(self.totalSegments,
                                                        self.numDistalInputs), requires_grad=False)

        elif self.wInit     == "uniform":
            self.weights   = nn.Parameter(torch.randint(low = 0, high = self.wMax + 1,
                                            size=(self.totalSegments,
                                                self.numDistalInputs)).type(torch.FloatTensor),
                                            requires_grad=False)

        elif self.wInit     == "normal":
            self.weights   = nn.Parameter(torch.round(( (self.wMax+1)/2 + \
                                                        torch.randn(self.totalSegments, self.numDistalInputs))\
                                                        .clamp_(0,self.wMax)), requires_grad=False)


    ########################################### Inference function ###########################################
    def __call__(self, data):
        # Data is of shape [self.numNeurons, self.numDistalInputs]

        inputData = data.unsqueeze(1).repeat(1,self.numDendrites*self.numSegments,1)\
                        .reshape(self.totalSegments,self.numDistalInputs)

        # Applying weights
        weightedInput = inputData * self.weights

        # Weighted input summation - SNL
        summedOutput = torch.sum(weightedInput, dim=1)

        zeroCondition = summedOutput < self.threshold
        summedOutput[zeroCondition] = 0

        # Winner Take All inhibition across segments
        reshapedSummedOutput = summedOutput.reshape(-1,self.numSegments)

        self.wtaOutput = torch.zeros(self.numNeurons*self.numDendrites, self.numSegments,
                                     dtype=reshapedSummedOutput.dtype, device=self.device)

        maxval, indices = torch.max(reshapedSummedOutput, dim=1)

        self.wtaOutput.scatter_(1, indices.unsqueeze(1), maxval.unsqueeze(1))

        outputData = (self.wtaOutput>0).reshape(-1).unsqueeze(1).repeat(1,self.numDistalInputs)

        return inputData, outputData


    ########################################### Update function ###########################################
    def update(self, inputData, inputLabels, outputData, weights, capture, search, backoff, wBase, rewardEn, frac=0.25):

        if rewardEn:
            captureVal = (inputData == 1) * (outputData == 1) * (inputLabels == 1) * capture + (inputData == 0) * (outputData == 1) * (inputLabels == 0) * capture * frac
            backoffVal = (inputData == 0) * (outputData == 1) * (inputLabels == 1) * backoff + (inputData == 1) * (outputData == 1) * (inputLabels == 0) * backoff * frac
            searchCondition  = (inputData == 1) * (outputData == 0) * (inputLabels == 1)
        else:
            captureVal = (inputData == 1) * (outputData == 1) * capture
            backoffVal = (inputData == 0) * (outputData == 1) * backoff
            searchCondition  = (inputData == 1) * (outputData == 0)

        # Capture and Backoff
        weights += captureVal - backoffVal

        # Search
        weights[searchCondition] = torch.maximum(weights[searchCondition],
                                                 torch.minimum(weights[searchCondition] + search,
                                                               torch.tensor([wBase], device=self.device)))

        return weights.clamp_(0, self.wMax)

### IMU dataset: load preprocessed signals from imu_dataset_clean and build spike windows ###

IMU_DATASET_ROOT = Path(__file__).resolve().parent / "imu_dataset_clean"
IMU_MANIFEST_PATH = IMU_DATASET_ROOT / "manifest.json"
try:
    with IMU_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        _imu_manifest = json.load(f)
    IMU_MANIFEST_BY_SOURCE = {
        Path(entry["source"]).resolve(): entry for entry in _imu_manifest
    }
except FileNotFoundError:
    _imu_manifest = []
    IMU_MANIFEST_BY_SOURCE = {}


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


class IMUSlidingWindowDataset(Dataset):
    """
    Build spike windows from preprocessed IMU (imu_dataset_clean) using delta modulation.

    - Uses only head IMU channels: HaccX/Y/Z and HgyroX/Y/Z.
    - Uses sliding windows of fixed length (window_len samples) over time.
    - Labels are locomotion conditions derived from ses-YY in the original source path.
    """

    def __init__(
        self,
        entries: list[dict],
        imu_dataset_root: Path,
        window_len: int,
        stride: int,
        n_thresholds: int = n_thresholds,
        threshold_base_acc: float = THRESHOLD_BASE_ACC,
        threshold_base_gyro: float = THRESHOLD_BASE_GYRO,
        scale: float = 2.0,
        max_windows: int | None = None,
        num_classes: int = NUM_CLASSES,
    ):
        self.samples: list[tuple[np.ndarray, int]] = []
        self.n_units: int = 0
        self.window_len: int = window_len

        imu_dataset_root = Path(imu_dataset_root).resolve()

        max_per_class = None
        if max_windows is not None and max_windows > 0:
            max_per_class = max(1, int(max_windows // max(1, num_classes)))
        per_class_count = {c: 0 for c in range(num_classes)}

        enc_acc = DeltaModulationEncoder(
            n_thresholds=n_thresholds,
            threshold_base=threshold_base_acc,
            scale=scale,
        )
        enc_gyro = DeltaModulationEncoder(
            n_thresholds=n_thresholds,
            threshold_base=threshold_base_gyro,
            scale=scale,
        )

        for entry in entries:
            source = entry["source"]
            # Only use ERP IMU sessions by default
            if "_task-ERP_imu" not in source:
                continue

            label = locomotion_label_from_source(source)
            if label is None:
                # Skip training session (ses-01) or unsupported sessions
                continue
            npy_rel = entry["npy"]
            ch_names = entry["ch_names"]
            sfreq = entry.get("sfreq") or 128.0

            data_path = imu_dataset_root / npy_rel
            if not data_path.exists():
                continue

            data = np.load(data_path)  # (n_channels, n_times)

            roles = channel_roles(ch_names)
            head_acc = [i for i, n in roles["acc"] if n.upper().startswith("H")]
            head_gyro = [i for i, n in roles["gyro"] if n.upper().startswith("H")]

            # Use 6 head IMU channels: HaccX/Y/Z + HgyroX/Y/Z.
            if len(head_acc) < 3 or len(head_gyro) < 3:
                continue

            data_acc = data[head_acc[:3]].astype(np.float64)    # (3, n_times)
            data_gyro = data[head_gyro[:3]].astype(np.float64)  # (3, n_times)

            spikes_acc = enc_acc.encode(data_acc)
            spikes_gyro = enc_gyro.encode(data_gyro)
            spikes = np.concatenate([spikes_acc, spikes_gyro], axis=0)  # (n_units, n_times-1)
            n_units_local, n_t = spikes.shape

            win_len = int(window_len)
            step = int(stride)
            if win_len <= 0 or step <= 0 or win_len > n_t:
                continue

            start = 0
            while start + win_len <= n_t:
                if max_per_class is not None and per_class_count[label] >= max_per_class:
                    break
                end = start + win_len
                win = spikes[:, start:end]
                self.samples.append((win, label))
                per_class_count[label] += 1
                start += step

            if self.samples and self.n_units == 0:
                self.n_units = n_units_local

        if self.samples:
            self.window_len = self.samples[0][0].shape[1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        win, label = self.samples[idx]
        return torch.from_numpy(win).float(), torch.tensor(label, dtype=torch.long)


def build_train_test_datasets(
    imu_dataset_root: Path,
    manifest: list[dict],
    train_ratio: float = 0.8,
    window_len: int = window_len,
    stride: int = window_len // 2,
    n_thresholds: int = n_thresholds,
    threshold_base_acc: float = THRESHOLD_BASE_ACC,
    threshold_base_gyro: float = THRESHOLD_BASE_GYRO,
    scale: float = 2.0,
    max_train_windows: int | None = MAX_TRAIN_WINDOWS,
    max_test_windows: int | None = MAX_TEST_WINDOWS,
    num_classes: int = NUM_CLASSES,
    seed: int = 42,
):
    """Split manifest entries into train/test groups and build sliding-window datasets."""
    if not manifest:
        return None, None

    entries = [e for e in manifest if "_task-ERP_imu" in e.get("source", "")]
    if not entries:
        return None, None

    entries = sorted(entries, key=lambda e: e["source"])
    train_entries, test_entries = split_entries_by_subject(
        entries,
        train_ratio=train_ratio,
        seed=seed,
    )

    print(f"Entry split: train={len(train_entries)}, test={len(test_entries)}")

    train_ds = IMUSlidingWindowDataset(
        entries=train_entries,
        imu_dataset_root=imu_dataset_root,
        window_len=window_len,
        stride=stride,
        n_thresholds=n_thresholds,
        threshold_base_acc=threshold_base_acc,
        threshold_base_gyro=threshold_base_gyro,
        scale=scale,
        max_windows=max_train_windows,
        num_classes=num_classes,
    )
    test_ds = IMUSlidingWindowDataset(
        entries=test_entries,
        imu_dataset_root=imu_dataset_root,
        window_len=window_len,
        stride=stride,
        n_thresholds=n_thresholds,
        threshold_base_acc=threshold_base_acc,
        threshold_base_gyro=threshold_base_gyro,
        scale=scale,
        max_windows=max_test_windows,
        num_classes=num_classes,
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


if _imu_manifest:
    train_dataset, test_dataset = build_train_test_datasets(
        imu_dataset_root=IMU_DATASET_ROOT,
        manifest=_imu_manifest,
        train_ratio=0.8,
        window_len=window_len,
        stride=window_len // 2,
        n_thresholds=n_thresholds,
        threshold_base_acc=THRESHOLD_BASE_ACC,
        threshold_base_gyro=THRESHOLD_BASE_GYRO,
        scale=2.0,
        max_train_windows=MAX_TRAIN_WINDOWS,
        max_test_windows=MAX_TEST_WINDOWS,
        num_classes=NUM_CLASSES,
        seed=42,
    )
    breakpoint1 = len(train_dataset) if train_dataset is not None else 0
    breakpoint2 = len(test_dataset) if test_dataset is not None else 0
    print(f"IMU train windows: {breakpoint1}, test windows: {breakpoint2}")
else:
    print("WARNING: No IMU manifest found. Make sure imu_dataset_clean/manifest.json exists.")
    train_dataset, test_dataset = None, None
    breakpoint1, breakpoint2 = 0, 0

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) if breakpoint1 > 0 else []
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) if breakpoint2 > 0 else []


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
            # print(f"[Test] {idx+1}/{n_test_batches} ({progress:.1f}%) "
            #       f"elapsed={elapsed/60:.2f} min")

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


def load_search_config(in_path: Path) -> SearchConfig | None:
    if not in_path.exists():
        print(f"Config file not found: {in_path}")
        return None

    try:
        with in_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return SearchConfig(**payload)
    except Exception as e:
        print(f"Failed to load config from {in_path}: {e}")
        return None


def build_default_config() -> SearchConfig:
    return SearchConfig(
        capture=float(capture),
        search=float(search),
        backoff=float(backoff),
        frac=float(frac),
        wBase=int(wBase),
        wMax=int(wMax),
        voting_thresh=float(voting_thresh),
        epochs=1,
        inc_learn_eval=int(inc_learn),
    )


def run_with_config(cfg: SearchConfig):
    print("Running config:", asdict(cfg))
    return train_func(
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
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Best SNL-IMU Result",
        "=" * 60,
        f"Train Accuracy: {train_acc:.4f}%",
        f"Test Accuracy:  {test_acc:.4f}%",
        "",
        "Best Config:",
        json.dumps(asdict(cfg), indent=2),
        "",
        "Confusion Matrix (rows=true, cols=pred):",
        np.array2string(conf_mat),
        "",
        f"Per-class accuracy:  {per_class_acc}",
        f"Per-class precision: {precision}",
        f"Per-class recall:    {recall}",
        f"Per-class F1:        {f1}",
        "",
    ]
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


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


parser = argparse.ArgumentParser(description="SNL IMU training and config loader")
parser.add_argument("--load-best-config", action="store_true", help="Load config JSON and run one config")
parser.add_argument("--config-path", type=str, default=str(BEST_CONFIG_PATH), help="Path to config JSON")
parser.add_argument("--save-current-config", action="store_true", help="Save current default config to --config-path")
args, _ = parser.parse_known_args()

cfg_path = Path(args.config_path)
if args.save_current_config:
    default_cfg = build_default_config()
    save_search_config(default_cfg, cfg_path)
    print(f"Saved current default config to: {cfg_path}")

if breakpoint1 > 0 and breakpoint2 > 0:
    if args.load_best_config:
        loaded_cfg = load_search_config(cfg_path)
        if loaded_cfg is not None:
            training_accuracy, testing_accuracy, errorlist2, myNetwork, conf_mat, per_class_acc, precision, recall, f1 = run_with_config(loaded_cfg)
        else:
            print("Falling back to existing run flow because config could not be loaded.")
            if RUN_CONFIG_SEARCH == 1:
                out = run_search_configs()
                if out is not None:
                    training_accuracy, testing_accuracy, errorlist2, myNetwork, conf_mat, per_class_acc, precision, recall, f1 = out
                else:
                    training_accuracy, testing_accuracy, errorlist2, myNetwork, conf_mat, per_class_acc, precision, recall, f1 = run_with_config(build_default_config())
            else:
                training_accuracy, testing_accuracy, errorlist2, myNetwork, conf_mat, per_class_acc, precision, recall, f1 = run_with_config(build_default_config())
    elif RUN_CONFIG_SEARCH == 1:
        out = run_search_configs()
        if out is not None:
            training_accuracy, testing_accuracy, errorlist2, myNetwork, conf_mat, per_class_acc, precision, recall, f1 = out
        else:
            training_accuracy, testing_accuracy, errorlist2, myNetwork, conf_mat, per_class_acc, precision, recall, f1 = run_with_config(build_default_config())
    else:
        training_accuracy, testing_accuracy, errorlist2, myNetwork, conf_mat, per_class_acc, precision, recall, f1 = run_with_config(build_default_config())

### Storing image ###

if breakpoint1 > 0 and breakpoint2 > 0:
    # Error rate over training+testing intervals (best-effort; ignore failures)
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(errorlist2, color='red', linestyle='solid', linewidth=1.5)
        plt.xlabel("Intervals")
        plt.ylabel("Error Rate")
        plt.legend(["activeDendriteLayer (IMU)"])
        plt.tight_layout()
        plt.savefig("online_adaptivity_imu.png")
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
    plt.title("IMU Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix_imu.png")
else:
    print("Skipping plot (no data).")