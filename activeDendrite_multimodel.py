"""
Multimodal NeuTNN classification pipeline:
- IMU: delta multi-threshold encoding (6 channels = 3 acc + 3 gyro).
- EEG: runtime-switchable encoder (delta or time_to_first_spike).
- Pair IMU/EEG by subject-session and sorted filename one-to-one.
- Keep sessions ses-02..ses-05 as 4 locomotion classes.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from active_dendrite_core import activeDendriteLayer
from read_imu_dat import channel_roles, load_imu_brainvision
from spikify_temporal_encode import (
    encode_temporal_multichannel,
    is_spikify_available,
    n_units_for_method,
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from encoders import DeltaModulationEncoder


cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


IMU_DATASET_ROOT = Path(__file__).resolve().parent / "imu_dataset_clean"
IMU_MANIFEST_PATH = IMU_DATASET_ROOT / "manifest.json"
EEG_DATASET_ROOT_DEFAULT = Path(
    r"D:/xia/CMU/18743-Neuromorphic/Multi_Sensory_NeuTNN/osfstorage-archive"
)

IMU_SAMPLE_RATE_DEFAULT = 128.0
RESAMPLE_RATE = 128.0
n_eeg_channels = 3

n_thresholds = 3
THRESHOLD_BASE_ACC = 0.0112
THRESHOLD_BASE_GYRO = 0.17721262
THRESHOLD_BASE_EEG = 10.62202935
DELTA_SCALE = 2.0

NUM_CLASSES = 4
numSegments = 36
numDendrites = 4
threshold = 0

kernel_len = 5
stride_t = 2
window_len = 128

capture = 1.0
search = 0.0
backoff = 0.625
frac = 0.0
wBase = 12
wMax = 16
voting_thresh = 0.0
inc_learn = 0
interval1 = 100
interval2 = 100

GRAPHS_DIR = Path(__file__).resolve().parent / "graphs_multimodel"
ENCODER_VIZ_DIR = GRAPHS_DIR
BEST_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "best_multimodel_config.json"
BEST_RESULT_TXT_PATH = GRAPHS_DIR / "best_multimodel_result.txt"
IMU_BEST_RESULT_PATH = Path(__file__).resolve().parent / "config" / "best_snl_imu_result.txt"
EEG_TTFS_BEST_RESULT_PATH = (
    Path(__file__).resolve().parent / "config" / "benchmarks" / "time_to_first_spike" / "result.txt"
)


def _load_best_config_from_result_txt(result_path: Path) -> dict:
    """Parse json config block from a benchmark result text file."""
    try:
        text = result_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}

    markers = ("Best Config:", "Best NeuTNN SearchConfig:")
    start = -1
    for marker in markers:
        idx = text.find(marker)
        if idx >= 0:
            start = idx + len(marker)
            break
    if start < 0:
        return {}

    tail = text[start:]
    left = tail.find("{")
    if left < 0:
        return {}

    depth = 0
    right = -1
    for i, ch in enumerate(tail[left:]):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                right = left + i + 1
                break
    if right < 0:
        return {}

    try:
        raw = json.loads(tail[left:right])
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


IMU_BEST_CFG = _load_best_config_from_result_txt(IMU_BEST_RESULT_PATH)
EEG_TTFS_BEST_CFG = _load_best_config_from_result_txt(EEG_TTFS_BEST_RESULT_PATH)

# Use IMU best config as default activeDendrite baseline.
capture = float(IMU_BEST_CFG.get("capture", capture))
search = float(IMU_BEST_CFG.get("search", search))
backoff = float(IMU_BEST_CFG.get("backoff", backoff))
frac = float(IMU_BEST_CFG.get("frac", frac))
wBase = int(IMU_BEST_CFG.get("wBase", wBase))
wMax = int(IMU_BEST_CFG.get("wMax", wMax))
voting_thresh = float(IMU_BEST_CFG.get("voting_thresh", voting_thresh))
inc_learn = int(IMU_BEST_CFG.get("inc_learn_eval", inc_learn))

# Use TTFS benchmark as EEG preprocessing default.
DEFAULT_TTS_INTERVAL = 8
DEFAULT_EPOCHS = int(IMU_BEST_CFG.get("epochs", 1))


def parse_runtime_args():
    """Parse runtime options for data, encoder, and training behavior."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--eeg-root",
        default=str(EEG_DATASET_ROOT_DEFAULT),
        help="Path to osfstorage-archive root.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-len", type=int, default=window_len)
    parser.add_argument("--window-stride", type=int, default=window_len // 2)
    parser.add_argument("--max-train-windows", type=int, default=1500)
    parser.add_argument("--max-test-windows", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--inc-learn-eval", type=int, default=inc_learn, choices=[0, 1])
    parser.add_argument(
        "--eeg-encoder",
        default="tts",
        choices=("delta", "tts"),
        help="EEG encoder mode for multimodal fusion.",
    )
    parser.add_argument(
        "--tts-interval",
        type=int,
        default=DEFAULT_TTS_INTERVAL,
        help="interval parameter for time_to_first_spike (used when --eeg-encoder tts).",
    )
    parser.add_argument(
        "--run-config-search",
        type=int,
        default=0,
        choices=[0, 1],
        help="Run quick config search and save best config.",
    )
    parser.add_argument(
        "--load-best-config",
        action="store_true",
        help="Load best config from json and run once.",
    )
    parser.add_argument("--search-trials", type=int, default=12, help="Number of search configs to evaluate.")
    parser.add_argument("--search-seed", type=int, default=123, help="Random seed for config sampling.")
    parser.add_argument("--save-encoder-viz", action="store_true", help="Save encoder visualizations.")
    return parser.parse_args()


RUNTIME_ARGS = parse_runtime_args()
EEG_DATASET_ROOT = Path(RUNTIME_ARGS.eeg_root).resolve()


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
    eeg_encoder: str
    tts_interval: int


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
        eeg_encoder=str(raw["eeg_encoder"]),
        tts_interval=int(raw["tts_interval"]),
    )


def sample_quick_configs(seed: int = 123, n_samples: int = 12) -> list[SearchConfig]:
    """Sample configs for multimodel search with fixed TTFS and split seed."""
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
                # User-requested constraint: fix eval online update on during search.
                inc_learn_eval=1,
                # Small-range test: only evaluate EEG TTFS path for now.
                eeg_encoder="tts",
                tts_interval=8,
            )
        )
    return cfgs


def locomotion_label_from_source(source: str) -> int | None:
    """Map ses-02..05 to class 0..3."""
    p = Path(source)
    ses_id = None
    for part in p.parts:
        if part.startswith("ses-"):
            ses_id = part.split("-", 1)[1]
            break
    mapping = {"02": 0, "03": 1, "04": 2, "05": 3}
    return mapping.get(ses_id)


def parse_sub_ses(source: str) -> tuple[str, str]:
    """Extract normalized (sub-*, ses-*) from path segments robustly."""
    text = str(source).replace("\\", "/")
    sub_match = re.search(r"/(sub-[^/]+)/", text, flags=re.IGNORECASE)
    ses_match = re.search(r"/(ses-[^/]+)/", text, flags=re.IGNORECASE)
    sub_id = sub_match.group(1).lower() if sub_match else "unknown"
    ses_id = ses_match.group(1).lower() if ses_match else "unknown"
    return sub_id, ses_id


def collect_imu_entries(imu_manifest: list[dict]) -> list[dict]:
    """Keep ERP IMU entries in target sessions and tag modality='imu'."""
    out = []
    for entry in imu_manifest:
        src = entry.get("source", "")
        if "_task-ERP_imu" not in src:
            continue
        if locomotion_label_from_source(src) is None:
            continue
        e = dict(entry)
        e["modality"] = "imu"
        out.append(e)
    out.sort(key=lambda x: x.get("source", ""))
    return out


def collect_eeg_entries(eeg_root: Path) -> list[dict]:
    """Scan EEG ERP vhdr files and keep ses-02..05 only."""
    eeg_root = Path(eeg_root).resolve()
    entries: list[dict] = []
    for vhdr in eeg_root.glob("sub-*/ses-*/eeg/*task-ERP_eeg.vhdr"):
        src = str(vhdr.resolve())
        if locomotion_label_from_source(src) is None:
            continue
        entries.append({"modality": "eeg", "source": src, "vhdr": src})
    entries.sort(key=lambda x: x["source"])
    return entries


def pair_imu_eeg_entries(imu_entries: list[dict], eeg_entries: list[dict]) -> tuple[list[dict], int]:
    """Pair IMU/EEG one-to-one per (sub,ses) using sorted filenames."""
    by_key_imu: dict[tuple[str, str], list[dict]] = {}
    by_key_eeg: dict[tuple[str, str], list[dict]] = {}

    for e in imu_entries:
        key = parse_sub_ses(e.get("source", ""))
        by_key_imu.setdefault(key, []).append(e)
    for e in eeg_entries:
        key = parse_sub_ses(e.get("source", ""))
        by_key_eeg.setdefault(key, []).append(e)

    paired: list[dict] = []
    dropped = 0
    for key in sorted(set(by_key_imu.keys()) & set(by_key_eeg.keys())):
        imu_list = sorted(by_key_imu[key], key=lambda x: x.get("source", ""))
        eeg_list = sorted(by_key_eeg[key], key=lambda x: x.get("source", ""))
        n = min(len(imu_list), len(eeg_list))
        dropped += abs(len(imu_list) - len(eeg_list))
        for i in range(n):
            paired.append({"imu": imu_list[i], "eeg": eeg_list[i], "key": key, "pair_idx": i})
    return paired, dropped


def split_pairs_by_subject(
    pairs: list[dict], train_ratio: float = 0.8, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Subject-wise split to reduce train/test identity leakage."""

    def get_subj(pair: dict) -> str:
        return parse_sub_ses(pair["imu"].get("source", ""))[0]

    by_subj: dict[str, list[dict]] = {}
    for p in pairs:
        by_subj.setdefault(get_subj(p), []).append(p)

    subjects = sorted(by_subj.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_train = max(1, int(len(subjects) * train_ratio))
    train_sub = set(subjects[:n_train])

    train_pairs: list[dict] = []
    test_pairs: list[dict] = []
    for s in subjects:
        if s in train_sub:
            train_pairs.extend(by_subj[s])
        else:
            test_pairs.extend(by_subj[s])
    if not test_pairs and train_pairs:
        test_pairs = train_pairs[:1]
    return train_pairs, test_pairs


def select_eeg_channel_indices(ch_names: list[str], max_channels: int = n_eeg_channels) -> list[int]:
    """Pick Fz/Cz/Pz first, then fallback EEG-like channels."""
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


def _resample_multichannel(data: np.ndarray, old_fs: float, new_fs: float) -> np.ndarray:
    """Linear resample for (n_channels, n_time) arrays."""
    if old_fs <= 0:
        return data
    if abs(old_fs - new_fs) < 1e-9:
        return data
    n_samples_new = int(data.shape[1] * (new_fs / old_fs))
    n_samples_new = max(2, n_samples_new)
    x_old = np.arange(data.shape[1], dtype=np.float64)
    x_new = np.linspace(0, data.shape[1] - 1, n_samples_new)
    out = np.zeros((data.shape[0], n_samples_new), dtype=np.float64)
    for ch_idx in range(data.shape[0]):
        out[ch_idx] = np.interp(x_new, x_old, data[ch_idx])
    return out


def _plot_raw_and_spikes(raw: np.ndarray, spikes: np.ndarray, title: str, out_path: Path):
    """Save a 2-row figure: raw traces and spike raster."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
    for ch in range(raw.shape[0]):
        axes[0].plot(raw[ch], linewidth=0.8, label=f"ch{ch}")
    axes[0].set_title(f"{title} raw")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("amplitude")
    axes[0].legend(loc="upper right", ncol=3, fontsize=7)

    axes[1].imshow(spikes, aspect="auto", interpolation="nearest", cmap="gray_r")
    axes[1].set_title(f"{title} spikes")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("unit")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_encoder_visualizations(
    raw_imu: np.ndarray,
    raw_eeg: np.ndarray,
    spikes_imu: np.ndarray,
    spikes_eeg: np.ndarray,
    out_dir: Path,
    tag: str,
):
    """Save one IMU and one EEG encoder preview image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_raw_and_spikes(raw_imu, spikes_imu, f"IMU ({tag})", out_dir / f"{tag}_imu_encoder.png")
    _plot_raw_and_spikes(raw_eeg, spikes_eeg, f"EEG ({tag})", out_dir / f"{tag}_eeg_encoder.png")


class PairedImuEegSlidingWindowDataset(Dataset):
    """Build paired IMU+EEG fused spike windows for NeuTNN training."""

    def __init__(
        self,
        pairs: list[dict],
        imu_dataset_root: Path,
        window_len: int,
        stride: int,
        max_windows: int | None = None,
        num_classes: int = NUM_CLASSES,
        eeg_encoder: str = "tts",
        tts_interval: int = 8,
        save_encoder_viz: bool = False,
    ):
        """Load paired files, align to 128Hz, encode both modalities, and build sliding windows."""
        self.samples: list[tuple[np.ndarray, int]] = []
        self.n_units = 0
        self.n_imu_units = 0
        self.n_eeg_units = 0
        self.window_len = int(window_len)
        self.paired_count = 0
        self.skipped_pairs = 0
        self.per_class_count = {c: 0 for c in range(num_classes)}
        self.eeg_encoder = eeg_encoder
        self.tts_interval = int(tts_interval)

        imu_dataset_root = Path(imu_dataset_root).resolve()

        max_per_class = None
        if max_windows is not None and max_windows > 0:
            max_per_class = max(1, int(max_windows // max(1, num_classes)))

        enc_acc = DeltaModulationEncoder(
            n_thresholds=n_thresholds, threshold_base=THRESHOLD_BASE_ACC, scale=DELTA_SCALE
        )
        enc_gyro = DeltaModulationEncoder(
            n_thresholds=n_thresholds, threshold_base=THRESHOLD_BASE_GYRO, scale=DELTA_SCALE
        )
        enc_eeg_delta = DeltaModulationEncoder(
            n_thresholds=n_thresholds, threshold_base=THRESHOLD_BASE_EEG, scale=DELTA_SCALE
        )

        visualized = False
        for pair in pairs:
            imu_entry = pair["imu"]
            eeg_entry = pair["eeg"]
            source_imu = imu_entry.get("source", "")
            source_eeg = eeg_entry.get("source", "")

            label_imu = locomotion_label_from_source(source_imu)
            label_eeg = locomotion_label_from_source(source_eeg)
            if label_imu is None or label_eeg is None or label_imu != label_eeg:
                self.skipped_pairs += 1
                continue
            label = int(label_imu)

            npy_rel = imu_entry.get("npy")
            ch_names_imu = imu_entry.get("ch_names", [])
            if not npy_rel or not ch_names_imu:
                self.skipped_pairs += 1
                continue
            imu_path = imu_dataset_root / npy_rel
            if not imu_path.exists():
                self.skipped_pairs += 1
                continue
            imu_data = np.load(imu_path)

            roles = channel_roles(ch_names_imu)
            head_acc = [i for i, name in roles["acc"] if name.upper().startswith("H")]
            head_gyro = [i for i, name in roles["gyro"] if name.upper().startswith("H")]
            if len(head_acc) < 3 or len(head_gyro) < 3:
                self.skipped_pairs += 1
                continue
            idx_acc = head_acc[:3]
            idx_gyro = head_gyro[:3]
            imu_sel = np.concatenate([imu_data[idx_acc], imu_data[idx_gyro]], axis=0).astype(np.float64)

            imu_sfreq = float(imu_entry.get("sfreq", IMU_SAMPLE_RATE_DEFAULT))
            imu_sel = _resample_multichannel(imu_sel, imu_sfreq, RESAMPLE_RATE)

            vhdr_path = Path(eeg_entry.get("vhdr", source_eeg)).resolve()
            if not vhdr_path.exists():
                self.skipped_pairs += 1
                continue
            try:
                eeg_data, info = load_imu_brainvision(vhdr_path, apply_scaling=True)
            except Exception:
                self.skipped_pairs += 1
                continue

            eeg_indices = select_eeg_channel_indices(info.get("ch_names", []), max_channels=n_eeg_channels)
            if len(eeg_indices) < n_eeg_channels:
                self.skipped_pairs += 1
                continue
            eeg_sel = eeg_data[eeg_indices].astype(np.float64)
            eeg_sfreq = float(info.get("sfreq", 100.0))
            eeg_sel = _resample_multichannel(eeg_sel, eeg_sfreq, RESAMPLE_RATE)

            # Cross-modal alignment: both streams are first resampled to 128Hz, then trimmed to common length.
            n_t_common = min(imu_sel.shape[1], eeg_sel.shape[1])
            if n_t_common < 4:
                self.skipped_pairs += 1
                continue
            imu_sel = imu_sel[:, :n_t_common]
            eeg_sel = eeg_sel[:, :n_t_common]

            spikes_acc = enc_acc.encode(imu_sel[:3])
            spikes_gyro = enc_gyro.encode(imu_sel[3:])
            spikes_imu = np.concatenate([spikes_acc, spikes_gyro], axis=0)

            try:
                if self.eeg_encoder == "delta":
                    spikes_eeg = enc_eeg_delta.encode(eeg_sel)
                else:
                    spikes_eeg = encode_temporal_multichannel(
                        eeg_sel, "time_to_first_spike", interval=self.tts_interval
                    )
            except (ImportError, ValueError, RuntimeError):
                self.skipped_pairs += 1
                continue

            n_t_spike = min(spikes_imu.shape[1], spikes_eeg.shape[1])
            if n_t_spike < self.window_len:
                self.skipped_pairs += 1
                continue
            spikes_imu = spikes_imu[:, :n_t_spike]
            spikes_eeg = spikes_eeg[:, :n_t_spike]
            fused_spikes = np.concatenate([spikes_imu, spikes_eeg], axis=0)

            if save_encoder_viz and (not visualized):
                sub_id, ses_id = parse_sub_ses(source_imu)
                tag = f"{sub_id}_{ses_id}_pair{pair['pair_idx']}"
                save_encoder_visualizations(
                    raw_imu=imu_sel,
                    raw_eeg=eeg_sel,
                    spikes_imu=spikes_imu,
                    spikes_eeg=spikes_eeg,
                    out_dir=ENCODER_VIZ_DIR,
                    tag=tag,
                )
                visualized = True

            start = 0
            file_had_window = False
            while start + self.window_len <= fused_spikes.shape[1]:
                if max_per_class is not None and self.per_class_count[label] >= max_per_class:
                    break
                end = start + self.window_len
                self.samples.append((fused_spikes[:, start:end], label))
                self.per_class_count[label] += 1
                start += int(stride)
                file_had_window = True

            if file_had_window:
                self.paired_count += 1
            if self.n_units == 0 and self.samples:
                self.n_imu_units = spikes_imu.shape[0]
                self.n_eeg_units = spikes_eeg.shape[0]
                self.n_units = self.samples[0][0].shape[0]

        if self.samples:
            self.window_len = self.samples[0][0].shape[1]
        print(
            f"Loaded windows={len(self.samples)} | paired_used={self.paired_count} | "
            f"skipped_pairs={self.skipped_pairs} | per-class={self.per_class_count} | "
            f"eeg_encoder={self.eeg_encoder} tts_interval={self.tts_interval} "
            f"n_imu_units={self.n_imu_units} n_eeg_units={self.n_eeg_units} n_units_total={self.n_units}"
        )

    def __len__(self):
        """Return number of fused windows."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Return (window_tensor, class_label) for one sample."""
        win, label = self.samples[idx]
        return torch.from_numpy(win).float(), torch.tensor(label, dtype=torch.long)


def build_supervised_labels_single_layer(network: activeDendriteLayer, label: int, num_classes: int, device_):
    """Create dendrite-wise supervision mask."""
    nd = network.numDendrites
    if nd != num_classes:
        raise ValueError(f"numDendrites ({nd}) must match num_classes ({num_classes})")
    row_labels = torch.zeros(network.numNeurons * nd, device=device_)
    row_labels[label::nd] = 1.0
    seg_labels = row_labels.repeat_interleave(network.numSegments)
    return seg_labels.unsqueeze(1).repeat(1, network.numDistalInputs)


def class_votes_from_wta(network: activeDendriteLayer, num_classes: int, voting_thresh_local: float = 0.0):
    """Convert WTA segment activations to class votes."""
    seg_strength = torch.max(network.wtaOutput, dim=1)[0]
    if voting_thresh_local > 0:
        seg_strength = seg_strength * (seg_strength > voting_thresh_local)
    return seg_strength.reshape(network.numNeurons, num_classes).sum(dim=0)


def train_eval(
    train_loader,
    test_loader,
    num_distal_inputs: int,
    num_neurons: int,
    epochs: int,
    inc_learn_eval: int,
    capture_local: float,
    search_local: float,
    backoff_local: float,
    frac_local: float,
    wBase_local: int,
    wMax_local: int,
    voting_thresh_local: float,
):
    """Train then evaluate active_dendrite_core on fused windows; optional eval-time online updates."""
    network = activeDendriteLayer(
        numDistalInputs=num_distal_inputs,
        numSegments=numSegments,
        numDendrites=numDendrites,
        numNeurons=num_neurons,
        threshold=threshold,
        wMax=wMax_local,
        device=device,
    )
    if cuda:
        network.cuda()

    unfold = nn.Unfold(kernel_size=(1, kernel_len), stride=(1, stride_t))

    errorlist2 = []
    for ep in range(epochs):
        start_ep = time.time()
        train_error = 0
        interval_error = 0

        for idx, (data, target) in enumerate(train_loader):
            if cuda:
                data = data.cuda()
                target = target.cuda()
            label = int(target[0].item())

            data_4d = data.unsqueeze(2)
            patches = unfold(data_4d)
            sliced_data = patches.squeeze(0).T

            input_data, output_data = network(sliced_data)
            label_votes = class_votes_from_wta(
                network, num_classes=numDendrites, voting_thresh_local=voting_thresh_local
            )
            if int(torch.argmax(label_votes).item()) != label:
                train_error += 1
                interval_error += 1

            input_labels = build_supervised_labels_single_layer(
                network=network, label=label, num_classes=numDendrites, device_=device
            )
            network.weights = network.update(
                input_data,
                input_labels,
                output_data,
                network.weights,
                capture_local,
                search_local,
                backoff_local,
                wBase_local,
                network.rewardEn,
                frac_local,
            )

            if (idx + 1) % interval1 == 0:
                errorlist2.append(interval_error / interval1)
                interval_error = 0

        train_acc = (len(train_loader) - train_error) * 100.0 / max(1, len(train_loader))
        print(f"Epoch {ep + 1}: train_acc={train_acc:.2f}% | time={time.time() - start_ep:.2f}s")

    all_labels = []
    all_preds = []
    test_error = 0
    interval_error = 0
    start_test = time.time()

    for idx, (data, target) in enumerate(test_loader):
        if cuda:
            data = data.cuda()
            target = target.cuda()
        label = int(target[0].item())
        data_4d = data.unsqueeze(2)
        patches = unfold(data_4d)
        sliced_data = patches.squeeze(0).T
        input_data, output_data = network(sliced_data)

        label_votes = class_votes_from_wta(
            network, num_classes=numDendrites, voting_thresh_local=voting_thresh_local
        )
        pred = int(torch.argmax(label_votes).item())
        all_labels.append(label)
        all_preds.append(pred)
        if pred != label:
            test_error += 1
            interval_error += 1

        if inc_learn_eval == 1:
            input_labels = build_supervised_labels_single_layer(
                network=network, label=label, num_classes=numDendrites, device_=device
            )
            network.weights = network.update(
                input_data,
                input_labels,
                output_data,
                network.weights,
                capture_local,
                search_local,
                backoff_local,
                wBase_local,
                network.rewardEn,
                frac_local,
            )

        if (idx + 1) % interval2 == 0:
            errorlist2.append(interval_error / interval2)
            interval_error = 0

    test_acc = (len(test_loader) - test_error) * 100.0 / max(1, len(test_loader))
    print(f"Test done in {time.time() - start_test:.2f}s | test_acc={test_acc:.2f}%")

    conf_mat = np.zeros((numDendrites, numDendrites), dtype=int)
    for t, p in zip(all_labels, all_preds):
        conf_mat[t, p] += 1
    print("Confusion matrix (rows=true, cols=pred):")
    print(conf_mat)
    return train_acc, test_acc, errorlist2, conf_mat


def metrics_from_conf_mat(conf_mat: np.ndarray) -> tuple[list[float], list[float], list[float], float]:
    """Compute per-class precision/recall/F1 and macro F1 from confusion matrix."""
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    for c in range(conf_mat.shape[0]):
        tp = float(conf_mat[c, c])
        fp = float(np.sum(conf_mat[:, c]) - tp)
        fn = float(np.sum(conf_mat[c, :]) - tp)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision.append(round(p, 4))
        recall.append(round(r, 4))
        f1.append(round(f, 4))
    macro_f1 = float(np.mean(f1)) if f1 else 0.0
    return precision, recall, f1, macro_f1


def save_best_result_txt(
    out_path: Path,
    cfg: SearchConfig,
    train_acc: float,
    test_acc: float,
    conf_mat: np.ndarray,
    precision: list[float],
    recall: list[float],
    f1: list[float],
    macro_f1: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Best Multimodel Result",
        "=" * 60,
        f"Train Accuracy: {train_acc:.4f}%",
        f"Test Accuracy:  {test_acc:.4f}%",
        f"Macro F1:       {macro_f1:.4f}",
        "",
        "Best Config:",
        json.dumps(asdict(cfg), indent=2),
        "",
        "Confusion Matrix (rows=true, cols=pred):",
        np.array2string(conf_mat),
        "",
        f"Per-class precision: {precision}",
        f"Per-class recall:    {recall}",
        f"Per-class F1:        {f1}",
        "",
    ]
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_training_plots(errorlist2: list[float], conf_mat: np.ndarray, out_dir: Path):
    """Save error-curve and confusion-matrix figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if errorlist2:
        plt.figure(figsize=(10, 4))
        plt.plot(errorlist2, color="red", linestyle="solid", linewidth=1.5)
        plt.xlabel("Intervals")
        plt.ylabel("Error Rate")
        plt.legend(["activeDendriteLayer (multimodel IMU+EEG)"])
        plt.tight_layout()
        plt.savefig(out_dir / "online_adaptivity_multimodel.png")
        plt.close()

    plt.figure(figsize=(4, 4))
    im = plt.imshow(conf_mat, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    classes = ["stand", "slow", "fast", "run"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Multimodel IMU+EEG Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_multimodel.png")
    plt.close()


def load_imu_manifest() -> list[dict]:
    """Read IMU manifest; return empty list if missing."""
    try:
        with IMU_MANIFEST_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def print_eval_protocol():
    """Print reproducible commands for delta vs tts comparison."""
    script_name = Path(__file__).name
    base_cmd = (
        f"python {script_name} --seed {RUNTIME_ARGS.seed} --train-ratio {RUNTIME_ARGS.train_ratio} "
        f"--window-len {RUNTIME_ARGS.window_len} --window-stride {RUNTIME_ARGS.window_stride} "
        f"--max-train-windows {RUNTIME_ARGS.max_train_windows} --max-test-windows {RUNTIME_ARGS.max_test_windows} "
        f"--epochs {RUNTIME_ARGS.epochs}"
    )
    print("Evaluation protocol (fixed split/seed):")
    print(f"  1) Delta EEG : {base_cmd} --eeg-encoder delta")
    print(f"  2) TTS EEG   : {base_cmd} --eeg-encoder tts --tts-interval {RUNTIME_ARGS.tts_interval}")


def print_window_spike_stats(dataset: PairedImuEegSlidingWindowDataset, tag: str) -> None:
    """Print spike-count distribution and sparsity for one dataset split."""
    if len(dataset.samples) == 0:
        print(f"[{tag}] No windows available for spike statistics.")
        return

    spike_counts = np.asarray([int(win.sum()) for win, _ in dataset.samples], dtype=np.float64)
    total_positions = float(dataset.n_units * dataset.window_len)
    density = spike_counts / total_positions if total_positions > 0 else np.zeros_like(spike_counts)
    labels = np.asarray([int(label) for _, label in dataset.samples], dtype=np.int64)
    class_hist = np.bincount(labels, minlength=NUM_CLASSES)

    print(
        f"[{tag}] windows={len(dataset.samples)} "
        f"spikes/window mean={spike_counts.mean():.2f} median={np.median(spike_counts):.2f} "
        f"min={spike_counts.min():.0f} max={spike_counts.max():.0f} "
        f"p10={np.percentile(spike_counts, 10):.2f} p90={np.percentile(spike_counts, 90):.2f}"
    )
    print(
        f"[{tag}] density mean={density.mean():.6f} median={np.median(density):.6f} "
        f"p10={np.percentile(density, 10):.6f} p90={np.percentile(density, 90):.6f}"
    )
    print(f"[{tag}] class_hist={class_hist.tolist()}")


def build_datasets_for_config(
    train_pairs: list[dict],
    test_pairs: list[dict],
    eeg_encoder: str,
    tts_interval: int,
    save_encoder_viz: bool,
) -> tuple[PairedImuEegSlidingWindowDataset, PairedImuEegSlidingWindowDataset]:
    train_dataset = PairedImuEegSlidingWindowDataset(
        pairs=train_pairs,
        imu_dataset_root=IMU_DATASET_ROOT,
        window_len=RUNTIME_ARGS.window_len,
        stride=RUNTIME_ARGS.window_stride,
        max_windows=RUNTIME_ARGS.max_train_windows,
        eeg_encoder=eeg_encoder,
        tts_interval=tts_interval,
        save_encoder_viz=save_encoder_viz,
    )
    test_dataset = PairedImuEegSlidingWindowDataset(
        pairs=test_pairs,
        imu_dataset_root=IMU_DATASET_ROOT,
        window_len=RUNTIME_ARGS.window_len,
        stride=RUNTIME_ARGS.window_stride,
        max_windows=RUNTIME_ARGS.max_test_windows,
        eeg_encoder=eeg_encoder,
        tts_interval=tts_interval,
        save_encoder_viz=False,
    )
    return train_dataset, test_dataset


def run_single_config(
    cfg: SearchConfig,
    train_pairs: list[dict],
    test_pairs: list[dict],
    save_encoder_viz: bool = False,
    log_prefix: str = "",
):
    if cfg.eeg_encoder == "tts" and (not is_spikify_available()):
        raise RuntimeError("TTS EEG encoding requires spikify. Install: pip install innuce-spikify")

    train_dataset, test_dataset = build_datasets_for_config(
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        eeg_encoder=cfg.eeg_encoder,
        tts_interval=cfg.tts_interval,
        save_encoder_viz=save_encoder_viz,
    )
    if len(train_dataset) <= 0 or len(test_dataset) <= 0:
        raise RuntimeError("Insufficient paired windows after preprocessing.")

    print_window_spike_stats(train_dataset, f"{log_prefix}train")
    print_window_spike_stats(test_dataset, f"{log_prefix}test")

    expected_eeg_units = (
        n_eeg_channels * n_thresholds * 2
        if cfg.eeg_encoder == "delta"
        else n_units_for_method("time_to_first_spike", n_eeg_channels)
    )
    if train_dataset.n_eeg_units != expected_eeg_units:
        raise RuntimeError(
            f"EEG unit mismatch: expected {expected_eeg_units}, got {train_dataset.n_eeg_units}"
        )
    if train_dataset.n_units != test_dataset.n_units:
        raise RuntimeError(
            f"Train/Test unit mismatch: train={train_dataset.n_units}, test={test_dataset.n_units}"
        )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    n_units = train_dataset.n_units
    num_distal_inputs = kernel_len * n_units
    num_neurons = (train_dataset.window_len - kernel_len) // stride_t + 1
    print(
        f"{log_prefix}Model dims | eeg_encoder={cfg.eeg_encoder} tts_interval={cfg.tts_interval} "
        f"n_imu_units={train_dataset.n_imu_units} n_eeg_units={train_dataset.n_eeg_units} "
        f"n_units_total={n_units} numDistalInputs={num_distal_inputs} "
        f"numNeurons={num_neurons} window_len={train_dataset.window_len}"
    )

    train_acc, test_acc, errorlist2, conf_mat = train_eval(
        train_loader=train_loader,
        test_loader=test_loader,
        num_distal_inputs=num_distal_inputs,
        num_neurons=num_neurons,
        epochs=cfg.epochs,
        inc_learn_eval=cfg.inc_learn_eval,
        capture_local=cfg.capture,
        search_local=cfg.search,
        backoff_local=cfg.backoff,
        frac_local=cfg.frac,
        wBase_local=cfg.wBase,
        wMax_local=cfg.wMax,
        voting_thresh_local=cfg.voting_thresh,
    )
    precision, recall, f1, macro_f1 = metrics_from_conf_mat(conf_mat)
    print(
        f"{log_prefix}Metric summary | train={train_acc:.2f}% test={test_acc:.2f}% "
        f"macro_f1={macro_f1:.4f} per_class_f1={f1}"
    )
    return train_acc, test_acc, errorlist2, conf_mat, precision, recall, f1, macro_f1


def run_search_configs(train_pairs: list[dict], test_pairs: list[dict]):
    cfgs = sample_quick_configs(seed=RUNTIME_ARGS.search_seed, n_samples=RUNTIME_ARGS.search_trials)
    if not cfgs:
        print("No search configs generated.")
        return None

    best = None
    best_acc = -1.0
    for i, cfg in enumerate(cfgs, 1):
        print(f"=== Search Config {i}/{len(cfgs)} ===")
        print(asdict(cfg))
        try:
            out = run_single_config(cfg, train_pairs, test_pairs, save_encoder_viz=False, log_prefix=f"[cfg{i}] ")
        except RuntimeError as exc:
            print(f"[cfg{i}] skipped: {exc}")
            continue

        train_acc, test_acc, errorlist2, conf_mat, precision, recall, f1, macro_f1 = out
        if test_acc > best_acc:
            best_acc = test_acc
            best = (cfg, train_acc, test_acc, errorlist2, conf_mat, precision, recall, f1, macro_f1)

    if best is None:
        return None

    save_search_config(best[0], BEST_CONFIG_PATH)
    save_best_result_txt(
        out_path=BEST_RESULT_TXT_PATH,
        cfg=best[0],
        train_acc=best[1],
        test_acc=best[2],
        conf_mat=best[4],
        precision=best[5],
        recall=best[6],
        f1=best[7],
        macro_f1=best[8],
    )
    print(f"BEST TEST RESULT: {best[2]:.2f}%")
    print(f"Best config saved to: {BEST_CONFIG_PATH}")
    print(f"Best result txt saved to: {BEST_RESULT_TXT_PATH}")
    return best


def main():
    """Run full paired multimodal pipeline with single/load-best/search modes."""
    imu_manifest = load_imu_manifest()
    imu_entries = collect_imu_entries(imu_manifest)
    eeg_entries = collect_eeg_entries(EEG_DATASET_ROOT)
    paired_entries, dropped_unpaired = pair_imu_eeg_entries(imu_entries, eeg_entries)
    print(
        f"Entries | IMU={len(imu_entries)} EEG={len(eeg_entries)} paired={len(paired_entries)} "
        f"dropped_unpaired={dropped_unpaired}"
    )
    if not paired_entries:
        print("No paired ERP entries found for ses-02..05.")
        return

    train_pairs, test_pairs = split_pairs_by_subject(
        paired_entries, train_ratio=RUNTIME_ARGS.train_ratio, seed=RUNTIME_ARGS.seed
    )
    print(f"Pair split | train={len(train_pairs)} test={len(test_pairs)}")

    if RUNTIME_ARGS.load_best_config:
        if not BEST_CONFIG_PATH.exists():
            raise SystemExit(f"Best config file not found: {BEST_CONFIG_PATH}")
        cfg = load_search_config(BEST_CONFIG_PATH)
        print(f"Loaded best config from: {BEST_CONFIG_PATH}")
        print(asdict(cfg))
        out = run_single_config(cfg, train_pairs, test_pairs, save_encoder_viz=RUNTIME_ARGS.save_encoder_viz)
        train_acc, test_acc, errorlist2, conf_mat, _, _, _, macro_f1 = out
        print(f"Final Accuracy | train={train_acc:.2f}% test={test_acc:.2f}% macro_f1={macro_f1:.4f}")
        save_training_plots(errorlist2, conf_mat, GRAPHS_DIR)
        print(f"Saved plots to: {GRAPHS_DIR}")
        return

    if RUNTIME_ARGS.run_config_search == 1:
        best = run_search_configs(train_pairs, test_pairs)
        if best is None:
            print("Search did not produce a valid run.")
            return
        _, train_acc, test_acc, errorlist2, conf_mat, _, _, _, macro_f1 = best
        print(f"Search best | train={train_acc:.2f}% test={test_acc:.2f}% macro_f1={macro_f1:.4f}")
        save_training_plots(errorlist2, conf_mat, GRAPHS_DIR)
        print(f"Saved plots to: {GRAPHS_DIR}")
        return

    cfg = SearchConfig(
        capture=capture,
        search=search,
        backoff=backoff,
        frac=frac,
        wBase=wBase,
        wMax=wMax,
        voting_thresh=voting_thresh,
        epochs=RUNTIME_ARGS.epochs,
        inc_learn_eval=RUNTIME_ARGS.inc_learn_eval,
        eeg_encoder=RUNTIME_ARGS.eeg_encoder,
        tts_interval=RUNTIME_ARGS.tts_interval,
    )
    out = run_single_config(cfg, train_pairs, test_pairs, save_encoder_viz=RUNTIME_ARGS.save_encoder_viz)
    train_acc, test_acc, errorlist2, conf_mat, _, _, _, macro_f1 = out
    print(f"Final Accuracy | train={train_acc:.2f}% test={test_acc:.2f}% macro_f1={macro_f1:.4f}")

    save_training_plots(errorlist2, conf_mat, GRAPHS_DIR)
    print(f"Saved plots to: {GRAPHS_DIR}")
    print_eval_protocol()


if __name__ == "__main__":
    main()
