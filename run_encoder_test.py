"""
Test delta-modulation encoder on IMU and EEG data.

Default mode scans BrainVision EEG files from:
    osfstorage-archive/sub-*/ses-*/eeg/*task-ERP*_eeg.eeg

Results are saved under visualize_encoder/.
"""

from pathlib import Path
import sys
import argparse
import numpy as np

# Add osfstorage-archive so we can import read_imu_dat
REPO_ROOT = Path(__file__).resolve().parent
OSF = REPO_ROOT.parent / "osfstorage-archive"
sys.path.insert(0, str(OSF))

try:
    from read_imu_dat import load_imu_brainvision, channel_roles
except ImportError:
    load_imu_brainvision = None
    channel_roles = None

from encoders import DeltaModulationEncoder


# Default IMU data directory (relative to repo root)
IMU_DIR = REPO_ROOT / "osfstorage-archive" / "sourcedata" / "sub-01" / "ses-01" / "imu"
OUT_DIR = REPO_ROOT / "visualize_encoder"
EEG_ROOT = REPO_ROOT.parent / "osfstorage-archive"
EEG_SUBJECT = "sub-01"
EEG_TASK = "ERP"
EEG_SESSIONS = ["ses-02", "ses-03", "ses-04", "ses-05"]
EEG_DURATION_SEC = 1.0
EEG_MAX_CHANNELS = 1
EEG_CHANNEL_NAME = "FZ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run delta encoder visualization for EEG or IMU.")
    parser.add_argument("--mode", choices=["eeg", "imu-npy"], default="eeg")
    return parser.parse_args()


def discover_eeg_files(
    eeg_root: Path,
    subject: str,
    session: str,
    task: str = "ERP",
) -> list[Path]:
    """Find EEG .eeg files under osfstorage-archive/sub-*/ses-*/eeg (no sourcedata folder)."""
    eeg_root = Path(eeg_root).resolve()
    task_upper = task.upper()

    if subject == "*" and session == "*":
        patt = "sub-*/ses-*/eeg/*.eeg"
    elif session == "*":
        patt = f"{subject}/ses-*/eeg/*.eeg"
    else:
        patt = f"{subject}/{session}/eeg/*.eeg"

    candidates = []
    for p in eeg_root.glob(patt):
        name_u = p.name.upper()
        if f"TASK-{task_upper}" in name_u and "_EEG" in name_u:
            candidates.append(p)
    return sorted(candidates)


def select_eeg_channels(ch_names: list[str], preferred: str = "FZ", max_channels: int = 1) -> list[int]:
    """Select EEG channels preferring FZ, then CZ/PZ, then first valid EEG-like channels."""
    name_to_idx = {name.upper(): i for i, name in enumerate(ch_names)}
    preferred_order = [preferred.upper(), "CZ", "PZ"]
    bad_prefixes = ("EOG", "ECG", "EMG", "TRIG", "STI", "EXG")

    picked: list[int] = []
    for name in preferred_order:
        idx = name_to_idx.get(name)
        if idx is not None and idx not in picked:
            picked.append(idx)
            if len(picked) >= max_channels:
                return picked

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


def _synthetic_imu(n_channels: int, n_times: int, sfreq: float, seed: int = 42) -> tuple[np.ndarray, dict]:
    """Generate synthetic IMU-like (n_channels, n_times) and minimal info dict."""
    rng = np.random.default_rng(seed)
    # Random walk + sine per channel
    t = np.arange(n_times, dtype=np.float64) / sfreq
    data = np.zeros((n_channels, n_times), dtype=np.float64)
    for c in range(n_channels):
        walk = np.cumsum(rng.standard_normal(n_times)) * 0.01
        wave = 0.3 * np.sin(2 * np.pi * (0.5 + c * 0.1) * t)
        data[c] = walk + wave + rng.standard_normal(n_times) * 0.02
    info = {
        "ch_names": [f"Ch{i+1}" for i in range(n_channels)],
        "sfreq": sfreq,
        "times": t,
        "n_channels": n_channels,
        "n_points": n_times,
    }
    return data, info


def load_imu_or_synthetic(
    imu_dir: Path,
    duration_sec: float = 5.0,
    max_load_sec: float = 60.0,
) -> tuple[np.ndarray, dict, bool]:
    """
    Load IMU from BrainVision in imu_dir, or generate synthetic if binary missing.
    Loads up to max_load_sec of data so the caller can pick an active window.
    Returns (data, info, used_synthetic).
    """
    vhdr = imu_dir / "sub-01_ses-01_task-ERP_imu.vhdr"
    if not vhdr.exists():
        vhdr = next(imu_dir.glob("*.vhdr"), None)
    if vhdr is None:
        sfreq = 128.0
        n_ch = 27
        n_times = int(duration_sec * sfreq)
        data, info = _synthetic_imu(n_ch, n_times, sfreq)
        return data, info, True

    if load_imu_brainvision is None:
        sfreq = 128.0
        n_ch = 27
        n_times = int(duration_sec * sfreq)
        data, info = _synthetic_imu(n_ch, n_times, sfreq)
        return data, info, True

    try:
        data, info = load_imu_brainvision(vhdr)
        sfreq = info["sfreq"] or 128.0
        n_samp = int(max_load_sec * sfreq)
        n_samp = min(n_samp, data.shape[1])
        data = data[:, :n_samp].copy()
        info = {**info, "times": info["times"][:n_samp], "n_points": n_samp}
        return data, info, False
    except (FileNotFoundError, OSError, ValueError) as e:
        print(f"Load failed ({e}), using synthetic data.")
        n_ch = 27
        sfreq = 128.0
        try:
            text = vhdr.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if line.strip().startswith("NumberOfChannels="):
                    n_ch = int(line.split("=", 1)[1].strip())
                elif line.strip().startswith("SamplingInterval="):
                    interval_us = float(line.split("=", 1)[1].strip())
                    if interval_us > 0:
                        sfreq = 1e6 / interval_us
        except Exception:
            pass
        n_times = int(duration_sec * sfreq)
        data, info = _synthetic_imu(n_ch, n_times, sfreq)
        return data, info, True


def _busiest_window(data: np.ndarray, sfreq: float, duration_sec: float) -> int:
    """
    Return start sample index of the contiguous window of length duration_sec
    that has the largest total variation (sum of squared differences over time and channels).
    data: (n_channels, n_times). Uses all channels.
    """
    n_ch, n_times = data.shape
    win_len = int(round(duration_sec * sfreq))
    win_len = min(win_len, n_times)
    if win_len >= n_times:
        return 0
    # Total "activity" per window = sum over channels of sum of squared diffs in that channel
    activity = np.zeros(n_times - win_len + 1, dtype=np.float64)
    for start in range(activity.shape[0]):
        w = data[:, start : start + win_len]
        activity[start] = np.sum(np.diff(w, axis=1) ** 2)
    return int(np.argmax(activity))


def encode_and_plot_from_npy(npy_path: Path, out_prefix: str, duration_sec: float = 1.0) -> None:
    """
    Load preprocessed IMU from a .npy file (+ optional _meta.json), run
    DeltaModulationEncoder, and save figures with the given prefix.
    """
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = Path(npy_path)
    print(f"\n=== Dataset: {npy_path} ===")
    if not npy_path.exists():
        print(f"  File not found, skip: {npy_path}")
        return

    data = np.load(npy_path)
    if data.ndim != 2:
        data = np.asarray(data)
        if data.ndim != 2:
            print(f"  Expected 2D data (n_channels, n_times), got shape {data.shape}, skip.")
            return

    n_ch, n_times = data.shape

    # Try to load meta info (sfreq, ch_names, etc.) from *_meta.json
    meta_path = npy_path.with_name(npy_path.stem + "_meta.json")
    sfreq = 128.0
    ch_names = [f"Ch{i+1}" for i in range(n_ch)]
    times = np.arange(n_times, dtype=np.float64) / sfreq
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "sfreq" in meta and meta["sfreq"]:
                sfreq = float(meta["sfreq"])
            if "ch_names" in meta and meta["ch_names"]:
                ch_names = meta["ch_names"]
            n_points = meta.get("n_points", n_times)
            times = np.arange(n_points, dtype=np.float64) / sfreq
            print(f"  Loaded meta from {meta_path} (sfreq={sfreq}, n_points={n_points})")
        except Exception as e:
            print(f"  Warning: failed to read meta ({e}), using defaults (sfreq=128 Hz).")
            times = np.arange(n_times, dtype=np.float64) / sfreq
    else:
        print("  Meta file not found, using defaults (sfreq=128 Hz, generic ch_names).")

    print(f"  Data shape: {data.shape}, sfreq: {sfreq} Hz")

    # Use first few channels for clearer plots
    channel_indices = list(range(min(6, n_ch)))
    data_sel_full = data[channel_indices]

    # Pick most "active" 1 s window based on total variation
    win_len = int(round(duration_sec * sfreq))
    win_len = min(win_len, data_sel_full.shape[1])
    if data_sel_full.shape[1] > win_len:
        start_idx = _busiest_window(data_sel_full, sfreq, duration_sec)
        data_sel = data_sel_full[:, start_idx : start_idx + win_len].copy()
        times = times[start_idx : start_idx + win_len]
        print(f"  Using busiest {duration_sec} s window starting at t = {times[0]:.2f} s")
    else:
        data_sel = data_sel_full
        times = times[: data_sel_full.shape[1]]

    labels_sel = [
        ch_names[i] if i < len(ch_names) else str(i) for i in channel_indices
    ]

    # ----- Delta modulation encoder -----
    diff_scale = np.median(np.abs(np.diff(data_sel))) or 0.01
    th_base = max(diff_scale * 0.5, 1e-6)
    delta_enc = DeltaModulationEncoder(n_thresholds=5, threshold_base=th_base, scale=2.0)
    delta_spikes = delta_enc.encode(data_sel, channel_indices=None)

    # ----- Visualization: IMU + delta-modulation spikes (time-aligned) -----
    n_samp = data_sel.shape[1]
    t_sec_rel = np.arange(n_samp, dtype=np.float64) / sfreq
    sig = data_sel[0]
    t_max = t_sec_rel[-1] if n_samp else 0.0

    # ---------- Figure: Delta modulation ----------
    Sd = delta_spikes[: delta_enc.n_thresholds * 2, :]
    n_d = delta_enc.n_thresholds
    thresholds_show = list(range(n_d))
    spike_half_len = 0.4

    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True, height_ratios=[1, 2])
    ax_sig2 = axes2[0]
    ax_sig2.plot(t_sec_rel, sig, color="blue", linewidth=1.2, label=labels_sel[0])
    ax_sig2.set_ylabel("IMU (a.u.)")
    ax_sig2.set_title(
        f"Delta modulation ({out_prefix}): IMU (one axis) and spikes — red ↑, blue ↓ (T0–T4)"
    )
    ax_sig2.legend(loc="upper right", fontsize=8)
    ax_sig2.grid(True, alpha=0.3)
    ax_sig2.set_xlim(0, t_max)

    ax_raster2 = axes2[1]
    for i, th in enumerate(thresholds_show):
        if th >= n_d:
            continue
        y_row = float(i)
        t_pos = np.where(Sd[th, :] > 0)[0]
        t_neg = np.where(Sd[n_d + th, :] > 0)[0]
        t_sec_pos = t_pos.astype(np.float64) / sfreq
        t_sec_neg = t_neg.astype(np.float64) / sfreq
        if t_sec_pos.size:
            ax_raster2.vlines(
                t_sec_pos,
                y_row - spike_half_len,
                y_row + spike_half_len,
                colors="red",
                linewidth=2.5,
                alpha=0.9,
            )
        if t_sec_neg.size:
            ax_raster2.vlines(
                t_sec_neg,
                y_row - spike_half_len,
                y_row + spike_half_len,
                colors="blue",
                linewidth=2.5,
                alpha=0.9,
            )
    ax_raster2.set_ylabel("Threshold")
    ax_raster2.set_xlabel("Time (s)")
    ax_raster2.set_ylim(-0.5, len(thresholds_show) - 0.5)
    ax_raster2.set_yticks(np.arange(len(thresholds_show)))
    ax_raster2.set_yticklabels([f"T{th}" for th in thresholds_show], fontsize=9)
    ax_raster2.set_xlim(0, t_max)
    ax_raster2.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path2 = out_dir / f"{out_prefix}_delta_encoder_with_imu.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved:", out_path2)


def encode_and_plot_from_eeg_file(
    eeg_path: Path,
    out_prefix: str,
    duration_sec: float = 1.0,
    max_channels: int = 1,
    preferred_channel: str = "FZ",
    out_dir: Path | None = None,
) -> None:
    """Load BrainVision EEG from .eeg (+ .vhdr), encode delta spikes, and save EEG-specific figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if load_imu_brainvision is None:
        print("  BrainVision loader unavailable, skip:", eeg_path)
        return

    out_dir = OUT_DIR if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eeg_path = Path(eeg_path)
    print(f"\n=== EEG Dataset: {eeg_path} ===")
    if not eeg_path.exists():
        print("  File not found, skip:", eeg_path)
        return

    try:
        data, info = load_imu_brainvision(eeg_path, apply_scaling=True)
    except Exception as e:
        print(f"  Failed to load EEG ({e}), skip.")
        return

    ch_names = info.get("ch_names", [])
    sfreq = float(info.get("sfreq") or 128.0)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        print(f"  Unexpected EEG shape {data.shape}, skip.")
        return

    channel_indices = select_eeg_channels(ch_names, preferred=preferred_channel, max_channels=max_channels)
    if not channel_indices:
        print("  No valid EEG channels found, skip.")
        return

    labels_sel = [ch_names[i] if i < len(ch_names) else f"Ch{i+1}" for i in channel_indices]
    data_sel_full = data[channel_indices].astype(np.float64)
    n_times = data_sel_full.shape[1]
    times = np.arange(n_times, dtype=np.float64) / sfreq

    win_len = int(round(duration_sec * sfreq))
    win_len = min(win_len, n_times)
    if n_times > win_len:
        start_idx = _busiest_window(data_sel_full, sfreq, duration_sec)
        data_sel = data_sel_full[:, start_idx : start_idx + win_len].copy()
        times = times[start_idx : start_idx + win_len]
        print(f"  Using busiest {duration_sec} s window starting at t = {times[0]:.2f} s")
    else:
        data_sel = data_sel_full

    print(f"  Chosen channels: {labels_sel}")
    print(f"  Data shape: {data_sel.shape}, sfreq: {sfreq} Hz")

    diff_scale = np.median(np.abs(np.diff(data_sel))) or 0.01
    th_base = max(diff_scale * 0.5, 1e-6)
    delta_enc = DeltaModulationEncoder(n_thresholds=5, threshold_base=th_base, scale=2.0)
    delta_spikes = delta_enc.encode(data_sel, channel_indices=None)

    n_samp = data_sel.shape[1]
    t_sec_rel = np.arange(n_samp, dtype=np.float64) / sfreq
    sig = data_sel[0]
    t_max = t_sec_rel[-1] if n_samp else 0.0

    Sd = delta_spikes[: delta_enc.n_thresholds * 2, :]
    n_d = delta_enc.n_thresholds
    thresholds_show = list(range(n_d))
    spike_half_len = 0.4

    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True, height_ratios=[1, 2])
    ax_sig2 = axes2[0]
    ax_sig2.plot(t_sec_rel, sig, color="blue", linewidth=1.2, label=labels_sel[0])
    ax_sig2.set_ylabel("EEG (a.u.)")
    ax_sig2.set_title(
        f"Delta modulation ({out_prefix}): EEG (one channel) and spikes - red up, blue down (T0-T4)"
    )
    ax_sig2.legend(loc="upper right", fontsize=8)
    ax_sig2.grid(True, alpha=0.3)
    ax_sig2.set_xlim(0, t_max)

    ax_raster2 = axes2[1]
    for i, th in enumerate(thresholds_show):
        if th >= n_d:
            continue
        y_row = float(i)
        t_pos = np.where(Sd[th, :] > 0)[0]
        t_neg = np.where(Sd[n_d + th, :] > 0)[0]
        t_sec_pos = t_pos.astype(np.float64) / sfreq
        t_sec_neg = t_neg.astype(np.float64) / sfreq
        if t_sec_pos.size:
            ax_raster2.vlines(
                t_sec_pos,
                y_row - spike_half_len,
                y_row + spike_half_len,
                colors="red",
                linewidth=2.5,
                alpha=0.9,
            )
        if t_sec_neg.size:
            ax_raster2.vlines(
                t_sec_neg,
                y_row - spike_half_len,
                y_row + spike_half_len,
                colors="blue",
                linewidth=2.5,
                alpha=0.9,
            )
    ax_raster2.set_ylabel("Threshold")
    ax_raster2.set_xlabel("Time (s)")
    ax_raster2.set_ylim(-0.5, len(thresholds_show) - 0.5)
    ax_raster2.set_yticks(np.arange(len(thresholds_show)))
    ax_raster2.set_yticklabels([f"T{th}" for th in thresholds_show], fontsize=9)
    ax_raster2.set_xlim(0, t_max)
    ax_raster2.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path2 = out_dir / f"{out_prefix}_delta_encoder_with_imu.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved:", out_path2)


def run_legacy_imu_examples() -> int:
    """Keep old hard-coded IMU .npy example behavior for compatibility."""
    sub01_ses02 = Path(
        r"D:\xia\CMU\18743-Neuromorphic\Multi_Sensory_NeuTNN\NeuTNN_imu\imu_dataset_clean\sub-01\ses-02\imu\sub-01_ses-02_task-ERP_imu.npy"
    )
    sub01_ses03 = Path(
        r"D:\xia\CMU\18743-Neuromorphic\Multi_Sensory_NeuTNN\NeuTNN_imu\imu_dataset_clean\sub-01\ses-03\imu\sub-01_ses-03_task-ERP_imu.npy"
    )
    sub01_ses04 = Path(
        r"D:\xia\CMU\18743-Neuromorphic\Multi_Sensory_NeuTNN\NeuTNN_imu\imu_dataset_clean\sub-01\ses-04\imu\sub-01_ses-04_task-ERP_imu.npy"
    )
    sub01_ses05 = Path(
        r"D:\xia\CMU\18743-Neuromorphic\Multi_Sensory_NeuTNN\NeuTNN_imu\imu_dataset_clean\sub-01\ses-05\imu\sub-01_ses-05_task-ERP_imu.npy"
    )

    encode_and_plot_from_npy(sub01_ses02, out_prefix="sub01_ses02")
    encode_and_plot_from_npy(sub01_ses03, out_prefix="sub01_ses03")
    encode_and_plot_from_npy(sub01_ses04, out_prefix="sub01_ses04")
    encode_and_plot_from_npy(sub01_ses05, out_prefix="sub01_ses05")
    return 0


def main():
    args = parse_args()

    if args.mode == "imu":
        return run_legacy_imu_examples()

    eeg_root = EEG_ROOT.resolve()
    out_dir = OUT_DIR.resolve()
    eeg_files: list[Path] = []
    for ses in EEG_SESSIONS:
        eeg_files.extend(
            discover_eeg_files(
                eeg_root=eeg_root,
                subject=EEG_SUBJECT,
                session=ses,
                task=EEG_TASK,
            )
        )

    # Deduplicate in case the dataset has duplicated links/files.
    eeg_files = sorted({p.resolve() for p in eeg_files})

    if not eeg_files:
        print(
            "No EEG files found for "
            f"subject={EEG_SUBJECT}, sessions={EEG_SESSIONS}, task={EEG_TASK} "
            f"under root={eeg_root}"
        )
        return 1

    print(f"Found {len(eeg_files)} EEG file(s) under {eeg_root}")
    for eeg_path in eeg_files:
        stem = eeg_path.stem
        out_prefix = stem
        encode_and_plot_from_eeg_file(
            eeg_path=eeg_path,
            out_prefix=out_prefix,
            duration_sec=EEG_DURATION_SEC,
            max_channels=EEG_MAX_CHANNELS,
            preferred_channel=EEG_CHANNEL_NAME,
            out_dir=out_dir,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
