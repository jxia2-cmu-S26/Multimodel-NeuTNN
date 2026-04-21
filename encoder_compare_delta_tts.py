"""
Visualize EEG delta encoder vs TTFS encoder on the same subject.

For one subject with all 4 sessions (ses-02/03/04/05), this script:
1) picks one ERP file from each session,
2) extracts a short time segment,
3) encodes with DeltaModulationEncoder and TTFS,
4) saves a 4x3 figure (raw / delta / tts per class).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from encoders import DeltaModulationEncoder
from read_imu_dat import load_imu_brainvision
from spikify_temporal_encode import encode_temporal_multichannel, is_spikify_available


EEG_ROOT_DEFAULT = Path(r"D:/xia/CMU/18743-Neuromorphic/Multi_Sensory_NeuTNN/osfstorage-archive")
OUT_DIR_DEFAULT = Path(__file__).resolve().parent / "graphs_multimodel"
TARGET_SESSIONS = ("ses-02", "ses-03", "ses-04", "ses-05")


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--eeg-root", type=str, default=str(EEG_ROOT_DEFAULT))
    parser.add_argument("--subject", type=str, default="", help="Subject id like sub-17; empty=auto-select.")
    parser.add_argument("--segment-sec", type=float, default=2.0, help="Segment length in seconds.")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Segment start time in seconds.")
    parser.add_argument("--resample-rate", type=float, default=128.0)
    parser.add_argument("--tts-interval", type=int, default=8)
    parser.add_argument("--n-thresholds", type=int, default=3)
    parser.add_argument("--delta-threshold-base-eeg", type=float, default=10.62202935)
    parser.add_argument("--delta-scale", type=float, default=2.0)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))
    return parser.parse_args()


def select_eeg_channel_indices(ch_names: list[str], max_channels: int = 3) -> list[int]:
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
    if old_fs <= 0 or abs(old_fs - new_fs) < 1e-9:
        return data
    n_samples_new = int(data.shape[1] * (new_fs / old_fs))
    n_samples_new = max(2, n_samples_new)
    x_old = np.arange(data.shape[1], dtype=np.float64)
    x_new = np.linspace(0, data.shape[1] - 1, n_samples_new)
    out = np.zeros((data.shape[0], n_samples_new), dtype=np.float64)
    for ch_idx in range(data.shape[0]):
        out[ch_idx] = np.interp(x_new, x_old, data[ch_idx])
    return out


def find_subject_files(eeg_root: Path, subject: str | None = None) -> tuple[str, dict[str, Path]]:
    by_sub_ses: dict[str, dict[str, list[Path]]] = {}
    for vhdr in eeg_root.glob("sub-*/ses-*/eeg/*task-ERP_eeg.vhdr"):
        sub_id = vhdr.parts[-4]
        ses_id = vhdr.parts[-3]
        by_sub_ses.setdefault(sub_id, {}).setdefault(ses_id, []).append(vhdr)

    for sub_id in list(by_sub_ses.keys()):
        for ses in list(by_sub_ses[sub_id].keys()):
            by_sub_ses[sub_id][ses] = sorted(by_sub_ses[sub_id][ses])

    if subject:
        subject = subject.strip()
        if subject not in by_sub_ses:
            raise ValueError(f"Subject {subject} not found in {eeg_root}")
        if not all(s in by_sub_ses[subject] and by_sub_ses[subject][s] for s in TARGET_SESSIONS):
            raise ValueError(f"Subject {subject} does not have all sessions {TARGET_SESSIONS}")
        return subject, {s: by_sub_ses[subject][s][0] for s in TARGET_SESSIONS}

    for sub_id in sorted(by_sub_ses.keys()):
        ses_map = by_sub_ses[sub_id]
        if all(s in ses_map and ses_map[s] for s in TARGET_SESSIONS):
            return sub_id, {s: ses_map[s][0] for s in TARGET_SESSIONS}

    raise RuntimeError(f"No subject has all sessions {TARGET_SESSIONS}")


def main():
    args = parse_args()
    eeg_root = Path(args.eeg_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not is_spikify_available():
        raise SystemExit("TTFS encoding requires spikify. Install: pip install innuce-spikify")

    subject, ses_to_file = find_subject_files(eeg_root, args.subject if args.subject else None)
    print(f"Using subject: {subject}")
    for ses in TARGET_SESSIONS:
        print(f"  {ses}: {ses_to_file[ses].name}")

    delta_enc = DeltaModulationEncoder(
        n_thresholds=args.n_thresholds,
        threshold_base=args.delta_threshold_base_eeg,
        scale=args.delta_scale,
    )

    fig, axes = plt.subplots(len(TARGET_SESSIONS), 3, figsize=(15, 3 * len(TARGET_SESSIONS)))
    if len(TARGET_SESSIONS) == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, ses in enumerate(TARGET_SESSIONS):
        vhdr = ses_to_file[ses]
        data, info = load_imu_brainvision(vhdr, apply_scaling=True)

        idx = select_eeg_channel_indices(info.get("ch_names", []), max_channels=3)
        if len(idx) < 3:
            raise RuntimeError(f"{vhdr.name}: cannot select 3 EEG channels")
        eeg = data[idx].astype(np.float64)
        eeg = _resample_multichannel(eeg, float(info.get("sfreq", 100.0)), float(args.resample_rate))

        start = max(0, int(args.start_sec * args.resample_rate))
        seg_len = max(2, int(args.segment_sec * args.resample_rate))
        end = min(eeg.shape[1], start + seg_len)
        if end - start < 2:
            raise RuntimeError(f"{vhdr.name}: segment too short after slicing")
        eeg_seg = eeg[:, start:end]

        delta_spikes = delta_enc.encode(eeg_seg)
        tts_spikes = encode_temporal_multichannel(
            eeg_seg, "time_to_first_spike", interval=int(args.tts_interval)
        )

        ax_raw = axes[r, 0]
        for ch in range(eeg_seg.shape[0]):
            ax_raw.plot(eeg_seg[ch], linewidth=0.8, label=f"ch{ch}")
        ax_raw.set_title(f"{ses} raw ({eeg_seg.shape[1]} samples)")
        ax_raw.set_ylabel("amp")
        if r == 0:
            ax_raw.legend(loc="upper right", fontsize=7, ncol=3)

        ax_delta = axes[r, 1]
        ax_delta.imshow(delta_spikes, aspect="auto", interpolation="nearest", cmap="gray_r")
        ax_delta.set_title(f"{ses} delta spikes ({delta_spikes.shape[0]}x{delta_spikes.shape[1]})")
        ax_delta.set_ylabel("unit")

        ax_tts = axes[r, 2]
        ax_tts.imshow(tts_spikes, aspect="auto", interpolation="nearest", cmap="gray_r")
        ax_tts.set_title(f"{ses} tts spikes ({tts_spikes.shape[0]}x{tts_spikes.shape[1]})")
        ax_tts.set_ylabel("unit")

    axes[-1, 0].set_xlabel("time")
    axes[-1, 1].set_xlabel("time")
    axes[-1, 2].set_xlabel("time")
    fig.suptitle(
        f"EEG encoder comparison | {subject} | delta vs TTFS(interval={args.tts_interval})",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = out_dir / f"{subject}_delta_vs_tts_4class.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

