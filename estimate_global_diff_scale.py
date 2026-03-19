from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from read_imu_dat import channel_roles, load_imu_brainvision


# Fixed estimation/report parameters
FIXED_METHOD = "quantile"
FIXED_QUANTILE = 0.9
FIXED_MAD_K = 0.5
FIXED_ONLY_ERP = True
FIXED_MAX_FILES = None
FIXED_N_THRESHOLDS = 3
FIXED_SCALE = 2.0


def _session_to_label(source: str) -> int | None:
    p = Path(source)
    ses_id = None
    for part in p.parts:
        if part.startswith("ses-"):
            ses_id = part.split("-", 1)[1]
            break
    mapping = {"02": 0, "03": 1, "04": 2, "05": 3}
    return mapping.get(ses_id)


def _select_eeg_indices(ch_names: list[str], n_select: int = 3) -> list[int]:
    preferred = ["FZ", "CZ", "PZ"]
    name_to_idx = {n.strip().upper(): i for i, n in enumerate(ch_names)}
    picked = [name_to_idx[n] for n in preferred if n in name_to_idx]
    if len(picked) >= n_select:
        return picked[:n_select]

    blocked = ("EOG", "ECG", "EMG", "TRIG", "STATUS", "MARK", "EXG")
    for i, n in enumerate(ch_names):
        u = n.upper()
        if i in picked:
            continue
        if any(tok in u for tok in blocked):
            continue
        picked.append(i)
        if len(picked) >= n_select:
            break
    return picked[:n_select]


def _get_head_imu_indices(ch_names: list[str], imu_sensor: str) -> list[int]:
    roles = channel_roles(ch_names)
    head_acc = [i for i, n in roles["acc"] if n.upper().startswith("H")]
    head_gyro = [i for i, n in roles["gyro"] if n.upper().startswith("H")]

    if imu_sensor == "acc":
        return head_acc[:3]
    if imu_sensor == "gyro":
        return head_gyro[:3]
    if imu_sensor == "both":
        return head_acc[:3] + head_gyro[:3]
    raise ValueError(f"Unsupported imu_sensor={imu_sensor}")


def _iter_imu_diff_abs(
    imu_dataset_root: Path,
    max_files: Optional[int],
    only_erp: bool,
    imu_sensor: str,
) -> list[np.ndarray]:
    manifest_path = imu_dataset_root / "manifest.json"
    if not manifest_path.exists():
        return []

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    out: list[np.ndarray] = []
    used = 0
    for entry in manifest:
        source = entry.get("source", "")
        if only_erp and "_task-ERP_imu" not in source:
            continue
        if _session_to_label(source) is None:
            continue

        npy_rel = entry.get("npy")
        ch_names = entry.get("ch_names", [])
        if not npy_rel or not ch_names:
            continue

        data_path = imu_dataset_root / npy_rel
        if not data_path.exists():
            continue

        data = np.load(data_path)
        head_idx = _get_head_imu_indices(ch_names, imu_sensor=imu_sensor)
        needed = 6 if imu_sensor == "both" else 3
        if len(head_idx) < needed:
            continue

        data_sel = data[head_idx].astype(np.float64)
        diff_abs = np.abs(np.diff(data_sel, axis=1)).reshape(-1)
        if diff_abs.size == 0:
            continue

        out.append(diff_abs)
        used += 1
        if max_files is not None and used >= max_files:
            break

    return out


def _iter_eeg_diff_abs(
    eeg_root: Path,
    max_files: Optional[int],
    only_erp: bool,
) -> list[np.ndarray]:
    pattern = "sub-*/ses-*/eeg/*task-ERP_eeg.vhdr" if only_erp else "sub-*/ses-*/eeg/*.vhdr"
    vhdrs = sorted(eeg_root.glob(pattern))

    out: list[np.ndarray] = []
    used = 0
    for vhdr in vhdrs:
        source = str(vhdr)
        if _session_to_label(source) is None:
            continue
        try:
            data, info = load_imu_brainvision(vhdr, apply_scaling=True)
        except Exception:
            continue

        idx = _select_eeg_indices(info.get("ch_names", []), n_select=3)
        if len(idx) < 3:
            continue

        data_sel = data[idx].astype(np.float64)
        diff_abs = np.abs(np.diff(data_sel, axis=1)).reshape(-1)
        if diff_abs.size == 0:
            continue

        out.append(diff_abs)
        used += 1
        if max_files is not None and used >= max_files:
            break

    return out


def estimate_threshold_base(
    imu_dataset_root: Path | str | None = None,
    eeg_root: Path | str | None = None,
    mode: str = "both",
    imu_sensor: str = "acc",
    method: str = "quantile",
    quantile: float = 0.9,
    mad_k: float = 1.0,
    max_files: Optional[int] = None,
    only_erp: bool = True,
) -> dict:
    """
    Estimate threshold_base for DeltaModulationEncoder using robust statistics.

        mode: 'imu' | 'eeg' | 'both'
        imu_sensor: IMU channel subset used when mode includes imu:
            - 'acc'  : head accelerometer (3 channels)
            - 'gyro' : head gyroscope (3 channels)
            - 'both' : head accelerometer + gyroscope (6 channels)
    method:
      - 'quantile': threshold_base = median_over_files( quantile(|Δx|, q) )
      - 'mad':      threshold_base = mad_k * median_over_files( MAD(|Δx|) )
    """
    if imu_dataset_root is None:
        imu_dataset_root = Path(__file__).resolve().parent / "imu_dataset_clean"
    else:
        imu_dataset_root = Path(imu_dataset_root).resolve()

    if eeg_root is None:
        eeg_root = Path(__file__).resolve().parent.parent / "osfstorage-archive"
    else:
        eeg_root = Path(eeg_root).resolve()

    per_file_stats: list[float] = []
    n_imu = 0
    n_eeg = 0

    if mode in ("imu", "both"):
        imu_diffs = _iter_imu_diff_abs(
            imu_dataset_root=imu_dataset_root,
            max_files=max_files,
            only_erp=only_erp,
            imu_sensor=imu_sensor,
        )
        n_imu = len(imu_diffs)
        for d in imu_diffs:
            if method == "quantile":
                per_file_stats.append(float(np.quantile(d, quantile)))
            else:
                med = float(np.median(d))
                mad = float(np.median(np.abs(d - med)))
                per_file_stats.append(mad_k * mad)

    if mode in ("eeg", "both"):
        eeg_diffs = _iter_eeg_diff_abs(
            eeg_root=eeg_root,
            max_files=max_files,
            only_erp=only_erp,
        )
        n_eeg = len(eeg_diffs)
        for d in eeg_diffs:
            if method == "quantile":
                per_file_stats.append(float(np.quantile(d, quantile)))
            else:
                med = float(np.median(d))
                mad = float(np.median(np.abs(d - med)))
                per_file_stats.append(mad_k * mad)

    if not per_file_stats:
        raise RuntimeError("No valid files found to estimate threshold_base.")

    th_base = float(np.median(np.asarray(per_file_stats, dtype=np.float64)))
    return {
        "threshold_base": th_base,
        "mode": mode,
        "imu_sensor": imu_sensor,
        "method": method,
        "quantile": quantile,
        "mad_k": mad_k,
        "n_imu_files": n_imu,
        "n_eeg_files": n_eeg,
        "n_total_files": len(per_file_stats),
    }


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Estimate robust threshold_base for delta encoder")
    ap.add_argument("--imu-root", type=str, default=None)
    ap.add_argument("--eeg-root", type=str, default=None)
    ap.add_argument("--mode", choices=["imu", "eeg", "both"], default="both")
    ap.add_argument(
        "--imu-sensor",
        choices=["acc", "gyro", "both"],
        default="acc",
        help="IMU channels used for threshold estimation when mode includes imu",
    )
    return ap


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    result = estimate_threshold_base(
        imu_dataset_root=args.imu_root,
        eeg_root=args.eeg_root,
        mode=args.mode,
        imu_sensor=args.imu_sensor,
        method=FIXED_METHOD,
        quantile=FIXED_QUANTILE,
        mad_k=FIXED_MAD_K,
        max_files=FIXED_MAX_FILES,
        only_erp=FIXED_ONLY_ERP,
    )

    th_base = result["threshold_base"]
    print("=== Threshold Base Estimation ===")
    print(json.dumps(result, indent=2))
    print("\nFixed config:")
    print(
        f"method={FIXED_METHOD}, quantile={FIXED_QUANTILE}, mad_k={FIXED_MAD_K}, "
        f"only_erp={FIXED_ONLY_ERP}, max_files={FIXED_MAX_FILES}, "
        f"n_thresholds={FIXED_N_THRESHOLDS}, scale={FIXED_SCALE}"
    )
    print("\nSuggested encoder thresholds:")
    for i in range(FIXED_N_THRESHOLDS):
        th = th_base * (FIXED_SCALE ** i)
        print(f"  th[{i}] = {th:.8f}")


