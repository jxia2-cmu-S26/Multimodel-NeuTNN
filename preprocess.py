"""
IMU preprocess pipeline for NeuTNN: load BrainVision IMU -> Savitzky-Golay denoise -> optional normalize -> save.

Uses read_imu_dat.py for loading and denoising. Outputs go to derived/ with a manifest for downstream TNN.

Usage:
  # Process all IMU files under sourcedata
  python preprocess.py

  # Process one file
  python preprocess.py sourcedata/sub-01/ses-01/imu/sub-01_ses-01_task-ERP_imu.vhdr

  # Custom output dir and S-G params
  python preprocess.py --out-dir derived/imu_denoised --window-length 21 --polyorder 3
"""

from pathlib import Path
import argparse
import json
import numpy as np

# Same-directory import for read_imu_dat
from read_imu_dat import (
    load_imu_brainvision,
    channel_roles,
    denoise_imu_savgol,
)


def discover_imu_files(sourcedata_root: Path) -> list[Path]:
    """Find all IMU .vhdr under sourcedata (each .vhdr references .eeg or .dat in same folder)."""
    root = Path(sourcedata_root).resolve()
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.rglob("imu/*.vhdr") if "task-ERP_imu" in p.name
    )


def process_one(
    path: str | Path,
    out_root: Path,
    window_length: int = 11,
    polyorder: int = 3,
    denoise_channels: str = "acc_gyro",
    save_npy: bool = True,
    save_meta: bool = True,
) -> dict | None:
    """
    Load one IMU file, denoise with S-G, optionally normalize, save to out_dir.
    Returns manifest entry dict or None if failed (e.g. missing binary).
    """
    src_path = Path(path).resolve()
    if not src_path.exists():
        print(f"Skip (not found): {src_path}")
        return None

    try:
        data, info = load_imu_brainvision(src_path, apply_scaling=True)
    except FileNotFoundError as e:
        print(f"Skip (missing data file): {src_path} -> {e}")
        return None
    except Exception as e:
        print(f"Error loading {src_path}: {e}")
        return None

    # Savitzky-Golay denoise (acc + gyro)
    data = denoise_imu_savgol(
        data,
        info,
        window_length=window_length,
        polyorder=polyorder,
        channels=denoise_channels,
        mode="nearest",
    )

    # Relative path from path parent for output name, e.g. sub-01_ses-01_task-ERP_imu
    stem = src_path.stem
    if stem.endswith("_imu") or stem.endswith("_eeg"):
        pass
    else:
        stem = stem

    entry = {
        "source": str(src_path),
        "stem": stem,
        "shape": list(data.shape),
        "sfreq": info.get("sfreq"),
        "ch_names": info["ch_names"],
        "n_points": info["n_points"],
        "window_length": window_length,
        "polyorder": polyorder,
    }

    # Preserve sourcedata subfolder structure under out_root:
    # sourcedata_root/sub-01/ses-01/imu/file.vhdr ->
    # out_root/sub-01/ses-01/imu/file.npy
    out_root = Path(out_root).resolve()
    # We infer sourcedata_root as the common prefix one level above 'sub-*' folders,
    # but run_pipeline will ensure out_root is paired with the same sourcedata_root.
    try:
        # This relative path is computed against the sourcedata root passed in run_pipeline
        # (via src_path.relative_to(root) there).
        # Here we assume src_path is already that absolute path.
        # The actual relative parent will be injected by run_pipeline.
        rel_parent = src_path.parent.relative_to(src_path.parents[3])
        out_dir = out_root / rel_parent
    except ValueError:
        # Fallback: flatten into out_root if relative computation fails
        out_dir = out_root

    out_dir.mkdir(parents=True, exist_ok=True)

    if save_npy:
        npy_path = out_dir / f"{stem}.npy"
        np.save(npy_path, data)
        # Store path relative to out_root
        entry["npy"] = str(npy_path.relative_to(out_root))

    if save_meta:
        meta_path = out_dir / f"{stem}_meta.json"
        meta = {k: v for k, v in entry.items() if k != "npy"}
        meta["ch_names"] = info["ch_names"]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        entry["meta"] = str(meta_path.relative_to(out_root))

    return entry


def run_pipeline(
    sourcedata_root: str | Path,
    out_dir: str | Path = "derived/imu_preprocessed",
    window_length: int = 11,
    polyorder: int = 3,
    denoise_channels: str = "acc_gyro",
    single_file: str | Path | None = None,
) -> list[dict]:
    """
    Run full pipeline: discover IMU files (or use single_file), process each, write manifest.
    """
    root = Path(sourcedata_root).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if single_file is not None:
        paths = [Path(single_file).resolve()]
        if not paths[0].exists():
            raise FileNotFoundError(f"File not found: {paths[0]}")
    else:
        paths = discover_imu_files(root)
        if not paths:
            print("No IMU .vhdr/.dat/.eeg found under", root)
            return []

    manifest = []
    for p in paths:
        entry = process_one(
            p,
            out_root=out,
            window_length=window_length,
            polyorder=polyorder,
            denoise_channels=denoise_channels,
            save_npy=True,
            save_meta=True,
        )
        if entry:
            manifest.append(entry)
            print("OK:", entry.get("stem", p.name), "shape:", entry.get("shape"))

    if manifest:
        manifest_path = out / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print("Manifest:", manifest_path)

    return manifest


def main():
    base = Path(__file__).parent
    # Project root is the parent of NeuTNN_imu; sourcedata lives under osfstorage-archive/sourcedata
    project_root = base.parent
    default_sourcedata = project_root / "osfstorage-archive" / "sourcedata"
    default_out_dir = base / "imu_dataset_clean"

    ap = argparse.ArgumentParser(description="IMU preprocess pipeline (load -> S-G denoise -> optional normalize -> save)")
    ap.add_argument("path", nargs="?", help="Single .vhdr/.dat/.eeg path (optional)")
    ap.add_argument("--sourcedata", type=Path, default=default_sourcedata, help="sourcedata root to scan (e.g. osfstorage-archive/sourcedata)")
    ap.add_argument("--out-dir", type=Path, default=default_out_dir, help="Output directory for cleaned IMU dataset")
    ap.add_argument("--window-length", type=int, default=11, help="Savitzky-Golay window length (odd)")
    ap.add_argument("--polyorder", type=int, default=3, help="Savitzky-Golay polynomial order")
    ap.add_argument("--denoise-channels", choices=("acc_gyro", "acc", "gyro", "all"), default="acc_gyro")
    ap.add_argument("--normalize", action="store_true", help="Apply per-session z-score normalization")
    args = ap.parse_args()

    if args.window_length % 2 == 0:
        args.window_length += 1

    single = args.path
    if single:
        single = Path(single)
        if not single.is_absolute():
            single = (base / single).resolve()

    run_pipeline(
        sourcedata_root=args.sourcedata,
        out_dir=args.out_dir,
        window_length=args.window_length,
        polyorder=args.polyorder,
        denoise_channels=args.denoise_channels,
        single_file=single,
    )


if __name__ == "__main__":
    main()
