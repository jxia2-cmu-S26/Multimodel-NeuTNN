"""
Visualize EEG channels from BrainVision files using the shared loader in read_imu_dat.py.

Usage examples:
    # Pipeline mode: process all EEG files under osfstorage-archive sub-01..sub-24
    python plot_eeg_dat.py

    # Explicit pipeline root
    python plot_eeg_dat.py --pipeline --root=D:/.../osfstorage-archive

    # Pipeline with encoded-output visualization (method configurable)
    python plot_eeg_dat.py --pipeline --method=moving_window --window-length=8

    # Single file mode
    python plot_eeg_dat.py D:/.../sub-01_ses-01_task-ERP_eeg.vhdr
    python plot_eeg_dat.py D:/.../sub-01_ses-01_task-ERP_eeg.eeg --save=eeg_plot.png
"""

from pathlib import Path
import sys
import numpy as np

from read_imu_dat import load_imu_brainvision
from spikify_temporal_encode import (
    TEMPORAL_METHODS,
    encode_temporal_multichannel,
    is_spikify_available,
)


def fixed_eeg_channel_indices(ch_names: list[str]) -> list[tuple[int, str]]:
    """Return fixed channels FZ, CZ, PZ in that order; raise if any is missing."""
    required = ["FZ", "CZ", "PZ"]
    name_to_idx = {name.upper(): i for i, name in enumerate(ch_names)}

    missing = [name for name in required if name not in name_to_idx]
    if missing:
        raise ValueError(f"Missing required EEG channels: {missing}. Available first 20: {ch_names[:20]}")

    return [(name_to_idx[name], ch_names[name_to_idx[name]]) for name in required]


def plot_eeg_channels(
    data: np.ndarray,
    info: dict,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Plot full-length stacked EEG traces for fixed channels FZ, CZ, PZ."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Plotting requires matplotlib: pip install matplotlib")

    ch_names = info["ch_names"]
    times = info["times"]
    sfreq = info["sfreq"]
    if sfreq is None or sfreq <= 0:
        raise ValueError("Sampling frequency 'sfreq' must be a positive number in info.")

    selected = fixed_eeg_channel_indices(ch_names)
    t = times
    traces = np.array([data[idx, :] for idx, _ in selected], dtype=np.float64)

    # Normalize each channel so stacked display is readable across amplitude scales.
    mu = traces.mean(axis=1, keepdims=True)
    sigma = traces.std(axis=1, keepdims=True)
    traces = (traces - mu) / (sigma + 1e-12)

    spacing = 4.0
    offsets = np.arange(len(selected))[::-1] * spacing

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle("EEG channels (stacked)", fontsize=12)
    for k, (_, name) in enumerate(selected):
        ax.plot(t, traces[k] + offsets[k], linewidth=0.9, label=name)

    ax.set_yticks(offsets)
    ax.set_yticklabels([name for _, name in selected])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel (normalized)")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved:", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _pick_default_eeg_path() -> Path:
    root = Path(r"D:\xia\CMU\18743-Neuromorphic\Multi_Sensory_NeuTNN\osfstorage-archive\sourcedata")
    candidate = root / "sub-01" / "ses-01" / "eeg" / "sub-01_ses-01_task-ERP_eeg.vhdr"
    if candidate.exists():
        return candidate
    found = next(root.glob("sub-*/ses-*/eeg/*task-ERP_eeg.vhdr"), None)
    if found is None:
        raise FileNotFoundError("No EEG .vhdr found under osfstorage-archive/sourcedata")
    return found


def _default_osf_root() -> Path:
    return Path(__file__).resolve().parent.parent / "osfstorage-archive"


def discover_eeg_vhdr_files(osf_root: Path) -> list[Path]:
    """Find *_eeg.vhdr for sub-01..sub-24 under osf root, excluding sourcedata."""
    osf_root = Path(osf_root).resolve()

    files: list[Path] = []
    subj_set = {f"sub-{i:02d}" for i in range(1, 25)}
    for p in osf_root.glob("sub-*/ses-*/eeg/*_eeg.vhdr"):
        if "task-ssvep" in p.name.lower():
            continue
        parts = set(p.parts)
        if parts.intersection(subj_set):
            files.append(p.resolve())

    return sorted(set(files))


def output_path_for_input(vhdr_path: Path) -> Path:
    """Save plot into the same eeg folder as the input file."""
    stem = vhdr_path.stem
    out_name = f"{stem}_FZ_CZ_PZ.png"
    return vhdr_path.parent / out_name


def output_path_for_encoded(vhdr_path: Path, method: str) -> Path:
    stem = vhdr_path.stem
    out_name = f"{stem}_{method}_encoded.png"
    return vhdr_path.parent / out_name


def select_fixed_eeg_data(data: np.ndarray, info: dict) -> tuple[np.ndarray, list[str], np.ndarray]:
    selected = fixed_eeg_channel_indices(info["ch_names"])
    indices = [idx for idx, _ in selected]
    names = [name for _, name in selected]
    data_sel = data[indices].astype(np.float64)
    return data_sel, names, info["times"]


def encode_fixed_eeg(
    data: np.ndarray,
    info: dict,
    method: str,
    spikify_kw: dict,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data_sel, ch_names_sel, times = select_fixed_eeg_data(data, info)
    spikes = encode_temporal_multichannel(data_sel, method, **spikify_kw)
    t = times[: spikes.shape[1]]
    return spikes, t, ch_names_sel


def _unit_labels(method: str, ch_names_sel: list[str]) -> list[str]:
    if method in ("moving_window", "step_forward", "threshold_based_representation"):
        pos = [f"{n}+" for n in ch_names_sel]
        neg = [f"{n}-" for n in ch_names_sel]
        return pos + neg
    return ch_names_sel


def plot_encoded_spikes(
    spikes: np.ndarray,
    t: np.ndarray,
    method: str,
    ch_names_sel: list[str],
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Plot encoded spike raster/intensity (units x time)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Plotting requires matplotlib: pip install matplotlib")

    labels = _unit_labels(method, ch_names_sel)
    fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))
    extent = [float(t[0]), float(t[-1]), -0.5, spikes.shape[0] - 0.5] if len(t) > 1 else None
    ax.imshow(spikes, aspect="auto", interpolation="nearest", cmap="gray_r", origin="lower", extent=extent)
    ax.set_title(f"Encoded spikes ({method})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Units")

    if len(labels) == spikes.shape[0]:
        ax.set_yticks(np.arange(spikes.shape[0]))
        ax.set_yticklabels(labels)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved:", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_pipeline(osf_root: Path, spikify_method: str, spikify_kw: dict) -> None:
    if not is_spikify_available():
        raise ImportError("spikify is required for encoded plots. Install: pip install innuce-spikify")

    files = discover_eeg_vhdr_files(osf_root)
    if not files:
        raise FileNotFoundError(f"No *_eeg.vhdr files found under {osf_root}")

    ok = 0
    fail = 0
    print(f"Pipeline start: found {len(files)} EEG file(s) under {Path(osf_root).resolve()}")

    for i, vhdr_path in enumerate(files, start=1):
        out_raw = output_path_for_input(vhdr_path)
        out_enc = output_path_for_encoded(vhdr_path, spikify_method)
        try:
            data, info = load_imu_brainvision(vhdr_path)
            selected = fixed_eeg_channel_indices(info["ch_names"])
            print(f"[{i}/{len(files)}] {vhdr_path.name} | channels={[name for _, name in selected]}")
            plot_eeg_channels(data, info, save_path=out_raw, show=False)
            spikes, t, ch_names_sel = encode_fixed_eeg(data, info, method=spikify_method, spikify_kw=spikify_kw)
            plot_encoded_spikes(
                spikes,
                t,
                method=spikify_method,
                ch_names_sel=ch_names_sel,
                save_path=out_enc,
                show=False,
            )
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(files)}] FAILED: {vhdr_path} -> {e}")

    print(f"Pipeline done: success={ok}, failed={fail}")


if __name__ == "__main__":
    args = sys.argv[1:]
    path_args = [a for a in args if not a.startswith("--")]

    pipeline = True
    root_path = _default_osf_root()
    save_path = None
    spikify_method = "moving_window"
    spikify_kw = {
        "window_length": 8,
        "threshold": 0.1,
        "factor": 0.5,
        "n_max": 4,
        "t_min": 2,
        "t_max": 6,
        "num_bits": 8,
        "interval": 8,
    }

    for a in args:
        if a == "--pipeline":
            pipeline = True
        elif a.startswith("--root="):
            root_path = Path(a.split("=", 1)[1])
        elif a.startswith("--method="):
            spikify_method = a.split("=", 1)[1].strip().lower()
        elif a.startswith("--window-length="):
            spikify_kw["window_length"] = int(a.split("=", 1)[1])
        elif a.startswith("--threshold="):
            spikify_kw["threshold"] = float(a.split("=", 1)[1])
        elif a.startswith("--factor="):
            spikify_kw["factor"] = float(a.split("=", 1)[1])
        elif a.startswith("--n-max="):
            spikify_kw["n_max"] = int(a.split("=", 1)[1])
        elif a.startswith("--t-min="):
            spikify_kw["t_min"] = int(a.split("=", 1)[1])
        elif a.startswith("--t-max="):
            spikify_kw["t_max"] = int(a.split("=", 1)[1])
        elif a.startswith("--num-bits="):
            spikify_kw["num_bits"] = int(a.split("=", 1)[1])
        elif a.startswith("--interval="):
            spikify_kw["interval"] = int(a.split("=", 1)[1])
        elif a.startswith("--save="):
            save_path = a.split("=", 1)[1]

    if spikify_method not in TEMPORAL_METHODS:
        raise ValueError(f"Unknown --method={spikify_method!r}. Choose from: {TEMPORAL_METHODS}")

    # Positional path triggers single-file mode.
    if path_args:
        pipeline = False

    if pipeline:
        run_pipeline(root_path, spikify_method=spikify_method, spikify_kw=spikify_kw)
        raise SystemExit(0)

    path = Path(path_args[0]) if path_args else _pick_default_eeg_path()

    graphs_dir = Path.cwd() / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    if save_path:
        sp = Path(save_path)
        if not sp.is_absolute():
            save_path = str(graphs_dir / sp.name)
        else:
            save_path = str(sp)
    else:
        save_path = str(graphs_dir / "eeg_channels.png")

    print("Loading:", path)
    data, info = load_imu_brainvision(path)
    print("Shape:", data.shape, "(n_channels, n_times)")
    print("SFreq:", info["sfreq"], "Hz")

    selected = fixed_eeg_channel_indices(info["ch_names"])
    print("Selected EEG channels:", [name for _, name in selected])

    plot_eeg_channels(
        data,
        info,
        save_path=save_path,
    )

    if not is_spikify_available():
        print("WARNING: spikify not installed, skipping encoded plot. Install: pip install innuce-spikify")
        raise SystemExit(0)

    spikes, t, ch_names_sel = encode_fixed_eeg(data, info, method=spikify_method, spikify_kw=spikify_kw)
    enc_save = str(Path(save_path).with_name(Path(save_path).stem + f"_{spikify_method}_encoded.png"))
    plot_encoded_spikes(
        spikes,
        t,
        method=spikify_method,
        ch_names_sel=ch_names_sel,
        save_path=enc_save,
    )
