"""
Read IMU data from BrainVision .dat (or .eeg) files using .vhdr metadata.
No MNE dependency: parses .vhdr and reads binary manually.

Usage:
  from read_imu_dat import load_imu_brainvision

  data, info = load_imu_brainvision("sourcedata/sub-01/ses-01/imu/sub-01_ses-01_task-ERP_imu.vhdr")
  # data: (n_channels, n_times), float, physical units
  # info: dict with 'ch_names', 'sfreq', 'times', 'scales', 'n_channels', 'n_points'

  # Print time-series values:
  print(data)                    # full array (27, n_times)
  print(data[0, :])              # one channel (e.g. HaccX), all time points
  print(data[:, :10])             # all channels, first 10 samples
  print(info["times"], data[0])   # time (s) and one channel

  Channel naming (so you can tell X/Y/Z and acc vs gyro):
  - Name format: [Place][Sensor][Axis], e.g. HaccX = Head accelerometer X.
  - Place: H=head, L=left ankle, R=right ankle.
  - Sensor: acc=accelerometer, gyro=gyroscope, mag=magnetometer.
  - Axis: X, Y, Z.
  - Example: data[info["ch_names"].index("HaccX")] = head acc X; "HgyroZ" = head gyro Z.
"""

from pathlib import Path
import re
import numpy as np

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None


def head_channel_indices(ch_names: list[str]) -> list[tuple[int, str]]:
    """Return (index, name) for all channels whose name starts with 'H' (head)."""
    return [(i, name) for i, name in enumerate(ch_names) if name.upper().startswith("H")]


def plot_imu_head(
    data: np.ndarray,
    info: dict,
    duration_sec: float | None = 60.0,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """
    Plot IMU time series for head (H) channels only.
    Three subplots: Accelerometer (X,Y,Z), Gyroscope (X,Y,Z), Magnetometer (X,Y,Z).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Plotting requires matplotlib: pip install matplotlib")

    ch_names = info["ch_names"]
    times = info["times"]
    head = head_channel_indices(ch_names)
    if not head:
        raise ValueError("No head (H*) channels found in ch_names")

    # Limit time range: if duration_sec is given, select a random window of that length;
    # otherwise use the full duration.
    sfreq = info["sfreq"]
    if sfreq is None or sfreq <= 0:
        raise ValueError("Sampling frequency 'sfreq' must be a positive number in info.")
    n_total = data.shape[1]
    if duration_sec is not None and duration_sec > 0:
        n_samp = int(sfreq * duration_sec)
        n_samp = max(1, min(n_samp, n_total))
        if n_samp < n_total:
            # pick a random start so that the window fits
            start_idx = np.random.randint(0, n_total - n_samp + 1)
        else:
            start_idx = 0
    else:
        start_idx = 0
        n_samp = n_total
    end_idx = start_idx + n_samp
    t = times[start_idx:end_idx]
    data_plot = data[:, start_idx:end_idx]

    # Group head channels by sensor
    roles = channel_roles(ch_names)
    head_acc = [(i, name) for i, name in roles["acc"] if name.upper().startswith("H")]
    head_gyro = [(i, name) for i, name in roles["gyro"] if name.upper().startswith("H")]
    head_mag = [(i, name) for i, name in roles["mag"] if name.upper().startswith("H")]

    # Ignore magnetometer for now
    # fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    fig.suptitle("Accelerometer and Gyroscope(Head only)", fontsize=12)

    for ax, group, title in [
        (axes[0], head_acc, "Accelerometer (HaccX, HaccY, HaccZ)"),
        (axes[1], head_gyro, "Gyroscope (HgyroX, HgyroY, HgyroZ)"),
        # (axes[2], head_mag, "Magnetometer (HmagX, HmagY, HmagZ)"),
    ]:
        if not group:
            continue
        for idx, name in group:
            ax.plot(t, data_plot[idx], label=name, alpha=0.9)
        ax.set_ylabel("a.u.")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved:", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_imu_raw_vs_denoised(
    data_raw: np.ndarray,
    info: dict,
    data_denoised: np.ndarray | None = None,
    window_length: int = 11,
    polyorder: int = 3,
    duration_sec: float | None = 30.0,
    save_path: str | Path | None = None,
    place: str = "H",
    axis: str = "X",
    show: bool = True,
):
    """
    Plot IMU raw vs Savitzky-Golay denoised. Only two lines per subplot: Raw (gray dashed)
    and Denoised (solid), so only two colors.

    - Raw = unprocessed sensor signal from the device.
    - Denoised = same signal after S-G smoothing (noise reduced, shape preserved).

    Parameters
    ----------
    data_raw : (n_channels, n_times) from load_imu_brainvision
    info : from load_imu_brainvision
    data_denoised : if None, denoise is applied to data_raw via denoise_imu_savgol
    window_length, polyorder : S-G filter params (used only if data_denoised is None)
    duration_sec : plot first N seconds (None = all)
    save_path : save figure path
    place : "H" | "L" | "R" for head / left ankle / right ankle
    axis : "X" | "Y" | "Z" to show only that axis (one channel per sensor); "all" for X,Y,Z
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Plotting requires matplotlib: pip install matplotlib")

    ch_names = info["ch_names"]
    times = info["times"]
    sfreq = info["sfreq"]
    if sfreq is None or sfreq <= 0:
        raise ValueError("Sampling frequency 'sfreq' must be a positive number in info.")
    n_total = data_raw.shape[1]
    if duration_sec is not None and duration_sec > 0:
        n_samp = int(sfreq * duration_sec)
        n_samp = max(1, min(n_samp, n_total))
        if n_samp < n_total:
            start_idx = np.random.randint(0, n_total - n_samp + 1)
        else:
            start_idx = 0
    else:
        start_idx = 0
        n_samp = n_total
    end_idx = start_idx + n_samp
    t = times[start_idx:end_idx]
    raw = data_raw[:, start_idx:end_idx]

    if data_denoised is None:
        data_denoised = denoise_imu_savgol(
            data_raw, info, window_length=window_length, polyorder=polyorder, channels="acc_gyro"
        )
    denoised = data_denoised[:, start_idx:end_idx]

    roles = channel_roles(ch_names)
    prefix = place.upper()
    axis_upper = axis.upper()
    # Pick channels: single axis (e.g. HaccX, HgyroX) or all X,Y,Z
    def filter_axis(lst):
        if axis_upper == "ALL":
            return lst
        return [(i, name) for i, name in lst if name.upper().endswith(axis_upper)]

    acc_list = filter_axis([(i, name) for i, name in roles["acc"] if name.upper().startswith(prefix)])
    gyro_list = filter_axis([(i, name) for i, name in roles["gyro"] if name.upper().startswith(prefix)])
    if not acc_list and not gyro_list:
        raise ValueError(f"No acc/gyro channels found for place '{place}' axis '{axis}' (e.g. axis=X)")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
    fig.suptitle(f"IMU Raw vs Denoised — {place} (axis {axis_upper})", fontsize=12)

    for ax, group, title in [
        (axes[0], acc_list, "Accelerometer"),
        (axes[1], gyro_list, "Gyroscope"),
    ]:
        if not group:
            ax.set_visible(False)
            continue
        for k, (idx, name) in enumerate(group):
            ax.plot(t, raw[idx], color="gray", linestyle="--", alpha=0.7, label=("Raw" if k == 0 else None))
            ax.plot(t, denoised[idx], color=f"C{k % 3}", alpha=0.9, label=(f"Denoised {name[-1]}" if axis_upper == "ALL" else ("Denoised (S-G)" if k == 0 else None)))
        ax.set_ylabel("a.u.")
        ax.set_title(title + (f" ({group[0][1]})" if len(group) == 1 else ""))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved:", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def channel_roles(ch_names: list[str]) -> dict:
    """
    Map channel names to sensor type and axis so you can tell acc X/Y/Z vs gyro X/Y/Z.

    Returns
    -------
    dict with keys:
      'acc': list of (channel_index, name) for accelerometer (X,Y,Z per place)
      'gyro': list of (channel_index, name) for gyroscope
      'mag': list of (channel_index, name) for magnetometer
    Name format: [Place][Sensor][Axis], e.g. HaccX, LgyroY, RmagZ.
    """
    roles = {"acc": [], "gyro": [], "mag": []}
    for i, name in enumerate(ch_names):
        n = name.upper()
        if "ACC" in n and ("X" in n or "Y" in n or "Z" in n):
            roles["acc"].append((i, name))
        elif "GYRO" in n and ("X" in n or "Y" in n or "Z" in n):
            roles["gyro"].append((i, name))
        elif "MAG" in n and ("X" in n or "Y" in n or "Z" in n):
            roles["mag"].append((i, name))
    return roles


def denoise_imu_savgol(
    data: np.ndarray,
    info: dict,
    window_length: int = 11,
    polyorder: int = 3,
    channels: str = "acc_gyro",
    mode: str = "nearest",
) -> np.ndarray:
    """
    Denoise IMU with Savitzky-Golay filter (Low-cost IMU Data Denoising using S-G Filters).
    Applied along time (axis=1) per channel. Requires scipy.

    Parameters
    ----------
    data : (n_channels, n_times)
    info : from load_imu_brainvision (need ch_names for acc/gyro selection)
    window_length : 2L+1, must be odd and > polyorder. Larger = smoother.
    polyorder : polynomial order N. Typical 2–5.
    channels : "acc_gyro" | "acc" | "gyro" | "all"
    mode : passed to savgol_filter for boundary handling ("nearest", "mirror", etc.)

    Returns
    -------
    denoised : same shape as data (selected channels filtered; rest copied).
    """
    if savgol_filter is None:
        raise ImportError("denoise_imu_savgol requires scipy: pip install scipy")
    if window_length % 2 == 0:
        window_length += 1
    if polyorder >= window_length:
        polyorder = max(2, window_length - 1)

    roles = channel_roles(info["ch_names"])
    idx_to_filter = []
    if channels in ("acc_gyro", "acc"):
        idx_to_filter.extend([i for i, _ in roles["acc"]])
    if channels in ("acc_gyro", "gyro"):
        idx_to_filter.extend([i for i, _ in roles["gyro"]])
    if channels == "all":
        idx_to_filter = list(range(data.shape[0]))

    out = data.copy()
    for i in idx_to_filter:
        out[i] = savgol_filter(
            data[i].astype(np.float64),
            window_length=window_length,
            polyorder=polyorder,
            axis=0,
            mode=mode,
        )
    return out


def _parse_vhdr(vhdr_path: Path) -> dict:
    """Parse BrainVision .vhdr; return dict with metadata and channel scales."""
    text = vhdr_path.read_text(encoding="utf-8", errors="replace")
    out = {
        "data_file": None,
        "n_channels": None,
        "n_points": None,
        "sampling_interval_µs": None,
        "binary_format": "INT_16",
        "multiplexed": True,
        "big_endian": False,
        "ch_names": [],
        "scales": [],
    }

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("DataFile="):
            out["data_file"] = line.split("=", 1)[1].strip()
        elif line.startswith("NumberOfChannels="):
            out["n_channels"] = int(line.split("=", 1)[1].strip())
        elif line.startswith("DataPoints="):
            out["n_points"] = int(line.split("=", 1)[1].strip())
        elif line.startswith("SamplingInterval="):
            out["sampling_interval_µs"] = float(line.split("=", 1)[1].strip())
        elif line.startswith("BinaryFormat="):
            out["binary_format"] = line.split("=", 1)[1].strip()
        elif line.startswith("UseBigEndianOrder="):
            out["big_endian"] = line.split("=", 1)[1].strip().upper() == "YES"
        elif line.startswith("Ch") and "=" in line:
            # Ch1=HaccX,,0.000362545,a.u.
            m = re.match(r"Ch\d+=(.+)", line)
            if m:
                rest = m.group(1)
                parts = rest.split(",")
                name = parts[0].strip()
                scale = float(parts[2].strip()) if len(parts) > 2 else 1.0
                out["ch_names"].append(name)
                out["scales"].append(scale)

    return out


def _resolve_data_file(vhdr_path: Path, data_file_name: str) -> Path:
    """Resolve path to binary data file; try .eeg then .dat if name from vhdr not found."""
    dir_path = vhdr_path.parent
    base = Path(data_file_name).stem  # e.g. sub-01_ses-01_task-ERP_imu
    # Try name as in vhdr first
    candidate = dir_path / data_file_name
    if candidate.exists():
        return candidate
    # Try .dat (paper format)
    candidate_dat = dir_path / f"{base}.dat"
    if candidate_dat.exists():
        return candidate_dat
    return dir_path / data_file_name  # may not exist; caller will get IOError


def load_imu_brainvision(vhdr_or_dat_path: str | Path, apply_scaling: bool = True):
    """
    Load IMU time-series from BrainVision format.

    Parameters
    ----------
    vhdr_or_dat_path : str or Path
        Path to .vhdr file, or to .dat/.eeg file (same-directory .vhdr will be used).
    apply_scaling : bool
        If True, multiply raw INT_16 by per-channel scale from .vhdr.

    Returns
    -------
    data : np.ndarray, shape (n_channels, n_times), float
        IMU time-series in physical units (if apply_scaling=True).
    info : dict
        'ch_names', 'sfreq', 'times', 'scales', 'n_channels', 'n_points', 'data_file'.
    """
    path = Path(vhdr_or_dat_path).resolve()

    if path.suffix.lower() in (".dat", ".eeg"):
        data_path = path
        vhdr_path = path.with_suffix(".vhdr")
        if not vhdr_path.exists():
            raise FileNotFoundError(f"Expected .vhdr not found: {vhdr_path}")
    else:
        vhdr_path = path
        if vhdr_path.suffix.lower() != ".vhdr":
            raise ValueError("Path must be .vhdr, .dat, or .eeg")
        meta = _parse_vhdr(vhdr_path)
        data_path = _resolve_data_file(vhdr_path, meta["data_file"])
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_path} (vhdr points to {meta['data_file']})"
            )

    meta = _parse_vhdr(vhdr_path)
    n_ch = meta["n_channels"]
    n_pts = meta["n_points"]
    if not meta["ch_names"] or len(meta["ch_names"]) != n_ch:
        meta["ch_names"] = [f"Ch{i+1}" for i in range(n_ch)]
    if not meta["scales"]:
        meta["scales"] = [1.0] * n_ch
    elif len(meta["scales"]) != n_ch:
        meta["scales"] = meta["scales"] + [1.0] * (n_ch - len(meta["scales"]))

    dtype = np.int16
    order = ">i2" if meta["big_endian"] else "<i2"
    raw = np.fromfile(data_path, dtype=order)
    if raw.size != n_pts * n_ch:
        raise ValueError(
            f"File size mismatch: got {raw.size} elements, expected {n_pts} * {n_ch} = {n_pts * n_ch}"
        )
    # MULTIPLEXED: [s0_ch0, s0_ch1, ..., s0_chN, s1_ch0, ...] -> (n_pts, n_ch)
    raw = raw.reshape(n_pts, n_ch).T  # (n_channels, n_times)

    if apply_scaling:
        scales = np.array(meta["scales"], dtype=np.float64)
        data = raw.astype(np.float64) * scales[:, np.newaxis]
    else:
        data = raw.astype(np.float64)

    # Sampling: SamplingInterval in µs -> Hz = 1e6 / interval
    interval_us = meta["sampling_interval_µs"]
    sfreq = 1e6 / interval_us if interval_us else None
    times = np.arange(n_pts, dtype=np.float64) / sfreq if sfreq else np.arange(n_pts)

    info = {
        "ch_names": meta["ch_names"],
        "sfreq": sfreq,
        "times": times,
        "scales": meta["scales"],
        "n_channels": n_ch,
        "n_points": n_pts,
        "data_file": str(data_path),
    }
    return data, info


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    path_args = [a for a in args if not a.startswith("--")]
    sourcedata_root = Path(r"D:\xia\CMU\18743-Neuromorphic\Multi_Sensory_NeuTNN\osfstorage-archive\sourcedata")
    default_path = sourcedata_root / "sub-01" / "ses-01" / "imu" / "sub-01_ses-01_task-ERP_imu.dat"
    if not default_path.exists():
        default_path = sourcedata_root / "sub-01" / "ses-01" / "imu" / "sub-01_ses-01_task-ERP_imu.vhdr"
    path = path_args[0] if path_args else str(default_path)

    # Default: plot full duration (None => use all samples)
    plot_sec = None
    save_path = None
    raw_vs_denoised = False
    place = "H"
    axis = "X"
    sg_window = 11
    sg_poly = 3
    for i, a in enumerate(args):
        if a == "--plot":
            pass
        elif a == "--raw-vs-denoise":
            raw_vs_denoised = True
        elif a.startswith("--axis="):
            axis = a.split("=", 1)[1].strip().upper() or "X"
        elif a.startswith("--duration="):
            plot_sec = float(a.split("=", 1)[1])
        elif a.startswith("--save="):
            save_path = a.split("=", 1)[1]
        elif a.startswith("--place="):
            place = a.split("=", 1)[1].strip().upper() or "H"
        elif a.startswith("--window-length="):
            sg_window = int(a.split("=", 1)[1])
        elif a.startswith("--polyorder="):
            sg_poly = int(a.split("=", 1)[1])
    do_plot = "--plot" in args

    # Determine output directory for figures (always under ./graphs relative to current CWD)
    graphs_dir = Path.cwd() / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Normalize/save figure path:
    # - If user provided --save and it is relative, place it inside ./graphs
    # - If user did not provide --save but requested a plot, auto-generate a filename in ./graphs
    if save_path:
        sp = Path(save_path)
        if not sp.is_absolute():
            save_path = str(graphs_dir / sp.name)
        else:
            save_path = str(sp)
    else:
        auto_name = None
        if raw_vs_denoised:
            auto_name = f"imu_raw_vs_denoised_{place}_{axis}.png"
        elif do_plot:
            auto_name = "imu_head.png"
        if auto_name is not None:
            save_path = str(graphs_dir / auto_name)

    print("Loading:", path)
    data, info = load_imu_brainvision(path)
    print("Shape:", data.shape, "(n_channels, n_times)")
    print("SFreq:", info["sfreq"], "Hz")
    print("Channels:", info["ch_names"][:5], "...")
    print("Times (s):", info["times"][0], "...", info["times"][-1])

    roles = channel_roles(info["ch_names"])
    print("\nAccelerometer (X,Y,Z per place):", [name for _, name in roles["acc"]])
    print("Gyroscope (X,Y,Z per place):     ", [name for _, name in roles["gyro"]])
    print("Magnetometer (X,Y,Z per place):  ", [name for _, name in roles["mag"]])

    if raw_vs_denoised:
        plot_imu_raw_vs_denoised(
            data, info,
            data_denoised=None,
            window_length=sg_window,
            polyorder=sg_poly,
            duration_sec=plot_sec,
            save_path=save_path,
            place=place,
            axis=axis,
        )
    elif do_plot:
        plot_imu_head(data, info, duration_sec=plot_sec, save_path=save_path)
    else:
        print("\nFirst 10 samples of HaccX (time_s, value):")
        t = info["times"]
        for i in range(min(10, data.shape[1])):
            print(f"  {t[i]:.4f} s  {data[0, i]:.6f}")
        print("\nFirst 5 time points, all channels (shape 27x5):")
        print(data[:, :5])
        print("\nTip: --plot = head IMU; --raw-vs-denoise = raw vs denoised; "
              "--axis=X (or Y,Z,all); --duration=10 (window length, default=all); "
              "--save=fig.png; --place=H|L|R.")
