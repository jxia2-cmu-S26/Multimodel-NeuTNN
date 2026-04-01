import argparse
import csv
import math
import os
import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EEG_PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "eeg_preprocessed")


def _discover_eeg_splits(root_dir, task="all"):
    """Find EEG runs under root_dir.

    Returns a list of dicts: {vhdr_path, events_path, task, ses}.
    task: "ERP" | "SSVEP" | "all"
    ses: integer session index parsed from "ses-YY" in the path (1..5), or None.
    """
    root = Path(root_dir)
    if not root.is_dir():
        return []
    out = []
    for vhdr in root.rglob("*_task-*_eeg.vhdr"):
        stem = vhdr.name.replace("_eeg.vhdr", "")
        events_path = vhdr.parent / f"{stem}_events.tsv"
        if not events_path.is_file():
            continue
        t = "ERP" if "task-ERP" in stem else "SSVEP" if "task-SSVEP" in stem else None
        if t is None or (task != "all" and task != t):
            continue
        m = re.search(r"ses-(\d+)", str(vhdr))
        ses = int(m.group(1)) if m else None
        out.append(
            dict(
                vhdr_path=str(vhdr),
                events_path=str(events_path),
                task=t,
                ses=ses,
            )
        )
    out.sort(key=lambda d: (d["vhdr_path"], d["events_path"]))
    return out


def _load_raw_and_epochs(vhdr_path, events_tsv_path, tmin=0.0, tmax=0.5, n_scalp_channels=32):
    """Load BrainVision raw and epoch per event.

    Uses only the first n_scalp_channels EEG channels (scalp-EEG); ear-EEG, EOG, and
    IMU channels are excluded so that n_ch <= n_scalp_channels.

    Returns:
      X: (n_epochs, n_ch, n_times) with n_ch = min(n_scalp_channels, available eeg channels)
      event_values: (n_epochs,) raw integer values from events.tsv "value" column
    """
    import mne
    mne.set_log_level("ERROR")
    with warnings.catch_warnings():
        # Some .vmrk files contain malformed marker-version strings; this warning is noisy
        # and does not affect this pipeline because epochs/labels come from _events.tsv.
        warnings.filterwarnings(
            "ignore",
            message=r"MNE-Python currently only supports marker versions 1\.0 and 2\.0.*",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*header versions.*",
            category=RuntimeWarning,
        )
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
    raw.pick_types(eeg=True, misc=False, stim=False)
    data, _ = raw.get_data(return_times=True)
    # Restrict to 32 scalp-EEG channels (dataset also has ear-EEG, EOG, IMU)
    if data.shape[0] > n_scalp_channels:
        data = data[:n_scalp_channels]
    sfreq = raw.info["sfreq"]
    onset, value = [], []
    with open(events_tsv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            onset.append(int(float(row["onset"])))
            value.append(int(row["value"]))
    onset, value = np.array(onset), np.array(value)
    if onset.max() < raw.n_times and onset.min() >= 0:
        sample_onset = onset.astype(int)
    else:
        sample_onset = (onset * sfreq).astype(int)
    n_times = int(round((tmax - tmin) * sfreq))
    n_ch = data.shape[0]
    valid_idx = [i for i, s in enumerate(sample_onset) if s + n_times <= data.shape[1] and s >= 0]
    X = np.zeros((len(valid_idx), n_ch, n_times), dtype=np.float64)
    event_values = np.zeros(len(valid_idx), dtype=np.int64)
    for j, i in enumerate(valid_idx):
        s = sample_onset[i]
        X[j] = data[:, s : s + n_times]
        event_values[j] = int(value[i])
    return X, event_values


def _load_eeg(task="all", label_mode="event", n_scalp_channels=32):
    """Load EEG epochs from eeg_preprocessed (BrainVision .vhdr + _events.tsv).

    Only the first n_scalp_channels (default 32) scalp-EEG channels are used;
    ear-EEG, EOG, and IMU channels are not loaded.

    Args:
      task: "ERP" | "SSVEP" | "all"
      label_mode:
        - "event": labels are derived from events.tsv "value" codes (global remap to 0..C-1)
        - "speed": labels are derived from session "ses-YY" (ses-01..ses-05 -> 0..4)
      n_scalp_channels: max number of EEG channels to use (32 for scalp-EEG only).

    Returns:
      X_all: (N, C, T) float32 with C = n_scalp_channels
      y_all: (N,) int64 remapped labels
    """
    if not os.path.isdir(EEG_PREPROCESSED_DIR):
        raise FileNotFoundError(f"Missing EEG folder: {EEG_PREPROCESSED_DIR}")
    splits = _discover_eeg_splits(EEG_PREPROCESSED_DIR, task=task)
    if not splits:
        raise FileNotFoundError("No *_task-*_eeg.vhdr + *_events.tsv found under eeg_preprocessed.")
    if label_mode not in ("event", "speed"):
        raise ValueError("label_mode must be 'event' or 'speed'")

    # Report which subjects and how many runs are being loaded
    subjects = sorted({re.search(r"sub-(\d+)", s["vhdr_path"]).group(0) for s in splits if re.search(r"sub-(\d+)", s["vhdr_path"])})
    print(f"Discovered {len(splits)} runs | subjects: {subjects} (n={len(subjects)})")

    X_list, labels_list = [], []
    ref_shape = None
    for s in splits:
        vhdr_path = s["vhdr_path"]
        events_path = s["events_path"]
        ses = s["ses"]
        if not os.path.isfile(vhdr_path):
            continue
        try:
            X, event_values = _load_raw_and_epochs(
                vhdr_path, events_path, tmin=0.0, tmax=0.5, n_scalp_channels=n_scalp_channels
            )
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Skipping run (load failed): {vhdr_path}\n  {type(e).__name__}: {e}")
            continue
        if ref_shape is None:
            ref_shape = (X.shape[1], X.shape[2])
        if (X.shape[1], X.shape[2]) != ref_shape:
            continue

        if label_mode == "event":
            labels = event_values
        else:
            # ses-01..ses-05 -> class 0..4
            if ses is None:
                continue
            labels = np.full((X.shape[0],), int(ses) - 1, dtype=np.int64)

        X_list.append(X)
        labels_list.append(labels)
    if not X_list:
        raise FileNotFoundError("Could not load any run from eeg_preprocessed (.vmrk/.eeg missing?).")
    print(f"Successfully loaded {len(X_list)} runs (skipped {len(splits) - len(X_list)}).")
    X_all = np.concatenate(X_list, axis=0).astype(np.float32)
    y_all = np.concatenate(labels_list, axis=0).astype(np.int64)

    # Global remap for event-mode to ensure 0..C-1 across all runs
    if label_mode == "event":
        uniq = np.unique(y_all)
        remap = {int(v): i for i, v in enumerate(uniq.tolist())}
        y_all = np.vectorize(lambda v: remap[int(v)])(y_all).astype(np.int64)

    return X_all, y_all


### Gaussian encoder used for UCR time-series benchmarks ###
class PopulationCoder:

    """Gaussian receptive field population encoder

    Attributes:
        tmax    (int): Exclusive upper bound on maximum spike time
        n       (int): Number of Gaussian curves to use for each time series sample
        data    (tensor): The data that is to be encoded
        centers (tensor): Tensor of the means of each Gaussian curve
        widths  (tensor): Tensor of the widths of each Gaussian curve
        beta    (int): Common scaling factor on the width of each Gaussian curve 
        scaler  (object): Used to scale incoming signals to zero mean, unit variance
    """
    
    def __init__(self, data, n, tmax, beta=1):
        self.tmax = tmax
        self.n = n
        self.beta = beta
        data_len = data.shape[1]

        # Save scaling parameters
        self.scaler = StandardScaler()
        data = torch.from_numpy(self.scaler.fit_transform(data))

        
        min_vals, _ = data.min(dim=0) 
        max_vals, _ = data.max(dim=0) 
        self.widths = beta*torch.abs(max_vals - min_vals) / (n - 2) 
        self.widths = self.widths.unsqueeze(-1).repeat(1, self.n)  # --> shape = (data_len, n)

        self.centers = torch.zeros(data_len, n)
        for i in range(data_len):

            idx_vec = torch.arange(n)
            self.centers[i] = (
                min_vals[i] + ((2 * idx_vec - 3)/ 2) * self.widths[i]
            )


    def encode(self, x):
        """Maps 1D time series to spike times (data_len, n). Accepts numpy or torch."""
        x = np.asarray(x, dtype=np.float64).ravel()
        x = self.scaler.transform(x.reshape(1, -1)).ravel()
        tensor = torch.from_numpy(x)

        tensor = tensor.unsqueeze(-1).repeat(1, self.n)
        exc = torch.exp(
            -0.5
            * ((tensor - self.centers) / self.widths)
            * ((tensor - self.centers) / self.widths)
        )
        spike_times = torch.floor((self.tmax + 1) * (1 - exc))
        spike_times[spike_times >= self.tmax] = float("Inf")

        return spike_times


def encode_ttfs_log_signal(signal, window=50):
    """
    Algorithm 5: Logarithmic TTFS encoding.
    Input: signal (1D), values in [0, 1] (or will be clipped).
    Result: spike_train (1D) of length window * len(signal); values 0, 1 (up), or -1 (down).
    - diff = 2 * (v_signal - 0.5) in [-1, 1].
    - If diff < 0: n = int(-20*log10(-diff)), spike_train[50*m + n] = -1.
    - Else: n = int(-20*log10(diff)), spike_train[50*m + n] = 1.
    n is clamped to [0, window-1] so the spike stays in the m-th window.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    # Normalize to [0, 1] if needed (algorithm assumes [0,1])
    smin, smax = signal.min(), signal.max()
    if smax > smin:
        signal = (signal - smin) / (smax - smin)
    else:
        signal = np.clip(signal, 0.0, 1.0)
    M = len(signal)
    spike_train = np.zeros(window * M, dtype=np.float64)
    for m in range(M):
        v_signal = np.clip(signal[m], 1e-12, 1.0 - 1e-12)
        diff = 2.0 * (v_signal - 0.5)
        if diff < -1e-12:
            n = -20.0 * math.log10(-diff)
            n = int(np.clip(n, 0, window - 1))
            spike_train[window * m + n] = -1.0
        elif diff > 1e-12:
            n = -20.0 * math.log10(diff)
            n = int(np.clip(n, 0, window - 1))
            spike_train[window * m + n] = 1.0
        else:
            # diff ≈ 0 (v_signal ≈ 0.5): put spike at end of window
            spike_train[window * m + (window - 1)] = 1.0
    return spike_train


def encode_binary_nbits_signal(signal, n_bits=10):
    """
    Algorithm 6: Binary encoding scheme (fractional bits).
    Input: signal (1D), values in [0, 1]. Result: spike_train (M, n_bits), 0 or 1.
    For each sample m: bit_decision = 0.5, then for each bit: if v_signal - bit_decision > 0
    set spike_train[m, bit]=1 and v_signal -= bit_decision; then bit_decision *= 0.5.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    smin, smax = signal.min(), signal.max()
    if smax > smin:
        signal = (signal - smin) / (smax - smin)
    else:
        signal = np.clip(signal, 0.0, 1.0)
    M = len(signal)
    spike_train = np.zeros((M, n_bits), dtype=np.float64)
    for m in range(M):
        v_signal = np.clip(signal[m], 0.0, 1.0)
        bit_decision = 0.5
        for bit in range(n_bits):
            if v_signal - bit_decision > 0:
                spike_train[m, bit] = 1.0
                v_signal = v_signal - bit_decision
            bit_decision = bit_decision * 0.5
    return spike_train


def encode_multithreshold_signal(signal, n_thresholds=5, thresholds=None):
    """
    Algorithm 7: The 5-Threshold encoding scheme using data modulation.

    Paper line-by-line:
      - Input: signal
      - Result: spike_train (we return pos_spikes, neg_spikes for Fig. 9 up/down)
      - Initialize empty spike_train     -> pos_spikes, neg_spikes zeros
      - For m from 0 to length(signal)-2  -> for m in range(M), M = len(diff)
      -   diff = signal[m+1] - signal[m]  -> diff = np.diff(signal); diff[m] = signal[m+1]-signal[m]
      -   For each threshold T_i         -> for i in range(K)
      -     If diff > T_i: spike_train[m,i] = 1  -> pos_spikes[m,i] = 1
      - (Extension) If diff < -T_i: neg_spikes[m,i] = 1  -> Fig. 9 blue "decrease"
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    # print("signal:", signal.shape)
    # Algorithm 7: "diff = signal[m+1] - signal[m]" for m = 0..length-2
    diff = np.diff(signal)  # diff[m] = signal[m+1] - signal[m], length M = len(signal)-1
    M = len(diff)
    # Thresholds T_i (paper does not define them; we use percentiles of |diff|)
    if thresholds is None:
        thresholds = np.percentile(np.abs(diff), np.linspace(20, 90, n_thresholds))
    thresholds = np.asarray(thresholds).ravel()
    K = len(thresholds)
    # Algorithm 7: "Initialize empty spike_train"
    pos_spikes = np.zeros((M, K), dtype=np.float64)
    neg_spikes = np.zeros((M, K), dtype=np.float64)
    # Algorithm 7: "For m from 0 to length(signal)-2" and "For each threshold T_i"
    for m in range(M):
        for i in range(K):
            # Algorithm 7: "If diff > T_i then spike_train[m, i] = 1"
            if diff[m] > thresholds[i]:
                pos_spikes[m, i] = 1.0
            # Extension (Fig. 9): spike when diff < -T_i (decrease)
            if diff[m] < -thresholds[i]:
                neg_spikes[m, i] = 1.0
    return pos_spikes, neg_spikes


# --------------- Visualization: comparison of 4 encodings ---------------

def plot_encoding_comparison(epoch, pop_spike_times, ttfs_window=50, n_bits=10, n_thresholds=5,
                             ch0=0, out_path="eeg_encoding_spikes.png"):
    """
    Five-panel comparison: raw EEG, Population (TTFS), Algorithm 5 (TTFS log),
    Algorithm 6 (Binary n_bits), Algorithm 7 (K-threshold). One channel only.
    """
    sig = np.asarray(epoch[ch0], dtype=np.float64)
    T = len(sig)
    t = np.arange(T)

    # Normalize to [0,1] for paper algorithms (Alg 5, 6, 7)
    smin, smax = sig.min(), sig.max()
    if smax > smin:
        sig_norm = (sig - smin) / (smax - smin)
    else:
        sig_norm = np.clip(sig, 0.0, 1.0)

    ttfs = encode_ttfs_log_signal(sig_norm, window=ttfs_window)
    binary = encode_binary_nbits_signal(sig_norm, n_bits=n_bits)
    multi_pos, multi_neg = encode_multithreshold_signal(sig_norm, n_thresholds=n_thresholds)
    M = multi_pos.shape[0]
    K = multi_pos.shape[1]

    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [1, 1.2, 1, 1.2, 1], "hspace": 0.35})
    for ax in axes:
        ax.set_xlim(0, T - 1)

    # 1) Raw EEG (one channel) — continuous line + blue square markers at sample points (paper style)
    ax0 = axes[0]
    sig_display = (sig - sig.mean()) / (sig.std() + 1e-8)
    ax0.plot(t, sig_display, color="#1f77b4", linewidth=1.0)
    step = max(1, T // 20)
    ax0.plot(t[::step], sig_display[::step], "s", color="#1f77b4", markersize=3, markeredgewidth=0.5)
    ax0.set_ylabel("EEG (a.u.)")
    ax0.set_title("Original signal (one channel)")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # 2) Population (TTFS) — raster, thin pulses
    ax1 = axes[1]
    if pop_spike_times is not None:
        raster = np.isfinite(pop_spike_times).astype(np.float32)
        n_neurons = raster.shape[1]
        spike_times_per_neuron = [np.where(raster[:, n])[0] for n in range(n_neurons)]
        ax1.eventplot(spike_times_per_neuron, lineoffsets=np.arange(n_neurons),
                      linelengths=0.85, linewidths=0.5, colors="C0")
        ax1.set_ylabel("Neuron index")
        ax1.set_ylim(-0.5, n_neurons - 0.5)
    ax1.set_title("Population (TTFS)")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # 3) Algorithm 5: TTFS logarithmic — thin red up / blue down pulses (Fig. 7 style)
    ax2 = axes[2]
    L = len(ttfs)
    x_ttfs = np.linspace(0, T - 1, L) if L > 1 else np.array([0.0])
    for i in range(L):
        if ttfs[i] == 1:
            ax2.vlines(x_ttfs[i], 0, 1, colors="red", linewidth=0.5)
        elif ttfs[i] == -1:
            ax2.vlines(x_ttfs[i], -1, 0, colors="blue", linewidth=0.5)
    ax2.axhline(0, color="gray", linewidth=0.3)
    ax2.set_ylabel("Logarithmic\nfunction")
    ax2.set_title("Algorithm 5: TTFS (log) — red ↑ increase, blue ↓ decrease")
    ax2.set_ylim(-1.1, 1.1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # 4) Algorithm 6: Binary n_bits — raster, one row per bit, thin pulses (Fig. 8 style)
    ax3 = axes[3]
    spike_times_per_bit = [np.where(binary[:, b])[0] for b in range(n_bits)]
    ax3.eventplot(spike_times_per_bit, lineoffsets=np.arange(n_bits), linelengths=0.85, linewidths=0.5, colors="red")
    ax3.set_ylabel("Bit")
    ax3.set_yticks(np.arange(n_bits))
    ax3.set_yticklabels([f"2^{{-{b+1}}}" for b in range(n_bits)], fontsize=7)
    ax3.set_title("Algorithm 6: Binary encoding ({} bits)".format(n_bits))
    ax3.set_ylim(-0.5, n_bits - 0.5)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # 5) Algorithm 7: K-threshold — five rows, red upward / blue downward pulses per row (Fig. 9 style)
    ax4 = axes[4]
    for i in range(K):
        for m in range(M):
            if multi_pos[m, i]:
                ax4.vlines(m, i, i + 0.45, colors="red", linewidth=0.5)
            if multi_neg[m, i]:
                ax4.vlines(m, i - 0.45, i, colors="blue", linewidth=0.5)
    ax4.axhline(0, color="gray", linewidth=0.2, alpha=0.5)
    for i in range(1, K):
        ax4.axhline(i, color="gray", linewidth=0.2, alpha=0.3)
    ax4.set_ylabel("Threshold")
    ax4.set_yticks(np.arange(K))
    ax4.set_yticklabels(["T{}".format(i + 1) for i in range(K)])
    ax4.set_xlabel("Time (samples)")
    ax4.set_title("Algorithm 7: {}-Threshold encoding (red ↑ increase, blue ↓ decrease)".format(K))
    ax4.set_ylim(-0.5, K - 0.5)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    plt.suptitle("EEG encoding comparison — Population + Paper algorithms (Alg. 5, 6, 7)", fontsize=11, y=1.02)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def _find_subject_session_run(task="ERP", subject=1, ses=2):
    """Return one discovered run dict for a given subject/session/task."""
    sub_tag = f"sub-{int(subject):02d}"
    ses_tag = f"ses-{int(ses):02d}"
    splits = _discover_eeg_splits(EEG_PREPROCESSED_DIR, task=task)
    for s in splits:
        p = s["vhdr_path"]
        if sub_tag in p and ses_tag in p:
            return s
    return None


def plot_alg7_across_sessions(
    subject=1,
    sessions=(2, 3, 4, 5),
    channel=0,
    trial=0,
    task="ERP",
    n_thresholds=5,
    n_scalp_channels=32,
    out_path="eeg_alg7_sub01_ses2-5.png",
):
    """
    Compare Algorithm 7 spikes across sessions for one subject and one EEG channel.
    Uses shared thresholds across sessions for fair comparison.
    """
    session_signals = []
    found_sessions = []
    for ses in sessions:
        run = _find_subject_session_run(task=task, subject=subject, ses=ses)
        if run is None:
            print(f"[skip] not found: sub-{subject:02d} ses-{ses:02d} task={task}")
            continue
        X, _ = _load_raw_and_epochs(
            run["vhdr_path"],
            run["events_path"],
            tmin=0.0,
            tmax=0.5,
            n_scalp_channels=n_scalp_channels,
        )
        if X.shape[0] == 0:
            print(f"[skip] empty epochs: sub-{subject:02d} ses-{ses:02d}")
            continue
        ch = int(np.clip(channel, 0, X.shape[1] - 1))
        tr = int(np.clip(trial, 0, X.shape[0] - 1))
        sig = np.asarray(X[tr, ch], dtype=np.float64)
        session_signals.append(sig)
        found_sessions.append(ses)
        print(f"Loaded sub-{subject:02d} ses-{ses:02d}: epochs={X.shape[0]}, ch={ch}, trial={tr}, T={len(sig)}")

    if not session_signals:
        raise FileNotFoundError("No valid sessions loaded for comparison.")

    # Shared normalization + shared thresholds for fair session comparison
    norm_signals = []
    all_abs_diff = []
    for sig in session_signals:
        smin, smax = sig.min(), sig.max()
        sig_norm = (sig - smin) / (smax - smin) if smax > smin else np.clip(sig, 0.0, 1.0)
        norm_signals.append(sig_norm)
        all_abs_diff.append(np.abs(np.diff(sig_norm)))
    all_abs_diff = np.concatenate(all_abs_diff, axis=0)
    thresholds = np.percentile(all_abs_diff, np.linspace(20, 90, n_thresholds))
    print("Shared thresholds:", np.round(thresholds, 6))

    n_rows = len(found_sessions)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.2 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.array([axes])

    for r, (ses, sig_norm) in enumerate(zip(found_sessions, norm_signals)):
        pos, neg = encode_multithreshold_signal(sig_norm, n_thresholds=n_thresholds, thresholds=thresholds)
        M, K = pos.shape

        # Left: normalized signal
        ax_l = axes[r, 0]
        t = np.arange(len(sig_norm))
        ax_l.plot(t, sig_norm, color="#1f77b4", linewidth=1.0)
        ax_l.set_title(f"sub-{subject:02d} ses-{ses:02d} | ch={channel} trial={trial} (normalized)")
        ax_l.set_ylabel("signal")
        ax_l.set_xlim(0, len(sig_norm) - 1)
        ax_l.spines["top"].set_visible(False)
        ax_l.spines["right"].set_visible(False)

        # Right: Algorithm 7 raster
        # Draw pos/neg on the SAME threshold line (same y), distinguished by color.
        # Small x-offset avoids complete overlap when both happen at same (m, i).
        ax_r = axes[r, 1]
        for i in range(K):
            for m in range(M):
                if pos[m, i]:
                    ax_r.vlines(m - 0.08, i - 0.35, i + 0.35, colors="red", linewidth=0.7)
                if neg[m, i]:
                    ax_r.vlines(m + 0.08, i - 0.35, i + 0.35, colors="blue", linewidth=0.7)
        ax_r.set_ylim(-0.5, K - 0.5)
        ax_r.set_yticks(np.arange(K))
        ax_r.set_yticklabels([f"T{i+1}" for i in range(K)])
        ax_r.set_xlim(0, M - 1)
        ax_r.set_title(
            f"Algorithm 7 spikes | pos_rate={pos.mean():.3f}, neg_rate={neg.mean():.3f}"
        )
        ax_r.spines["top"].set_visible(False)
        ax_r.spines["right"].set_visible(False)
        print(
            f"ses-{ses:02d} | pos_rate={pos.mean():.4f} | neg_rate={neg.mean():.4f} | total_rate={(pos.mean()+neg.mean()):.4f}"
        )

    axes[-1, 0].set_xlabel("Time (samples)")
    axes[-1, 1].set_xlabel("Time index (diff domain)")
    plt.suptitle(
        f"Algorithm 7 session comparison | sub-{subject:02d} | task={task} | channel={channel} | trial={trial}",
        fontsize=11,
        y=1.01,
    )
    # If out_path is empty/None, show inline in notebook.
    out_path = "" if out_path is None else str(out_path).strip()
    if out_path == "":
        print("No output path provided; showing plot inline.")
        plt.show()
        plt.close()
    else:
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved session comparison:", out_path)


def plot_classification_matching(y_true, y_pred, out_path=""):
    """
    Visualize test classification quality:
    1) confusion matrix
    2) sample-wise true vs predicted labels
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true/y_pred cannot be empty.")
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have same length.")

    labels = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = int(labels.max()) + 1
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    acc = float(np.mean(y_true == y_pred)) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    # Left: confusion matrix
    ax0 = axes[0]
    im = ax0.imshow(cm, cmap="Blues", aspect="auto")
    ax0.set_title(f"Confusion Matrix | Acc={acc:.2f}%")
    ax0.set_xlabel("Predicted")
    ax0.set_ylabel("True")
    ax0.set_xticks(np.arange(n_classes))
    ax0.set_yticks(np.arange(n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            ax0.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    # Right: true/pred timeline + mismatch marks
    ax1 = axes[1]
    x = np.arange(y_true.size)
    ax1.plot(x, y_true, color="#1f77b4", linewidth=1.2, label="True")
    ax1.plot(x, y_pred, color="#d62728", linewidth=1.0, alpha=0.85, label="Pred")
    mismatch = y_true != y_pred
    if np.any(mismatch):
        ax1.scatter(x[mismatch], y_true[mismatch], color="black", s=12, marker="x", label="Mismatch")
    ax1.set_title("Test Classification Matching")
    ax1.set_xlabel("Sample index")
    ax1.set_ylabel("Class")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(alpha=0.25)

    out_path = "" if out_path is None else str(out_path).strip()
    if out_path == "":
        print(f"Classification matching plot shown inline | acc={acc:.2f}%")
        plt.show()
        plt.close()
    else:
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved classification matching plot: {out_path} | acc={acc:.2f}%")


def _normalize_01(signal):
    signal = np.asarray(signal, dtype=np.float64).ravel()
    smin, smax = signal.min(), signal.max()
    if smax > smin:
        return (signal - smin) / (smax - smin)
    return np.clip(signal, 0.0, 1.0)


def _build_features_population(X, n_pop=8, tmax=15, coders=None):
    """
    Build dense features from population TTFS rasters.
    Feature per channel = flattened finite-mask raster from spike times.
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_ch, _ = X.shape
    if coders is None:
        coders = [PopulationCoder(X[:, ch, :], n=n_pop, tmax=tmax, beta=1) for ch in range(n_ch)]

    feats = []
    for i in range(n_samples):
        ch_feats = []
        for ch in range(n_ch):
            st = coders[ch].encode(X[i, ch])
            if isinstance(st, torch.Tensor):
                st = st.numpy()
            raster = np.isfinite(st).astype(np.float32)
            ch_feats.append(raster.reshape(-1))
        feats.append(np.concatenate(ch_feats, axis=0))
    return np.stack(feats, axis=0).astype(np.float32), coders


def _build_features_alg5(X, ttfs_window=50):
    """Build dense features from Algorithm 5 (TTFS log)."""
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_ch, _ = X.shape
    feats = []
    for i in range(n_samples):
        ch_feats = []
        for ch in range(n_ch):
            sig = _normalize_01(X[i, ch])
            enc = encode_ttfs_log_signal(sig, window=ttfs_window).astype(np.float32)
            ch_feats.append(enc.reshape(-1))
        feats.append(np.concatenate(ch_feats, axis=0))
    return np.stack(feats, axis=0).astype(np.float32)


def _build_features_alg6(X, n_bits=10):
    """Build dense features from Algorithm 6 (binary n-bits)."""
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_ch, _ = X.shape
    feats = []
    for i in range(n_samples):
        ch_feats = []
        for ch in range(n_ch):
            sig = _normalize_01(X[i, ch])
            enc = encode_binary_nbits_signal(sig, n_bits=n_bits).astype(np.float32)
            ch_feats.append(enc.reshape(-1))
        feats.append(np.concatenate(ch_feats, axis=0))
    return np.stack(feats, axis=0).astype(np.float32)


def _compute_alg7_thresholds_from_train(X_train, n_thresholds=5):
    X_train = np.asarray(X_train, dtype=np.float64)
    diffs = np.abs(np.diff(X_train, axis=-1)).reshape(-1)
    return np.percentile(diffs, np.linspace(20, 90, n_thresholds))


def _build_features_alg7(X, n_thresholds=5, thresholds=None):
    """
    Build dense features from Algorithm 7 (multi-threshold).
    Feature per channel = flatten([pos_spikes, neg_spikes]).
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_ch, _ = X.shape
    feats = []
    for i in range(n_samples):
        ch_feats = []
        for ch in range(n_ch):
            sig = _normalize_01(X[i, ch])
            pos, neg = encode_multithreshold_signal(sig, n_thresholds=n_thresholds, thresholds=thresholds)
            enc = np.concatenate([pos.reshape(-1), neg.reshape(-1)], axis=0).astype(np.float32)
            ch_feats.append(enc)
        feats.append(np.concatenate(ch_feats, axis=0))
    return np.stack(feats, axis=0).astype(np.float32)


def _evaluate_linear_separability(Xtr, ytr, Xte, yte):
    """
    Return:
      model_name, test_acc_percent, intra_mean_distance, inter_mean_distance, inter_over_intra_ratio
    """
    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xte = np.asarray(Xte, dtype=np.float32)
    ytr = np.asarray(ytr, dtype=np.int64).ravel()
    yte = np.asarray(yte, dtype=np.int64).ravel()

    mu = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    Xtr_z = (Xtr - mu) / std
    Xte_z = (Xte - mu) / std

    uniq_train = np.unique(ytr)
    if uniq_train.size < 2:
        raise ValueError(
            f"Need at least 2 classes for separability metrics, got classes={uniq_train.tolist()}."
        )

    model_name = None
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=300, solver="lbfgs")
        clf.fit(Xtr_z, ytr)
        yhat = clf.predict(Xte_z)
        model_name = "LogisticRegression(multinomial)"
    except Exception:
        from sklearn.linear_model import RidgeClassifier
        clf = RidgeClassifier()
        clf.fit(Xtr_z, ytr)
        yhat = clf.predict(Xte_z)
        model_name = "RidgeClassifier"

    lin_acc = float((yhat == yte).mean() * 100.0)

    classes = sorted(np.unique(ytr).tolist())
    centroids = []
    intra_vals = []
    for c in classes:
        Xc = Xtr_z[ytr == c]
        ctr = Xc.mean(axis=0)
        centroids.append(ctr)
        d = np.linalg.norm(Xc - ctr, axis=1)
        intra_vals.append(float(np.mean(d)))
    centroids = np.stack(centroids, axis=0)

    inter_d = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            inter_d.append(float(np.linalg.norm(centroids[i] - centroids[j])))

    intra_mean = float(np.mean(intra_vals))
    inter_mean = float(np.mean(inter_d))
    ratio = inter_mean / (intra_mean + 1e-12)
    return model_name, lin_acc, intra_mean, inter_mean, ratio


def check_all_encodings(
    X_train,
    y_train,
    X_test,
    y_test,
    n_pop=8,
    tmax=15,
    ttfs_window=50,
    n_bits=10,
    n_thresholds=5,
    max_train_eval=1200,
    max_test_eval=600,
):
    """
    Quantitatively compare all 4 encodings with the same linear separability diagnostics.
    """
    def _stratified_subsample(X, y, max_n, seed=42):
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.int64).ravel()
        max_n = min(int(max_n), len(y))
        if max_n <= 0:
            raise ValueError("max_n must be positive.")
        if max_n >= len(y):
            return X, y

        rng = np.random.default_rng(seed)
        classes, counts = np.unique(y, return_counts=True)
        expected = counts / counts.sum() * max_n
        picks = np.floor(expected).astype(int)
        picks = np.minimum(picks, counts)

        remainder = max_n - int(picks.sum())
        if remainder > 0:
            frac = expected - np.floor(expected)
            order = np.argsort(-frac)
            for oi in order:
                if remainder <= 0:
                    break
                c_count = counts[oi]
                if picks[oi] < c_count:
                    picks[oi] += 1
                    remainder -= 1

        idx_all = []
        for c, k in zip(classes.tolist(), picks.tolist()):
            if k <= 0:
                continue
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            idx_all.extend(idx[:k].tolist())
        idx_all = np.array(idx_all, dtype=np.int64)
        rng.shuffle(idx_all)
        return X[idx_all], y[idx_all]

    Xtr, ytr = _stratified_subsample(X_train, y_train, max_train_eval, seed=42)
    Xte, yte = _stratified_subsample(X_test, y_test, max_test_eval, seed=43)

    print(f"[check-all] subset train/test: {Xtr.shape} / {Xte.shape}")
    print(f"[check-all] classes(train): {sorted(np.unique(ytr).tolist())}")

    rows = []

    print("[check-all] Building Population features...")
    Xtr_pop, pop_coders = _build_features_population(Xtr, n_pop=n_pop, tmax=tmax, coders=None)
    Xte_pop, _ = _build_features_population(Xte, n_pop=n_pop, tmax=tmax, coders=pop_coders)
    model, acc, intra, inter, ratio = _evaluate_linear_separability(Xtr_pop, ytr, Xte_pop, yte)
    rows.append(dict(method="population_ttfs", model=model, acc=acc, intra=intra, inter=inter, ratio=ratio))

    print("[check-all] Building Algorithm 5 features...")
    Xtr_a5 = _build_features_alg5(Xtr, ttfs_window=ttfs_window)
    Xte_a5 = _build_features_alg5(Xte, ttfs_window=ttfs_window)
    model, acc, intra, inter, ratio = _evaluate_linear_separability(Xtr_a5, ytr, Xte_a5, yte)
    rows.append(dict(method="alg5_ttfs_log", model=model, acc=acc, intra=intra, inter=inter, ratio=ratio))

    print("[check-all] Building Algorithm 6 features...")
    Xtr_a6 = _build_features_alg6(Xtr, n_bits=n_bits)
    Xte_a6 = _build_features_alg6(Xte, n_bits=n_bits)
    model, acc, intra, inter, ratio = _evaluate_linear_separability(Xtr_a6, ytr, Xte_a6, yte)
    rows.append(dict(method="alg6_binary", model=model, acc=acc, intra=intra, inter=inter, ratio=ratio))

    print("[check-all] Building Algorithm 7 features...")
    thresholds = _compute_alg7_thresholds_from_train(Xtr, n_thresholds=n_thresholds)
    print("[check-all] Algorithm 7 shared thresholds:", np.round(thresholds, 8))
    Xtr_a7 = _build_features_alg7(Xtr, n_thresholds=n_thresholds, thresholds=thresholds)
    Xte_a7 = _build_features_alg7(Xte, n_thresholds=n_thresholds, thresholds=thresholds)
    model, acc, intra, inter, ratio = _evaluate_linear_separability(Xtr_a7, ytr, Xte_a7, yte)
    rows.append(dict(method="alg7_multithreshold", model=model, acc=acc, intra=intra, inter=inter, ratio=ratio))

    rows = sorted(rows, key=lambda d: (d["acc"], d["ratio"]), reverse=True)
    print("\n[All encoding check]")
    print("method                | linear_model                    | test_acc(%) | intra_mean | inter_mean | inter/intra")
    print("-" * 108)
    for r in rows:
        print(
            f"{r['method']:<21} | {r['model']:<31} | {r['acc']:>10.4f} | "
            f"{r['intra']:>10.6f} | {r['inter']:>10.6f} | {r['ratio']:>11.6f}"
        )

    best = rows[0]
    print(
        f"\nBest by test_acc then ratio: {best['method']} | "
        f"acc={best['acc']:.4f}% | ratio={best['ratio']:.6f}"
    )
    return rows


def _stratified_train_test_split(X, y, test_ratio=0.2, seed=42):
    """
    Simple stratified split without extra dependencies.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64).ravel()
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    train_idx, test_idx = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        if idx.size == 0:
            continue
        rng.shuffle(idx)
        n_test = max(1, int(round(test_ratio * idx.size)))
        n_test = min(n_test, max(1, idx.size - 1)) if idx.size > 1 else 1
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist() if idx.size > 1 else [])

    # Fallback when some class has only 1 sample and train set becomes empty
    if len(train_idx) == 0:
        all_idx = np.arange(len(y))
        rng.shuffle(all_idx)
        cut = max(1, int(round((1.0 - test_ratio) * len(all_idx))))
        train_idx = all_idx[:cut].tolist()
        test_idx = all_idx[cut:].tolist()

    train_idx = np.array(train_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def main():
    parser = argparse.ArgumentParser(description="EEG spike encodings: Population + Paper algorithms (Alg. 5, 6, 7)")
    parser.add_argument("--sample", type=int, default=0, help="Sample index to visualize")
    parser.add_argument("--n-pop", type=int, default=8, help="Population encoder neurons")
    parser.add_argument("--tmax", type=int, default=15, help="Max spike time (population)")
    parser.add_argument("--ttfs-window", type=int, default=50, help="Algorithm 5: window per sample")
    parser.add_argument("--n-bits", type=int, default=10, help="Algorithm 6: number of bits")
    parser.add_argument("--n-thresholds", type=int, default=5, help="Algorithm 7: number of thresholds")
    parser.add_argument("--out", type=str, default="eeg_encoding_spikes.png", help="Output figure path")
    parser.add_argument("--task", type=str, default="ERP", choices=("ERP", "SSVEP", "all"))
    parser.add_argument("--label-mode", type=str, default="event", choices=("event", "speed"),
                        help="Label source: event codes or session speed classes.")
    parser.add_argument("--n-scalp-channels", type=int, default=32, help="Use first N EEG channels")
    parser.add_argument("--compare-sessions", action="store_true",
                        help="Compare Algorithm 7 spikes across sessions for one subject/channel.")
    parser.add_argument("--subject", type=int, default=1, help="Subject index for --compare-sessions")
    # parser.add_argument("--sessions", type=str, default="2,3,4,5",
    #                     help="Comma-separated sessions for --compare-sessions, e.g. 2,3,4,5")
    parser.add_argument("--channel", type=int, default=0, help="EEG channel index for --compare-sessions")
    parser.add_argument("--trial", type=int, default=0, help="Epoch index for --compare-sessions")
    parser.add_argument("--check-all-encodings", action="store_true",
                        help="Run quantitative check for Population + Alg. 5/6/7 encodings.")
    parser.add_argument("--max-train-eval", type=int, default=1200,
                        help="Max train samples used by --check-all-encodings.")
    parser.add_argument("--max-test-eval", type=int, default=600,
                        help="Max test samples used by --check-all-encodings.")
    args = parser.parse_args()

    if args.compare_sessions:
        ses_list = [2, 3, 4, 5]
        # ses_list = [int(x.strip()) for x in args.sessions.split(",") if x.strip()]
        plot_alg7_across_sessions(
            subject=args.subject,
            sessions=ses_list,
            channel=args.channel,
            trial=args.trial,
            task=args.task,
            n_thresholds=args.n_thresholds,
            n_scalp_channels=args.n_scalp_channels,
            out_path=args.out,
        )
        print("Done. Open", args.out, "to compare sessions.")
        return

    X_train, y_train = _load_eeg(
        task=args.task,
        label_mode=args.label_mode,
        n_scalp_channels=args.n_scalp_channels,
    )
    print("Loaded EEG from", EEG_PREPROCESSED_DIR)
    X_test, y_test = X_train, y_train
    try:
        X_train, y_train, X_test, y_test = _stratified_train_test_split(X_train, y_train, test_ratio=0.2, seed=42)
        print("Using internal stratified 80/20 split for evaluation mode.")
    except Exception as e:
        print("Could not build train/test split; fallback to full data for visualization only. reason:", e)

    if args.check_all_encodings:
        uniq = np.unique(y_train)
        if uniq.size < 2 and args.label_mode == "event":
            print(
                "[check-all] Only one class found in event labels. "
                "Retrying with label_mode='speed' for meaningful comparison..."
            )
            X_train, y_train = _load_eeg(
                task=args.task,
                label_mode="speed",
                n_scalp_channels=args.n_scalp_channels,
            )
            X_train, y_train, X_test, y_test = _stratified_train_test_split(
                X_train, y_train, test_ratio=0.2, seed=42
            )
            uniq = np.unique(y_train)
            print(f"[check-all] classes(train) after fallback: {uniq.tolist()}")
        if uniq.size < 2:
            raise ValueError(
                "check-all-encodings requires at least 2 classes in train set. "
                "Try --label-mode speed or --task all."
            )
        check_all_encodings(
            X_train, y_train, X_test, y_test,
            n_pop=args.n_pop,
            tmax=args.tmax,
            ttfs_window=args.ttfs_window,
            n_bits=args.n_bits,
            n_thresholds=args.n_thresholds,
            max_train_eval=args.max_train_eval,
            max_test_eval=args.max_test_eval,
        )
        return

    sample_idx = min(args.sample, len(X_train) - 1)
    epoch = X_train[sample_idx]  # (C, T)

    # Population encoding (fit on first channel, encode channel 0 of this epoch)
    pop_coder = PopulationCoder(X_train[:, 0, :], n=args.n_pop, tmax=args.tmax, beta=1)
    pop_spike_times = pop_coder.encode(epoch[0])
    if isinstance(pop_spike_times, torch.Tensor):
        pop_spike_times = pop_spike_times.numpy()

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    plot_encoding_comparison(
        epoch, pop_spike_times,
        ttfs_window=args.ttfs_window, n_bits=args.n_bits, n_thresholds=args.n_thresholds,
        ch0=0, out_path=args.out
    )
    print("Done. Open", args.out, "to view the 4-encoding comparison.")


if __name__ == "__main__":
    main()
