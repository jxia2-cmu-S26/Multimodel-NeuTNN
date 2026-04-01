"""
Multichannel wrappers for spikify temporal encoders -> uint8 raster for NeuTNN.

Expects continuous data (n_channels, n_time). Uses spikify's (time, features) layout
internally (data.T). Signed encoders (-1,0,1) are split into positive / negative binary
rows (2 * n_channels units); binary-only encoders keep n_channels rows.

Install: pip install innuce-spikify  (import name: spikify)
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from spikify.encoding.temporal.contrast import (
        moving_window,
        step_forward,
        threshold_based_representation,
        zero_cross_step_forward,
    )
    from spikify.encoding.temporal.latency import burst_encoding
    from spikify.encoding.temporal.global_referenced import (
        phase_encoding,
        time_to_first_spike,
    )
    from spikify.encoding.temporal.deconvolution import (
        bens_spiker,
        hough_spiker,
        modified_hough_spiker,
    )

    SPIKIFY_AVAILABLE = True
except ImportError:
    SPIKIFY_AVAILABLE = False

TEMPORAL_METHODS = [
    "moving_window",
    "step_forward",
    "threshold_based_representation",
    "zero_cross_step_forward",
    "burst_encoding",
    "phase_encoding",
    "time_to_first_spike",
    "bens_spiker",
    "hough_spiker",
    "modified_hough_spiker",
]


def is_spikify_available() -> bool:
    return SPIKIFY_AVAILABLE


def _trinary_split(signed: np.ndarray) -> np.ndarray:
    """(n_ch, T) int8 with -1,0,1 -> (2*n_ch, T) uint8."""
    pos = (signed > 0).astype(np.uint8)
    neg = (signed < 0).astype(np.uint8)
    return np.concatenate([pos, neg], axis=0)


def _binary_tf_to_ct(raw: np.ndarray) -> np.ndarray:
    """spikify deconv returns (T, F) or 1D (T,); output (F, T) uint8."""
    raw = np.asarray(raw, dtype=np.int8)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    return (raw > 0).astype(np.uint8).T


def encode_temporal_multichannel(data: np.ndarray, method: str, **kwargs: Any) -> np.ndarray:
    """
    Parameters
    ----------
    data : (n_channels, n_time) float
    method : name in TEMPORAL_METHODS
    **kwargs : encoder-specific (see below)

    Returns
    -------
    spikes : (n_units, n_time) uint8

    Common kwargs
    -------------
    moving_window: window_length (int, default min(16, T//8), at least 2)
    step_forward: threshold (float)
    threshold_based_representation: factor (float)
    zero_cross_step_forward: threshold (float)
    burst_encoding: n_max, t_min, t_max (ints); length defaults to T per channel
    phase_encoding: num_bits (int); if T % num_bits != 0, time axis is truncated
    time_to_first_spike: interval (int); if T % interval != 0, time axis is truncated
    bens_spiker: window_length (int), threshold (float), window_type (str, optional)
    hough_spiker: window_length (int), window_type (str, optional)
    modified_hough_spiker: (see spikify docs; pass through kwargs)
    """
    if not SPIKIFY_AVAILABLE:
        raise ImportError("spikify is not installed. Try: pip install innuce-spikify")

    method = method.strip().lower()
    if method not in TEMPORAL_METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from {TEMPORAL_METHODS}")

    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (n_ch, T), got shape {data.shape}")
    n_ch, T = data.shape
    if T < 2:
        raise ValueError("Need at least 2 time samples")

    sig_tf = data.T  # (T, n_ch)

    if method == "moving_window":
        wl = int(kwargs.get("window_length", max(2, min(16, T // 8))))
        wl = max(2, min(wl, T - 1))
        raw = moving_window(sig_tf, wl)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        else:
            raw = raw.T
        return _trinary_split(raw.astype(np.int8))

    if method == "step_forward":
        th = float(kwargs.get("threshold", 0.1))
        raw = step_forward(sig_tf, th)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        else:
            raw = raw.T
        return _trinary_split(raw.astype(np.int8))

    if method == "threshold_based_representation":
        fac = float(kwargs.get("factor", 0.5))
        raw = threshold_based_representation(sig_tf, fac)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        else:
            raw = raw.T
        return _trinary_split(raw.astype(np.int8))

    if method == "zero_cross_step_forward":
        th = float(kwargs.get("threshold", 0.1))
        raw = zero_cross_step_forward(sig_tf, th)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        else:
            raw = raw.T
        return (raw > 0).astype(np.uint8)

    if method == "burst_encoding":
        n_max = int(kwargs.get("n_max", 4))
        t_min = int(kwargs.get("t_min", 2))
        t_max = int(kwargs.get("t_max", 6))
        length = int(kwargs.get("length", T))
        rows = []
        for c in range(n_ch):
            x = data[c].astype(np.float64)
            rows.append(burst_encoding(x, n_max, t_min, t_max, length))
        return np.stack(rows, axis=0).astype(np.uint8)

    if method == "phase_encoding":
        num_bits = int(kwargs.get("num_bits", 8))
        if num_bits < 1:
            raise ValueError("phase_encoding: num_bits must be >= 1")
        t_use = (T // num_bits) * num_bits
        if t_use < num_bits:
            raise ValueError(
                f"phase_encoding: need T>={num_bits} after alignment, got T={T}, num_bits={num_bits}"
            )
        if t_use != T:
            data = data[:, :t_use]
            T = t_use
        rows = []
        for c in range(n_ch):
            rows.append(phase_encoding(data[c].astype(np.float64), num_bits))
        arrs = np.stack(rows, axis=0)
        return (arrs > 0).astype(np.uint8)

    if method == "time_to_first_spike":
        interval = int(kwargs.get("interval", 8))
        if interval < 1:
            raise ValueError("time_to_first_spike: interval must be >= 1")
        t_use = (T // interval) * interval
        if t_use < interval:
            raise ValueError(
                f"time_to_first_spike: need T>={interval} after alignment, got T={T}, interval={interval}"
            )
        if t_use != T:
            data = data[:, :t_use]
            T = t_use
        rows = []
        for c in range(n_ch):
            raw = time_to_first_spike(data[c].astype(np.float64), interval)
            r = np.asarray(raw).reshape(-1)
            if r.size != T:
                raise RuntimeError(
                    f"time_to_first_spike length mismatch: got {r.size}, expected T={T}"
                )
            rows.append(r)
        return np.stack(rows, axis=0).astype(np.uint8)

    if method == "bens_spiker":
        wl = int(kwargs.get("window_length", max(3, min(16, T // 10))))
        wl = max(2, min(wl, T - 1))
        th = float(kwargs.get("threshold", 0.5))
        wt = kwargs.get("window_type", "boxcar")
        raw = bens_spiker(sig_tf, wl, th, window_type=wt)
        return _binary_tf_to_ct(raw)

    if method == "hough_spiker":
        wl = int(kwargs.get("window_length", max(3, min(16, T // 10))))
        wl = max(2, min(wl, T - 1))
        wt = kwargs.get("window_type", "boxcar")
        raw = hough_spiker(sig_tf, wl, window_type=wt)
        return _binary_tf_to_ct(raw)

    if method == "modified_hough_spiker":
        wl = int(kwargs.get("window_length", max(3, min(16, T // 10))))
        wl = max(2, min(wl, T - 1))
        th = float(kwargs.get("threshold", 0.5))
        wt = kwargs.get("window_type", "boxcar")
        raw = modified_hough_spiker(sig_tf, wl, th, window_type=wt)
        return _binary_tf_to_ct(raw)

    raise RuntimeError(f"Unhandled method {method}")


def n_units_for_method(method: str, n_channels: int) -> int:
    """Expected n_units for a given method (after encode_temporal_multichannel)."""
    method = method.strip().lower()
    if method in (
        "moving_window",
        "step_forward",
        "threshold_based_representation",
    ):
        return 2 * n_channels
    if method in (
        "zero_cross_step_forward",
        "burst_encoding",
        "phase_encoding",
        "time_to_first_spike",
        "bens_spiker",
        "hough_spiker",
        "modified_hough_spiker",
    ):
        return n_channels
    raise ValueError(method)
