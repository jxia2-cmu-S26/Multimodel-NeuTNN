"""
IMU-to-spike encoders for temporal / spiking neural networks.
- DeltaModulationEncoder: signal change -> spikes when delta exceeds thresholds.
"""

import numpy as np
from typing import Optional
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Delta Modulation Encoder (multi-threshold)
# ---------------------------------------------------------------------------


class DeltaModulationEncoder:
    """
    Encode signal *changes* into spikes when delta exceeds thresholds.
    Multi-threshold: several positive and negative thresholds; each has a channel.
    Output: (n_thresholds * 2, n_times) for positive and negative spike channels.
    """

    def __init__(
        self,
        thresholds: Optional[list[float]] = None,
        n_thresholds: int = 5,
        threshold_base: float = 0.001,
        scale: float = 2.0,
    ):
        if thresholds is not None:
            self.thresholds = np.asarray(thresholds, dtype=np.float64)
        else:
            # Geometric-like: base, base*scale, base*scale^2, ...
            self.thresholds = np.array(
                [threshold_base * (scale ** i) for i in range(n_thresholds)],
                dtype=np.float64,
            )
        self.n_thresholds = len(self.thresholds)

    def encode_channel(self, signal: np.ndarray) -> np.ndarray:
        """
        Encode one channel: diff = signal[t] - signal[t-1]; emit +1 spike channel when
        diff >= threshold, -1 spike channel when diff <= -threshold.

        Returns
        -------
        spikes : (n_thresholds * 2, n_times) uint8. First n_thresholds rows = positive
                deltas; next n_thresholds = negative deltas. n_times = len(signal)-1.
        """
        signal = np.asarray(signal, dtype=np.float64)
        diff = np.diff(signal)  # (n_times - 1,)
        n_t = len(diff)
        out = np.zeros((self.n_thresholds * 2, n_t), dtype=np.uint8)
        for k, th in enumerate(self.thresholds):
            out[k, :] = (diff >= th).astype(np.uint8)           # positive
            out[self.n_thresholds + k, :] = (diff <= -th).astype(np.uint8)  # negative
        return out

    def encode(
        self,
        data: np.ndarray,
        channel_indices: Optional[list[int]] = None,
    ) -> np.ndarray:
        """
        Encode (n_channels, n_times). Stack raster per channel.

        Returns
        -------
        spikes : (n_channels * n_thresholds * 2, n_times - 1) uint8
        """
        if data.ndim == 1:
            return self.encode_channel(data)
        n_ch, n_times = data.shape
        if channel_indices is None:
            channel_indices = list(range(n_ch))
        out_list = []
        for i in channel_indices:
            out_list.append(self.encode_channel(data[i]))
        return np.concatenate(out_list, axis=0)


# ---------------------------------------------------------------------------
# Helpers for visualization
# ---------------------------------------------------------------------------


def raster_events(spikes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    From binary raster (n_units, n_times), return (times, unit_indices) of all spikes
    for use in plt.scatter(times, units) or similar.
    """
    n_units, n_times = spikes.shape
    times, units = np.where(spikes > 0)
    return times.astype(np.float64), units.astype(np.float64)
