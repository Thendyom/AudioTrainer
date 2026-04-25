"""Small preprocessing functions for analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def to_mono(audio: NDArray[np.floating]) -> NDArray[np.float64]:
    """Convert an audio array to mono float64."""

    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim == 1:
        return signal
    if signal.ndim == 2:
        return np.mean(signal, axis=1)
    raise ValueError("audio must be a 1D mono or 2D channel array")


def remove_dc(audio: NDArray[np.floating]) -> NDArray[np.float64]:
    """Remove the DC offset from a signal."""

    signal = to_mono(audio)
    if signal.size == 0:
        return signal
    return signal - float(np.mean(signal))


def normalize_peak(audio: NDArray[np.floating], peak: float = 0.98) -> NDArray[np.float64]:
    """Peak-normalize audio without changing silent signals."""

    signal = np.asarray(audio, dtype=np.float64)
    max_abs = float(np.max(np.abs(signal))) if signal.size else 0.0
    if max_abs == 0.0:
        return signal
    return signal * (peak / max_abs)


def rms_energy(audio: NDArray[np.floating]) -> float:
    """Return root-mean-square energy."""

    signal = np.asarray(audio, dtype=np.float64)
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal))))
