"""Autocorrelation pitch helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def estimate_frequency_autocorrelation(
    frame: NDArray[np.floating],
    sr: int,
    *,
    fmin: float = 50.0,
    fmax: float = 1_200.0,
) -> tuple[float | None, float]:
    """Estimate frame frequency using normalized autocorrelation."""

    signal = np.asarray(frame, dtype=np.float64)
    signal = signal - float(np.mean(signal))
    if not np.any(signal):
        return None, 0.0

    min_lag = max(1, int(sr / fmax))
    max_lag = min(signal.size - 1, int(sr / fmin))
    if max_lag <= min_lag:
        return None, 0.0

    corr = np.correlate(signal, signal, mode="full")[signal.size - 1 :]
    if corr[0] <= 0:
        return None, 0.0
    corr = corr / corr[0]

    search = corr[min_lag : max_lag + 1]
    if search.size == 0:
        return None, 0.0
    lag = int(np.argmax(search)) + min_lag
    confidence = float(np.clip(corr[lag], 0.0, 1.0))
    if confidence < 0.15:
        return None, confidence

    refined_lag = _parabolic_refine(corr, lag)
    return float(sr / refined_lag), confidence


def _parabolic_refine(values: NDArray[np.float64], index: int) -> float:
    if index <= 0 or index >= values.size - 1:
        return float(index)
    left, center, right = values[index - 1], values[index], values[index + 1]
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return float(index)
    return float(index + 0.5 * (left - right) / denominator)
