"""Frame-based signal helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def frame_audio(
    audio: NDArray[np.floating],
    frame_length: int,
    hop_length: int,
    *,
    center: bool = True,
) -> NDArray[np.float64]:
    """Return overlapping frames with shape ``(num_frames, frame_length)``."""

    if frame_length <= 0:
        raise ValueError("frame_length must be positive")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")

    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("audio must be mono")
    if center:
        pad = frame_length // 2
        signal = np.pad(signal, (pad, pad), mode="constant")
    if signal.size < frame_length:
        signal = np.pad(signal, (0, frame_length - signal.size), mode="constant")

    frame_count = 1 + (signal.size - frame_length) // hop_length
    shape = (frame_count, frame_length)
    strides = (signal.strides[0] * hop_length, signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides).copy()


def frame_times(frame_count: int, sr: int, hop_length: int, *, centered: bool = True) -> NDArray[np.float64]:
    """Return frame timestamps in seconds."""

    offset = 0.0 if centered else 0.0
    return (np.arange(frame_count, dtype=np.float64) * hop_length / sr) + offset
