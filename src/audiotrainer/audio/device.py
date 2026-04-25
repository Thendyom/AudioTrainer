"""Microphone recording helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from audiotrainer.audio.preprocessing import to_mono


def record_audio(duration: float, sr: int = 44_100, channels: int = 1) -> NDArray[np.float64]:
    """Record audio from the default input device with sounddevice."""

    if duration <= 0:
        raise ValueError("duration must be positive")
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("sounddevice is required for microphone recording") from exc

    data = sd.rec(int(duration * sr), samplerate=sr, channels=channels, dtype="float64")
    sd.wait()
    return to_mono(np.asarray(data))
