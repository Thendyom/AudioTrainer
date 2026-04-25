"""Audio file I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from audiotrainer.audio.preprocessing import to_mono


def load_audio(path: str | Path, *, mono: bool = True) -> tuple[NDArray[np.float64], int]:
    """Load an audio file using soundfile and return ``(audio, sample_rate)``."""

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - dependency is installed in dev tests
        raise RuntimeError("soundfile is required to load audio files") from exc

    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    try:
        data, sr = sf.read(str(audio_path), always_2d=False)
    except RuntimeError as exc:
        raise ValueError(f"Unsupported or unreadable audio file: {audio_path}") from exc

    signal = np.asarray(data, dtype=np.float64)
    if mono:
        signal = to_mono(signal)
    return signal, int(sr)


def write_audio(path: str | Path, audio: NDArray[np.floating], sr: int) -> Path:
    """Write audio to disk using soundfile."""

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("soundfile is required to write audio files") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), np.asarray(audio, dtype=np.float64), sr)
    return output_path
