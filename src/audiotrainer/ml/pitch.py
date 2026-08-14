"""Lazy pYIN adapter with deterministic YIN fallback."""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
from numpy.typing import NDArray

from audiotrainer.api.schemas import PitchFrame, PitchTrack
from audiotrainer.ml.manager import BackendUnavailableError, require_feature
from audiotrainer.pitch.notes import cents_error, hz_to_note
from audiotrainer.pitch.smoothing import median_smooth_pitch_track
from audiotrainer.pitch.yin import detect_pitch


def detect_pitch_backend(
    audio: NDArray[np.floating],
    sr: int,
    *,
    backend: str = "yin",
    ai_enabled: bool = False,
    fmin: float = 50.0,
    fmax: float = 1_200.0,
) -> tuple[PitchTrack, str, str | None]:
    """Run the requested pitch backend and disclose auto fallback."""

    requested = backend.lower().strip()
    if requested in {"baseline", "yin"}:
        return detect_pitch(audio, sr, fmin=fmin, fmax=fmax), "yin", None
    if requested not in {"auto", "pyin"}:
        raise ValueError("pitch backend must be yin, pyin, or auto")
    if not ai_enabled:
        if requested == "pyin":
            require_feature(False, "pitch", "pyin")
        return detect_pitch(audio, sr, fmin=fmin, fmax=fmax), "yin", "Local AI is disabled; used YIN."
    if find_spec("librosa") is None:
        if requested == "pyin":
            raise BackendUnavailableError('pYIN is unavailable; install with pip install ".[ml-pitch]"')
        return detect_pitch(audio, sr, fmin=fmin, fmax=fmax), "yin", "pYIN is not installed; used YIN."
    return _detect_pyin(audio, sr, fmin=fmin, fmax=fmax), "pyin", None


def _detect_pyin(
    audio: NDArray[np.floating],
    sr: int,
    *,
    fmin: float,
    fmax: float,
    frame_length: int = 2_048,
    hop_length: int = 256,
) -> PitchTrack:
    import librosa

    signal = np.asarray(audio, dtype=np.float64).reshape(-1)
    if signal.size < frame_length:
        signal = np.pad(signal, (0, frame_length - signal.size))
    frequencies, voiced_flag, probabilities = librosa.pyin(
        signal,
        sr=sr,
        fmin=fmin,
        fmax=min(fmax, sr / 2.1),
        frame_length=frame_length,
        hop_length=hop_length,
        center=False,
    )
    times = np.arange(len(frequencies), dtype=np.float64) * hop_length / sr
    frames: list[PitchFrame] = []
    for time, frequency, voiced, probability in zip(times, frequencies, voiced_flag, probabilities, strict=True):
        confidence = float(np.clip(0.0 if probability is None else probability, 0.0, 1.0))
        if not voiced or not np.isfinite(frequency):
            frames.append(PitchFrame(time=float(time), frequency_hz=None, confidence=confidence, note=None, cents=None))
            continue
        note = hz_to_note(float(frequency))
        frames.append(
            PitchFrame(
                time=float(time),
                frequency_hz=float(frequency),
                confidence=confidence,
                note=note.name,
                cents=cents_error(float(frequency)),
            )
        )
    return median_smooth_pitch_track(PitchTrack(sample_rate=sr, frames=frames))
