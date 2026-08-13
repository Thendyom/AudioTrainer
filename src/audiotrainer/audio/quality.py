"""Audio input quality analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from audiotrainer.api.schemas import AudioQualityReport, PitchTrack
from audiotrainer.audio.framing import frame_audio
from audiotrainer.audio.preprocessing import remove_dc, rms_energy, to_mono


def analyze_audio_quality(
    audio: NDArray[np.floating],
    sr: int,
    *,
    pitch_track: PitchTrack | None = None,
) -> AudioQualityReport:
    """Measure recording level, clipping, noise floor, and voiced coverage."""

    if sr <= 0:
        raise ValueError("sr must be positive")
    raw_signal = to_mono(audio)
    signal = remove_dc(raw_signal)
    duration = float(signal.size / sr)
    absolute = np.abs(raw_signal)
    peak = float(np.max(absolute)) if absolute.size else 0.0
    clipping_ratio = float(np.mean(absolute >= 0.99)) if absolute.size else 0.0
    frame_length = max(128, min(2048, int(0.03 * sr)))
    if signal.size:
        frames = frame_audio(signal, frame_length, max(64, frame_length // 2), center=False)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
        noise_floor = float(np.percentile(frame_rms, 10))
    else:
        noise_floor = 0.0
    if pitch_track and pitch_track.frames:
        voiced = sum(frame.frequency_hz is not None and frame.confidence >= 0.3 for frame in pitch_track.frames)
        voiced_coverage = voiced / len(pitch_track.frames)
    else:
        voiced_coverage = 0.0

    warnings: list[str] = []
    level = rms_energy(signal)
    if duration < 0.25:
        warnings.append("The recording is too short for reliable coaching.")
    if level < 0.005:
        warnings.append("The recording level is very low; move closer to the microphone.")
    if clipping_ratio > 0.005:
        warnings.append("Clipping was detected; reduce microphone gain or move farther away.")
    if level > 0 and noise_floor / level > 0.45:
        warnings.append("Background noise is high relative to the foreground audio.")
    if pitch_track is not None and voiced_coverage < 0.15:
        warnings.append("Too little stable pitch was detected for a confident result.")
    return AudioQualityReport(
        duration=duration,
        rms=level,
        peak=peak,
        clipping_ratio=clipping_ratio,
        estimated_noise_floor=noise_floor,
        voiced_coverage=voiced_coverage,
        warnings=warnings,
    )
