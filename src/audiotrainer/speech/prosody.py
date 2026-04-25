"""Speech prosody analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from audiotrainer.api.schemas import PauseReport, ProsodyReport
from audiotrainer.audio.framing import frame_audio
from audiotrainer.audio.preprocessing import remove_dc
from audiotrainer.pitch.notes import frequency_to_midi
from audiotrainer.pitch.yin import detect_pitch


def analyze_prosody(audio: NDArray[np.floating], sr: int) -> ProsodyReport:
    """Analyze pitch, intensity, pauses, and monotony for speech."""

    if sr <= 0:
        raise ValueError("sr must be positive")
    signal = remove_dc(audio)
    duration = float(signal.size / sr) if sr else 0.0
    if signal.size == 0:
        return ProsodyReport(
            duration=0.0,
            mean_pitch_hz=None,
            pitch_range_semitones=None,
            mean_intensity=0.0,
            pause_count=0,
            estimated_speech_rate=None,
            monotony_score=1.0,
        )

    pitch_track = detect_pitch(
        signal,
        sr,
        frame_length=min(2048, max(512, _next_power_of_two(int(0.046 * sr)))),
        hop_length=max(128, int(0.015 * sr)),
        fmin=70.0,
        fmax=500.0,
    )
    voiced = np.array([frame.frequency_hz for frame in pitch_track.frames if frame.frequency_hz is not None])
    mean_pitch = float(np.mean(voiced)) if voiced.size else None
    pitch_range = None
    if voiced.size >= 2:
        midi = np.array([frequency_to_midi(float(frequency)) for frequency in voiced], dtype=np.float64)
        pitch_range = float(np.percentile(midi, 90) - np.percentile(midi, 10))

    intensity = _frame_rms(signal, sr)
    mean_intensity = float(np.mean(intensity)) if intensity.size else 0.0
    pauses = detect_pause_patterns(signal, sr)
    speech_rate = _estimate_speech_rate(intensity, duration)
    report = ProsodyReport(
        duration=duration,
        mean_pitch_hz=mean_pitch,
        pitch_range_semitones=pitch_range,
        mean_intensity=mean_intensity,
        pause_count=pauses.pause_count,
        estimated_speech_rate=speech_rate,
        monotony_score=0.0,
    )
    return report.model_copy(update={"monotony_score": score_monotony(report)})


def detect_pause_patterns(audio: NDArray[np.floating], sr: int) -> PauseReport:
    """Detect silence-like pause regions in speech audio."""

    signal = remove_dc(audio)
    if signal.size == 0:
        return PauseReport(pause_count=0, total_pause_time=0.0, mean_pause_time=None, pauses=[])

    frame_length = max(128, int(0.03 * sr))
    hop_length = max(64, int(0.01 * sr))
    frames = frame_audio(signal, frame_length=frame_length, hop_length=hop_length, center=False)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    if not np.any(rms):
        duration = signal.size / sr
        return PauseReport(pause_count=1, total_pause_time=duration, mean_pause_time=duration, pauses=[(0.0, duration)])

    threshold = max(float(np.percentile(rms, 20)) * 1.5, float(np.max(rms)) * 0.04)
    silent = rms <= threshold
    min_pause = 0.18
    pauses: list[tuple[float, float]] = []
    start_index: int | None = None
    for index, is_silent in enumerate(silent):
        if is_silent and start_index is None:
            start_index = index
        if (not is_silent or index == silent.size - 1) and start_index is not None:
            end_index = index if not is_silent else index + 1
            start = start_index * hop_length / sr
            end = min(signal.size / sr, (end_index * hop_length + frame_length) / sr)
            if end - start >= min_pause:
                pauses.append((float(start), float(end)))
            start_index = None

    total = float(sum(end - start for start, end in pauses))
    mean = total / len(pauses) if pauses else None
    return PauseReport(pause_count=len(pauses), total_pause_time=total, mean_pause_time=mean, pauses=pauses)


def score_monotony(report: ProsodyReport) -> float:
    """Score monotonous delivery from 0 to 1."""

    if report.pitch_range_semitones is None:
        return 1.0
    pitch_component = 1.0 - np.clip(report.pitch_range_semitones / 8.0, 0.0, 1.0)
    pause_component = 0.2 if report.pause_count > max(1, report.duration / 3.0) else 0.0
    return float(np.clip(pitch_component + pause_component, 0.0, 1.0))


def _frame_rms(signal: NDArray[np.float64], sr: int) -> NDArray[np.float64]:
    frame_length = max(128, int(0.03 * sr))
    hop_length = max(64, int(0.01 * sr))
    frames = frame_audio(signal, frame_length=frame_length, hop_length=hop_length, center=False)
    return np.sqrt(np.mean(np.square(frames), axis=1))


def _estimate_speech_rate(intensity: NDArray[np.float64], duration: float) -> float | None:
    if duration <= 0 or intensity.size < 3:
        return None
    normalized = intensity / max(float(np.max(intensity)), 1e-12)
    derivative = np.diff(normalized)
    threshold = max(0.05, float(np.std(derivative)) * 1.2)
    onset_count = int(np.sum((derivative[1:] > threshold) & (derivative[:-1] <= threshold)))
    return float(onset_count / duration)


def _next_power_of_two(value: int) -> int:
    return 1 << max(1, value - 1).bit_length()
