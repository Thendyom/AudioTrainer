"""Pitch-track smoothing helpers."""

from __future__ import annotations

import numpy as np

from audiotrainer.api.schemas import PitchFrame, PitchTrack
from audiotrainer.pitch.notes import cents_error, hz_to_note


def median_smooth_pitch_track(track: PitchTrack, kernel_size: int = 5) -> PitchTrack:
    """Median-filter voiced frequencies while preserving unvoiced frames."""

    if kernel_size <= 1 or not track.frames:
        return track
    if kernel_size % 2 == 0:
        kernel_size += 1

    frequencies = np.array(
        [frame.frequency_hz if frame.frequency_hz is not None else np.nan for frame in track.frames],
        dtype=np.float64,
    )
    voiced = np.isfinite(frequencies)
    if int(np.sum(voiced)) < kernel_size:
        return track

    filled = frequencies.copy()
    median_value = float(np.nanmedian(filled[voiced]))
    filled[~voiced] = median_value
    smoothed = _median_filter(filled, kernel_size)
    frames: list[PitchFrame] = []
    for frame, frequency, is_voiced in zip(track.frames, smoothed, voiced, strict=True):
        if not is_voiced:
            frames.append(frame)
            continue
        note = hz_to_note(float(frequency))
        frames.append(
            PitchFrame(
                time=frame.time,
                frequency_hz=float(frequency),
                confidence=frame.confidence,
                note=note.name,
                cents=cents_error(float(frequency)),
            )
        )
    return stabilize_pitch_track(PitchTrack(sample_rate=track.sample_rate, frames=frames))


def stabilize_pitch_track(
    track: PitchTrack,
    *,
    confidence_on: float = 0.22,
    confidence_off: float = 0.12,
    max_octave_jump: float = 0.8,
) -> PitchTrack:
    """Apply voiced hysteresis and suppress isolated octave errors."""

    stabilized: list[PitchFrame] = []
    active = False
    last_frequency: float | None = None
    for index, frame in enumerate(track.frames):
        threshold = confidence_off if active else confidence_on
        if frame.frequency_hz is None or frame.confidence < threshold:
            active = False
            stabilized.append(frame.model_copy(update={"frequency_hz": None, "note": None, "cents": None}))
            continue
        frequency = frame.frequency_hz
        if last_frequency is not None:
            ratio = frequency / last_frequency
            if abs(ratio - 2.0) < max_octave_jump * 0.15:
                frequency /= 2.0
            elif abs(ratio - 0.5) < max_octave_jump * 0.075:
                frequency *= 2.0
            elif ratio > 1 + max_octave_jump and index + 1 < len(track.frames):
                following = track.frames[index + 1].frequency_hz
                if following and abs(following / last_frequency - 1.0) < 0.12:
                    frequency = last_frequency
        note = hz_to_note(float(frequency))
        stabilized.append(
            frame.model_copy(
                update={
                    "frequency_hz": float(frequency),
                    "note": note.name,
                    "cents": cents_error(float(frequency)),
                }
            )
        )
        active = True
        last_frequency = float(frequency)
    return PitchTrack(sample_rate=track.sample_rate, frames=stabilized)


def _median_filter(values: np.ndarray, kernel_size: int) -> np.ndarray:
    radius = kernel_size // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.array(
        [float(np.median(padded[index : index + kernel_size])) for index in range(values.size)],
        dtype=np.float64,
    )
