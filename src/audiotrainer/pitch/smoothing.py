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
    return PitchTrack(sample_rate=track.sample_rate, frames=frames)


def _median_filter(values: np.ndarray, kernel_size: int) -> np.ndarray:
    radius = kernel_size // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.array(
        [float(np.median(padded[index : index + kernel_size])) for index in range(values.size)],
        dtype=np.float64,
    )
