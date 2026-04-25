"""Numeric scoring functions for coaching modes."""

from __future__ import annotations

import numpy as np

from audiotrainer.api.schemas import PitchScore, PitchTrack
from audiotrainer.pitch.notes import cents_error, frequency_to_midi


def score_pitch_accuracy(track: PitchTrack, target_notes: str | list[str] | None = None) -> PitchScore:
    """Score pitch accuracy against target notes or frame cents."""

    voiced = [frame for frame in track.frames if frame.frequency_hz is not None]
    if not voiced:
        return PitchScore(
            accuracy=0.0,
            stability=0.0,
            mean_abs_cents=None,
            voiced_frame_count=0,
            target_note=_target_label(target_notes),
        )

    errors = _frame_errors(voiced, target_notes)
    mean_abs_cents = float(np.mean(np.abs(errors))) if errors.size else None
    if mean_abs_cents is None:
        accuracy = 0.0
    else:
        accuracy = float(np.clip(1.0 - mean_abs_cents / 50.0, 0.0, 1.0))
    return PitchScore(
        accuracy=accuracy,
        stability=score_pitch_stability(track),
        mean_abs_cents=mean_abs_cents,
        voiced_frame_count=len(voiced),
        target_note=_target_label(target_notes),
    )


def infer_target_note(track: PitchTrack, *, min_confidence: float = 0.3) -> str | None:
    """Infer the closest practical target note from the strongest voiced frames."""

    weights: dict[str, float] = {}
    for frame in track.frames:
        if frame.note is None or frame.frequency_hz is None or frame.confidence < min_confidence:
            continue
        weights[frame.note] = weights.get(frame.note, 0.0) + frame.confidence
    if not weights:
        return None
    return max(weights.items(), key=lambda item: item[1])[0]


def score_pitch_stability(track: PitchTrack) -> float:
    """Score pitch stability from frame-to-frame variation."""

    midi = np.array(
        [frequency_to_midi(frame.frequency_hz) for frame in track.frames if frame.frequency_hz is not None],
        dtype=np.float64,
    )
    if midi.size < 3:
        return 0.0
    cents_std = float(np.std((midi - np.median(midi)) * 100.0))
    return float(np.clip(1.0 - cents_std / 45.0, 0.0, 1.0))


def _frame_errors(voiced, target_notes: str | list[str] | None) -> np.ndarray:
    if target_notes is None:
        values = [frame.cents for frame in voiced if frame.cents is not None]
        return np.array(values, dtype=np.float64)
    if isinstance(target_notes, str):
        return np.array([cents_error(frame.frequency_hz, target_notes) for frame in voiced], dtype=np.float64)
    if len(target_notes) == 0:
        return np.array([], dtype=np.float64)
    repeated_targets = [target_notes[min(index, len(target_notes) - 1)] for index in range(len(voiced))]
    return np.array(
        [cents_error(frame.frequency_hz, target) for frame, target in zip(voiced, repeated_targets, strict=True)],
        dtype=np.float64,
    )


def _target_label(target_notes: str | list[str] | None) -> str | None:
    if target_notes is None:
        return None
    if isinstance(target_notes, str):
        return target_notes
    return ",".join(target_notes)
