"""Reference-based speech comparison without phoneme-level claims."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from audiotrainer.api.schemas import PronunciationReport
from audiotrainer.speech.prosody import analyze_prosody


def compare_reference_speech(
    user_audio: NDArray[np.floating],
    ref_audio: NDArray[np.floating],
    sr: int,
) -> PronunciationReport:
    """Compare a user recording against a reference recording."""

    user = analyze_prosody(user_audio, sr)
    reference = analyze_prosody(ref_audio, sr)
    duration_ratio = _ratio_similarity(user.duration, reference.duration)
    pitch_similarity = _optional_similarity(user.mean_pitch_hz, reference.mean_pitch_hz, scale=160.0)
    energy_similarity = _ratio_similarity(user.mean_intensity, reference.mean_intensity)
    pause_similarity = _ratio_similarity(float(user.pause_count + 1), float(reference.pause_count + 1))

    scores = [score for score in [duration_ratio, pitch_similarity, energy_similarity, pause_similarity] if score is not None]
    overall = float(np.mean(scores)) if scores else 0.0
    explanation = (
        "Prosody-level comparison only; this version does not perform phoneme alignment "
        "or word-level pronunciation scoring."
    )
    return PronunciationReport(
        duration_ratio=duration_ratio,
        pitch_similarity=pitch_similarity,
        energy_similarity=energy_similarity,
        pause_similarity=pause_similarity,
        overall_score=overall,
        explanation=explanation,
    )


def _ratio_similarity(left: float, right: float) -> float | None:
    if left <= 0 or right <= 0:
        return None
    ratio = min(left, right) / max(left, right)
    return float(np.clip(ratio, 0.0, 1.0))


def _optional_similarity(left: float | None, right: float | None, *, scale: float) -> float | None:
    if left is None or right is None:
        return None
    return float(np.clip(1.0 - abs(left - right) / scale, 0.0, 1.0))
