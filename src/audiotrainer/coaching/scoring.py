"""Numeric scoring functions for coaching modes."""

from audiotrainer.api.schemas import PitchScore, PitchTrack


def score_pitch_accuracy(track: PitchTrack, target_notes=None) -> PitchScore:
    """Score pitch accuracy against target notes or frame cents."""

    raise NotImplementedError


def score_pitch_stability(track: PitchTrack) -> float:
    """Score pitch stability from frame-to-frame variation."""

    raise NotImplementedError
