"""Convert analysis results into coaching feedback."""

from audiotrainer.coaching.feedback import (
    generate_pitch_feedback,
    generate_speech_feedback,
    generate_voice_feedback,
)
from audiotrainer.coaching.scoring import infer_target_note, score_pitch_accuracy, score_pitch_stability

__all__ = [
    "generate_pitch_feedback",
    "generate_speech_feedback",
    "generate_voice_feedback",
    "infer_target_note",
    "score_pitch_accuracy",
    "score_pitch_stability",
]
