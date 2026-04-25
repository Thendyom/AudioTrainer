"""Human-readable feedback generation."""

from audiotrainer.api.schemas import FeedbackItem, PitchScore, ProsodyReport, VocalRange, VoiceTypeEstimate


def generate_pitch_feedback(score: PitchScore) -> list[FeedbackItem]:
    """Generate feedback from a pitch score."""

    raise NotImplementedError


def generate_speech_feedback(report: ProsodyReport) -> list[FeedbackItem]:
    """Generate feedback from a prosody report."""

    raise NotImplementedError


def generate_voice_feedback(profile: VocalRange | VoiceTypeEstimate) -> list[FeedbackItem]:
    """Generate feedback from a voice range or voice type estimate."""

    raise NotImplementedError
