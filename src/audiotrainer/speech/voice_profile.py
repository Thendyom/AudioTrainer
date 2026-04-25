"""Voice range and rough type estimation."""

from audiotrainer.api.schemas import PitchTrack, VocalRange, VoiceTypeEstimate


def estimate_vocal_range(track: PitchTrack) -> VocalRange:
    """Estimate stable vocal range from a pitch track."""

    raise NotImplementedError


def classify_voice_type(
    vocal_range: VocalRange,
    speaking_pitch: float | None = None,
) -> VoiceTypeEstimate:
    """Classify likely voice type with explicit uncertainty."""

    raise NotImplementedError
