"""YIN pitch detection baseline."""

from audiotrainer.api.schemas import PitchTrack


def detect_pitch(audio, sr: int) -> PitchTrack:
    """Estimate a framewise pitch track."""

    raise NotImplementedError
