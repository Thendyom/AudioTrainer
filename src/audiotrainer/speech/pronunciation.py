"""Reference-based speech comparison without phoneme-level claims."""

from audiotrainer.api.schemas import PronunciationReport


def compare_reference_speech(user_audio, ref_audio, sr: int) -> PronunciationReport:
    """Compare a user recording against a reference recording."""

    raise NotImplementedError
