"""Instrument feature extraction."""

from audiotrainer.api.schemas import InstrumentFeatureVector


def extract_instrument_features(audio, sr: int) -> InstrumentFeatureVector:
    """Extract compact spectral and temporal instrument features."""

    raise NotImplementedError
