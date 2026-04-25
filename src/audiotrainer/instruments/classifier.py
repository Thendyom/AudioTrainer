"""Rule-based instrument classifier."""

from audiotrainer.api.schemas import InstrumentEstimate, InstrumentFeatureVector


def classify_instrument(features: InstrumentFeatureVector) -> InstrumentEstimate:
    """Return a lightweight instrument estimate."""

    raise NotImplementedError
