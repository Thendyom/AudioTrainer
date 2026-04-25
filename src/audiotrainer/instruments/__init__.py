"""Instrument feature extraction and lightweight classification."""

from audiotrainer.instruments.classifier import classify_instrument
from audiotrainer.instruments.features import extract_instrument_features

__all__ = ["classify_instrument", "extract_instrument_features"]
