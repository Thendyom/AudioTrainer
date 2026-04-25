import numpy as np

from audiotrainer.instruments import classify_instrument, extract_instrument_features


def test_extract_instrument_features_returns_expected_shape() -> None:
    sr = 16_000
    t = np.arange(sr, dtype=np.float64) / sr
    audio = 0.2 * np.sin(2 * np.pi * 440 * t)
    features = extract_instrument_features(audio, sr)
    assert features.rms > 0
    assert len(features.mfcc) == 13
    assert features.spectral_centroid > 0


def test_classifier_returns_supported_label() -> None:
    sr = 16_000
    t = np.arange(sr, dtype=np.float64) / sr
    audio = 0.2 * np.sin(2 * np.pi * 220 * t)
    estimate = classify_instrument(extract_instrument_features(audio, sr))
    assert estimate.label in {"voice", "piano", "guitar", "violin", "flute", "saxophone", "unknown"}
    assert 0.0 <= estimate.confidence <= 1.0
