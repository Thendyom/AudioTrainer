"""Rule-based instrument classifier."""

from __future__ import annotations

from audiotrainer.api.schemas import InstrumentEstimate, InstrumentFeatureVector


def classify_instrument(features: InstrumentFeatureVector) -> InstrumentEstimate:
    """Return a lightweight instrument estimate."""

    voice_score = _score_voice(features)
    scores = {
        "voice": voice_score,
        "piano": _score_piano(features),
        "guitar": _score_guitar(features),
        "violin": _score_violin(features),
        "flute": _score_flute(features),
        "saxophone": _score_saxophone(features),
    }
    label, score = max(scores.items(), key=lambda item: item[1])
    if _looks_human_voice(features) and voice_score >= score - 0.18:
        label = "voice"
        score = voice_score
    if score < 0.32 or features.rms < 1e-5:
        return InstrumentEstimate(
            label="unknown",
            confidence=0.25 if features.rms >= 1e-5 else 0.0,
            explanation="No rule-based instrument profile matched strongly enough.",
        )
    margin = score - sorted(scores.values(), reverse=True)[1]
    confidence = max(0.35, min(0.85, score * (0.75 + margin)))
    return InstrumentEstimate(
        label=label,
        confidence=float(confidence),
        explanation="Rule-based estimate from centroid, bandwidth, zero-crossing rate, RMS, and harmonicity.",
    )


def _score_voice(features: InstrumentFeatureVector) -> float:
    centroid = _range_score(features.spectral_centroid, 120, 1_200)
    harmonic = min(1.0, features.harmonic_ratio / 0.65)
    zcr = 1.0 - min(1.0, features.zero_crossing_rate / 0.22)
    return 0.4 * centroid + 0.35 * harmonic + 0.25 * zcr


def _score_piano(features: InstrumentFeatureVector) -> float:
    bandwidth = _range_score(features.spectral_bandwidth, 900, 4_500)
    rolloff = _range_score(features.spectral_rolloff, 1_500, 8_000)
    harmonic = 1.0 - min(1.0, abs(features.harmonic_ratio - 0.35) / 0.55)
    return 0.35 * bandwidth + 0.35 * rolloff + 0.3 * harmonic


def _score_guitar(features: InstrumentFeatureVector) -> float:
    centroid = _range_score(features.spectral_centroid, 350, 2_500)
    bandwidth = _range_score(features.spectral_bandwidth, 700, 3_500)
    zcr = 1.0 - min(1.0, features.zero_crossing_rate / 0.35)
    return 0.35 * centroid + 0.35 * bandwidth + 0.3 * zcr


def _score_violin(features: InstrumentFeatureVector) -> float:
    centroid = _range_score(features.spectral_centroid, 900, 4_000)
    rolloff = _range_score(features.spectral_rolloff, 2_000, 9_000)
    harmonic = min(1.0, features.harmonic_ratio / 0.55)
    return 0.35 * centroid + 0.3 * rolloff + 0.35 * harmonic


def _score_flute(features: InstrumentFeatureVector) -> float:
    centroid = _range_score(features.spectral_centroid, 600, 3_500)
    bandwidth = 1.0 - min(1.0, features.spectral_bandwidth / 4_000)
    harmonic = min(1.0, features.harmonic_ratio / 0.75)
    return 0.4 * centroid + 0.3 * bandwidth + 0.3 * harmonic


def _score_saxophone(features: InstrumentFeatureVector) -> float:
    centroid = _range_score(features.spectral_centroid, 500, 2_800)
    bandwidth = _range_score(features.spectral_bandwidth, 1_000, 4_500)
    harmonic = 1.0 - min(1.0, abs(features.harmonic_ratio - 0.55) / 0.45)
    zcr = _range_score(features.zero_crossing_rate, 0.035, 0.2)
    return 0.3 * centroid + 0.35 * bandwidth + 0.2 * harmonic + 0.15 * zcr


def _range_score(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        midpoint = (low + high) / 2.0
        radius = max(1.0, (high - low) / 2.0)
        return 1.0 - min(1.0, abs(value - midpoint) / radius) * 0.35
    distance = low - value if value < low else value - high
    return max(0.0, 1.0 - distance / max(high - low, 1.0))


def _looks_human_voice(features: InstrumentFeatureVector) -> bool:
    return (
        80.0 <= features.spectral_centroid <= 1_500.0
        and features.harmonic_ratio >= 0.55
        and features.zero_crossing_rate <= 0.12
        and features.spectral_bandwidth <= 3_000.0
    )
