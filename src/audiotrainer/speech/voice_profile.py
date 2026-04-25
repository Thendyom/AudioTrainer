"""Voice range and rough type estimation."""

from __future__ import annotations

import numpy as np

from audiotrainer.api.schemas import PitchTrack, VocalRange, VoiceTypeEstimate
from audiotrainer.pitch.notes import frequency_to_midi, hz_to_note

VOICE_RANGES = {
    "Bass": (40, 64),
    "Baritone": (45, 69),
    "Tenor": (48, 72),
    "Alto": (53, 77),
    "Mezzo-soprano": (57, 81),
    "Soprano": (60, 84),
}


def estimate_vocal_range(track: PitchTrack, *, min_confidence: float = 0.45) -> VocalRange:
    """Estimate stable vocal range from a pitch track."""

    frequencies = np.array(
        [
            frame.frequency_hz
            for frame in track.frames
            if frame.frequency_hz is not None and frame.confidence >= min_confidence
        ],
        dtype=np.float64,
    )
    if frequencies.size < 3:
        return VocalRange(lowest_note=None, highest_note=None, stable_range_semitones=0.0, confidence=0.0)

    midi = np.array([frequency_to_midi(float(frequency)) for frequency in frequencies], dtype=np.float64)
    low_midi = float(np.percentile(midi, 10))
    high_midi = float(np.percentile(midi, 90))
    stable_range = max(0.0, high_midi - low_midi)
    sample_confidence = min(1.0, frequencies.size / 40.0)
    range_confidence = min(1.0, stable_range / 18.0)
    confidence = float(np.clip(0.65 * sample_confidence + 0.35 * range_confidence, 0.0, 1.0))
    return VocalRange(
        lowest_note=hz_to_note(float(np.percentile(frequencies, 10))).name,
        highest_note=hz_to_note(float(np.percentile(frequencies, 90))).name,
        stable_range_semitones=float(stable_range),
        confidence=confidence,
    )


def classify_voice_type(
    vocal_range: VocalRange,
    speaking_pitch: float | None = None,
) -> VoiceTypeEstimate:
    """Classify likely voice type with explicit uncertainty."""

    if vocal_range.lowest_note is None or vocal_range.highest_note is None or vocal_range.confidence < 0.25:
        return VoiceTypeEstimate(
            primary_type="unknown",
            secondary_type=None,
            confidence=0.0,
            explanation="Insufficient sample for confident classification.",
        )

    low_midi = hz_to_note_name_to_midi(vocal_range.lowest_note)
    high_midi = hz_to_note_name_to_midi(vocal_range.highest_note)
    scores: dict[str, float] = {}
    for voice_type, (range_low, range_high) in VOICE_RANGES.items():
        overlap = max(0, min(high_midi, range_high) - max(low_midi, range_low))
        span = max(1, high_midi - low_midi)
        coverage = overlap / span
        center = (low_midi + high_midi) / 2.0
        range_center = (range_low + range_high) / 2.0
        center_penalty = min(1.0, abs(center - range_center) / 18.0)
        scores[voice_type] = 0.75 * coverage + 0.25 * (1.0 - center_penalty)

    if speaking_pitch is not None and speaking_pitch > 0:
        speaking_midi = frequency_to_midi(speaking_pitch)
        for voice_type, (_, range_high) in VOICE_RANGES.items():
            if speaking_midi > range_high - 4:
                scores[voice_type] += 0.05

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary, primary_score = ranked[0]
    secondary, secondary_score = ranked[1]
    confidence = float(np.clip(primary_score * vocal_range.confidence, 0.0, 0.9))
    secondary_type = secondary if primary_score - secondary_score < 0.15 else None
    if confidence < 0.45:
        explanation = f"Likely {primary.lower()} range, but the sample is too limited for a confident classification."
    elif secondary_type:
        explanation = f"Likely {primary.lower()}/{secondary_type.lower()} overlap based on the stable range."
    else:
        explanation = f"Likely {primary.lower()}, with uncertainty from the available recording length and stability."

    return VoiceTypeEstimate(
        primary_type=primary,
        secondary_type=secondary_type,
        confidence=confidence,
        explanation=explanation,
    )


def hz_to_note_name_to_midi(note_name: str) -> int:
    from audiotrainer.pitch.notes import note_name_to_midi

    return note_name_to_midi(note_name)
