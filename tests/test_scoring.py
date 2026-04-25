import pytest

from audiotrainer.api.schemas import PitchFrame, PitchScore, PitchTrack, ProsodyReport, VocalRange
from audiotrainer.coaching import (
    generate_pitch_feedback,
    generate_speech_feedback,
    generate_voice_feedback,
    score_pitch_accuracy,
    score_pitch_stability,
)


def make_pitch_track(offset_hz: float = 0.0) -> PitchTrack:
    frames = [
        PitchFrame(time=0.00, frequency_hz=440.0 + offset_hz, confidence=0.9, note="A4", cents=0.0),
        PitchFrame(time=0.05, frequency_hz=440.5 + offset_hz, confidence=0.9, note="A4", cents=2.0),
        PitchFrame(time=0.10, frequency_hz=439.5 + offset_hz, confidence=0.9, note="A4", cents=-2.0),
    ]
    return PitchTrack(sample_rate=22_050, frames=frames)


def test_score_pitch_accuracy_uses_target_note() -> None:
    score = score_pitch_accuracy(make_pitch_track(), "A4")
    assert score.accuracy > 0.95
    assert score.stability > 0.9
    assert score.voiced_frame_count == 3


def test_score_pitch_stability_penalizes_wide_variation() -> None:
    stable = score_pitch_stability(make_pitch_track())
    unstable = score_pitch_stability(
        PitchTrack(
            sample_rate=22_050,
            frames=[
                PitchFrame(time=0.0, frequency_hz=440.0, confidence=0.9, note="A4", cents=0.0),
                PitchFrame(time=0.1, frequency_hz=466.16, confidence=0.9, note="A#4", cents=0.0),
                PitchFrame(time=0.2, frequency_hz=415.30, confidence=0.9, note="G#4", cents=0.0),
            ],
        )
    )
    assert stable > unstable


def test_pitch_feedback_handles_unvoiced_score() -> None:
    items = generate_pitch_feedback(PitchScore(accuracy=0.0, stability=0.0, mean_abs_cents=None, voiced_frame_count=0))
    assert items[0].severity == "critical"


def test_speech_feedback_flags_monotony() -> None:
    report = ProsodyReport(
        duration=4.0,
        mean_pitch_hz=180.0,
        pitch_range_semitones=1.0,
        mean_intensity=0.1,
        pause_count=0,
        estimated_speech_rate=2.0,
        monotony_score=0.9,
    )
    assert generate_speech_feedback(report)[0].category == "pronunciation"


def test_voice_feedback_reports_low_confidence() -> None:
    items = generate_voice_feedback(
        VocalRange(lowest_note=None, highest_note=None, stable_range_semitones=0.0, confidence=0.1)
    )
    assert items[0].severity == "warning"


def test_pydantic_schema_validates_score_bounds() -> None:
    with pytest.raises(ValueError):
        PitchScore(accuracy=1.2, stability=0.0, mean_abs_cents=0.0, voiced_frame_count=1)
