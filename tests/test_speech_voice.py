import numpy as np

from audiotrainer.api.schemas import PitchFrame, PitchTrack
from audiotrainer.speech import (
    analyze_prosody,
    classify_voice_type,
    compare_reference_speech,
    detect_pause_patterns,
    estimate_vocal_range,
)


def test_detect_pause_patterns_finds_silence_between_tones() -> None:
    sr = 8_000
    tone = 0.2 * np.sin(2 * np.pi * 180 * np.arange(sr // 4) / sr)
    silence = np.zeros(sr // 4)
    audio = np.concatenate([tone, silence, tone])
    report = detect_pause_patterns(audio, sr)
    assert report.pause_count >= 1
    assert report.total_pause_time > 0.15


def test_analyze_prosody_reports_pitch_and_monotony() -> None:
    sr = 8_000
    t = np.arange(sr, dtype=np.float64) / sr
    audio = 0.2 * np.sin(2 * np.pi * 180 * t)
    report = analyze_prosody(audio, sr)
    assert report.duration == 1.0
    assert report.mean_pitch_hz is not None
    assert 0.0 <= report.monotony_score <= 1.0


def test_reference_speech_comparison_is_prosody_level() -> None:
    sr = 8_000
    t = np.arange(sr // 2, dtype=np.float64) / sr
    audio = 0.2 * np.sin(2 * np.pi * 180 * t)
    report = compare_reference_speech(audio, audio, sr)
    assert report.overall_score > 0.8
    assert "phoneme" in report.explanation.lower()


def test_voice_range_and_type_report_uncertainty() -> None:
    frames = []
    for index, frequency in enumerate(np.linspace(130.81, 261.63, 40)):
        frames.append(
            PitchFrame(
                time=index * 0.05,
                frequency_hz=float(frequency),
                confidence=0.8,
                note=None,
                cents=None,
            )
        )
    profile = estimate_vocal_range(PitchTrack(sample_rate=22_050, frames=frames))
    estimate = classify_voice_type(profile)
    assert profile.lowest_note is not None
    assert profile.highest_note is not None
    assert estimate.primary_type in {"Bass", "Baritone", "Tenor", "Alto", "Mezzo-soprano", "Soprano", "unknown"}
    assert "likely" in estimate.explanation.lower() or "insufficient" in estimate.explanation.lower()
