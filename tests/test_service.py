import numpy as np
import pytest

from audiotrainer.api import service
from audiotrainer.api.schemas import PitchFrame, PitchTrack


def test_analyze_pitch_file_infers_target_note(monkeypatch: pytest.MonkeyPatch) -> None:
    track = PitchTrack(
        sample_rate=8_000,
        frames=[
            PitchFrame(time=0.0, frequency_hz=440.0, confidence=0.9, note="A4", cents=0.0),
            PitchFrame(time=0.1, frequency_hz=440.5, confidence=0.8, note="A4", cents=2.0),
            PitchFrame(time=0.2, frequency_hz=439.5, confidence=0.8, note="A4", cents=-2.0),
        ],
    )
    monkeypatch.setattr(service, "load_audio", lambda path: (np.zeros(8_000), 8_000))
    monkeypatch.setattr(service, "detect_pitch", lambda audio, sr: track)

    _, score, feedback = service.analyze_pitch_file("ignored.wav")

    assert score.target_note == "A4"
    assert score.accuracy > 0.95
    assert feedback


def test_compare_speech_files_resamples_mismatched_sample_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    user = 0.2 * np.sin(2 * np.pi * 180 * np.arange(8_000) / 8_000)
    reference = 0.2 * np.sin(2 * np.pi * 180 * np.arange(16_000) / 16_000)
    calls = iter([(user, 8_000), (reference, 16_000)])
    monkeypatch.setattr(service, "load_audio", lambda path: next(calls))

    report = service.compare_speech_files("user.wav", "reference.wav")
    assert report.overall_score > 0.8
