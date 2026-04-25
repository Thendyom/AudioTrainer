import numpy as np
import pytest

from audiotrainer.pitch import detect_pitch
from audiotrainer.pitch.autocorrelation import estimate_frequency_autocorrelation


def sine_wave(frequency: float, sr: int = 22_050, duration: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * duration), dtype=np.float64) / sr
    return 0.5 * np.sin(2.0 * np.pi * frequency * t)


def test_detect_pitch_tracks_synthetic_a4() -> None:
    sr = 22_050
    track = detect_pitch(sine_wave(440.0, sr), sr, frame_length=1024, hop_length=256)
    voiced = [frame.frequency_hz for frame in track.frames if frame.frequency_hz is not None]
    assert len(voiced) > 5
    assert float(np.median(voiced)) == pytest.approx(440.0, abs=4.0)
    assert {frame.note for frame in track.frames if frame.note} == {"A4"}


def test_detect_pitch_marks_silence_unvoiced() -> None:
    sr = 16_000
    track = detect_pitch(np.zeros(sr // 2), sr, frame_length=1024, hop_length=512)
    assert all(frame.frequency_hz is None for frame in track.frames)


def test_autocorrelation_estimator_tracks_sine() -> None:
    sr = 22_050
    frame = sine_wave(220.0, sr, duration=0.1)
    frequency, confidence = estimate_frequency_autocorrelation(frame, sr, fmin=80, fmax=500)
    assert frequency == pytest.approx(220.0, abs=3.0)
    assert confidence > 0.5
