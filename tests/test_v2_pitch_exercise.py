import numpy as np
import pytest

from audiotrainer.api import service
from audiotrainer.api.schemas import PitchFrame, PitchTrack


def test_pitch_sequence_preserves_missed_target_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    track = PitchTrack(
        sample_rate=8_000,
        frames=[
            PitchFrame(time=0.0, frequency_hz=261.63, confidence=0.9, note="C4", cents=0.0),
            PitchFrame(time=0.1, frequency_hz=261.63, confidence=0.9, note="C4", cents=0.0),
            PitchFrame(time=0.2, frequency_hz=None, confidence=0.0, note=None, cents=None),
            PitchFrame(time=0.3, frequency_hz=None, confidence=0.0, note=None, cents=None),
            PitchFrame(time=0.4, frequency_hz=329.63, confidence=0.9, note="E4", cents=0.0),
            PitchFrame(time=0.5, frequency_hz=329.63, confidence=0.9, note="E4", cents=0.0),
        ],
    )
    monkeypatch.setattr(service, "load_audio", lambda path: (np.zeros(4_800), 8_000))
    monkeypatch.setattr(service, "detect_pitch", lambda audio, sr: track)

    result = service.run_pitch_exercise("ignored.wav", ["C4", "D4", "E4"])

    assert [item.detected_note for item in result.notes] == ["C4", None, "E4"]
    assert result.notes[1].missed is True
    assert result.notes[1].accuracy == 0.0
    assert result.overall_accuracy < 1.0


def test_pitch_exercise_rejects_removed_model_backend() -> None:
    with pytest.raises(ValueError, match="not part of this release"):
        service.run_pitch_exercise("ignored.wav", ["A4"], backend="pyin")
