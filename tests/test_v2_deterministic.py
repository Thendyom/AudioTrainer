import numpy as np
import pytest

from audiotrainer.api import service
from audiotrainer.backends import capabilities


def test_capabilities_expose_only_built_in_engines() -> None:
    report = capabilities()
    assert report["pitch_backends"] == ["yin"]
    assert report["speech_backends"] == ["baseline"]
    assert report["instrument_backends"] == ["baseline"]
    assert "optional_dependencies" not in report


def test_instrument_analysis_is_quality_qualified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "load_audio", lambda path: (np.zeros(16_000), 16_000))
    result = service.analyze_instrument_file("ignored.wav")
    assert result.metadata.actual_backend == "baseline"
    assert result.estimate.label == "unknown"
    assert len(result.candidates) == 5
    assert result.experimental is True


@pytest.mark.parametrize(
    ("operation", "backend"),
    [
        (lambda: service.analyze_instrument_file("ignored.wav", backend="ast"), "instrument"),
        (lambda: service.coach_speech_file("ignored.wav", backend="faster-whisper"), "speech"),
        (lambda: service.create_score_file("ignored.wav", backend="pyin"), "pitch"),
    ],
)
def test_removed_model_backends_fail_before_reading_audio(operation, backend: str) -> None:
    with pytest.raises(ValueError, match=f"{backend} backend"):
        operation()
