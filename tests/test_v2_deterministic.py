import numpy as np
import pytest

from audiotrainer.api import service
from audiotrainer.backends import capabilities
from audiotrainer.ml.manager import BackendDisabledError


def test_capabilities_expose_supported_backends_but_keep_ai_off_by_default() -> None:
    report = capabilities()
    assert report["pitch_backends"] == ["yin"]
    assert report["speech_backends"] == ["baseline"]
    assert report["instrument_backends"] == ["baseline"]
    assert report["ai_supported"] is True
    assert report["ai_enabled"] is False
    assert report["supported_pitch_backends"] == ["yin", "pyin", "auto"]
    assert len(report["model_capabilities"]) == 4


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
def test_disabled_model_backends_fail_before_reading_audio(operation, backend: str) -> None:
    with pytest.raises(BackendDisabledError, match=f"{backend}.*disabled"):
        operation()
