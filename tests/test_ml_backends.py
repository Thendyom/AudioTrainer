from __future__ import annotations

import io
import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from audiotrainer.api import service
from audiotrainer.api.schemas import AudioQualityReport, PitchFrame, PitchTrack, ProsodyReport, TranscriptWord
from audiotrainer.history import SessionRepository
from audiotrainer.ml.alignment import align_transcripts
from audiotrainer.ml.generative import generate_local_coaching
from audiotrainer.ml.instruments import map_ast_labels
from audiotrainer.ml.manager import (
    AISettings,
    BackendDisabledError,
    get_ai_settings,
    get_model_capabilities,
    model_path,
    remove_model,
    save_ai_settings,
)
from audiotrainer.ml.speech import build_transcript_report


def _track() -> PitchTrack:
    return PitchTrack(
        sample_rate=8_000,
        frames=[PitchFrame(time=0.0, frequency_hz=440.0, confidence=0.9, note="A4", cents=0.0)],
    )


def _transcript(words: list[str], *, language: str = "en", duration: float = 2.0):
    timed = [
        TranscriptWord(word=word, start=index * 0.4, end=index * 0.4 + 0.25, confidence=0.9)
        for index, word in enumerate(words)
    ]
    return build_transcript_report(
        text=" ".join(words),
        language=language,
        language_probability=0.95,
        duration=duration,
        words=timed,
    )


def test_ai_settings_round_trip_and_global_disable(tmp_path: Path) -> None:
    repository = SessionRepository(tmp_path)
    defaults = get_ai_settings(repository)
    assert defaults.enabled is False
    changed = defaults.model_copy(update={"enabled": True, "speech_enabled": False})
    save_ai_settings(changed, repository)
    assert get_ai_settings(repository) == changed
    statuses = get_model_capabilities(changed, data_dir=tmp_path)
    speech = next(item for item in statuses if item.feature == "speech")
    assert speech.enabled is False


def test_model_removal_is_scoped_to_managed_feature(tmp_path: Path) -> None:
    speech = model_path("speech", tmp_path)
    other = tmp_path / "keep.txt"
    speech.mkdir(parents=True)
    (speech / "model.bin").write_bytes(b"weights")
    other.write_text("keep", encoding="utf-8")
    assert remove_model("speech", data_dir=tmp_path) is True
    assert not speech.exists() and other.read_text(encoding="utf-8") == "keep"


def test_pyin_auto_fallback_and_explicit_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "load_audio", lambda path: (np.zeros(8_000), 8_000))
    monkeypatch.setattr(service, "detect_pitch", lambda audio, sr, **kwargs: _track())
    result = service.run_pitch_exercise("ignored.wav", ["A4"], backend="auto", ai_enabled=False)
    assert result.metadata.actual_backend == "yin"
    assert "disabled" in (result.metadata.fallback_reason or "").lower()
    with pytest.raises(BackendDisabledError):
        service.run_pitch_exercise("ignored.wav", ["A4"], backend="pyin", ai_enabled=False)


def test_pyin_adapter_is_selected_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from audiotrainer.ml import pitch as pitch_backend

    monkeypatch.setattr(pitch_backend, "find_spec", lambda name: object())
    monkeypatch.setattr(pitch_backend, "_detect_pyin", lambda *args, **kwargs: _track())
    track, actual, fallback = pitch_backend.detect_pitch_backend(
        np.zeros(8_000), 8_000, backend="pyin", ai_enabled=True
    )
    assert track.frames and actual == "pyin" and fallback is None


def test_transcript_metrics_are_language_aware() -> None:
    english = _transcript(["Um", "we", "we", "start"], language="en", duration=2.0)
    assert english.word_count == 4 and english.words_per_minute == 120
    assert english.repetitions == ["we"] and english.filler_words == ["um"]
    german = _transcript(["also", "start"], language="de")
    assert german.filler_words is None


def test_faster_whisper_adapter_converts_mocked_word_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from audiotrainer.ml import speech as speech_backend

    weights = model_path("speech", tmp_path)
    weights.mkdir(parents=True)
    (weights / "model.bin").write_bytes(b"local")
    audio_path = tmp_path / "speech.wav"
    sf.write(audio_path, np.zeros(16_000), 16_000)

    class FakeModel:
        def transcribe(self, *args, **kwargs):
            words = [
                SimpleWord(" Hello", 0.1, 0.4, 0.92),
                SimpleWord(" world", 0.5, 0.9, 0.88),
            ]
            return iter([SimpleSegment(" Hello world", words)]), SimpleInfo("en", 0.97)

    class SimpleWord:
        def __init__(self, word, start, end, probability):
            self.word, self.start, self.end, self.probability = word, start, end, probability

    class SimpleSegment:
        def __init__(self, text, words):
            self.text, self.words = text, words

    class SimpleInfo:
        def __init__(self, language, probability):
            self.language, self.language_probability = language, probability

    monkeypatch.setattr(speech_backend, "find_spec", lambda name: object())
    monkeypatch.setattr(speech_backend, "_load_model", lambda path: FakeModel())
    report = speech_backend.transcribe_file_local(audio_path, ai_enabled=True, data_dir=tmp_path)
    assert report.text == "Hello world"
    assert [item.word for item in report.words] == ["Hello", "world"]
    assert report.confidence == pytest.approx(0.9)
    assert report.language == "en"


def test_ordered_word_alignment_reports_all_difference_types() -> None:
    reference = _transcript(["we", "practice", "today"])
    user = _transcript(["we", "train", "today", "again"], duration=2.5)
    result = align_transcripts(user, reference)
    assert result.matched_words == 2
    assert result.substituted_words == [("practice", "train")]
    assert result.added_words == ["again"]
    assert result.delivery_timing_difference == 0.5
    assert "not phoneme" in result.explanation


def test_mocked_asr_populates_reference_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = np.zeros(8_000)
    monkeypatch.setattr(service, "load_audio", lambda path: (audio, 8_000))
    monkeypatch.setattr(service, "detect_speech_pitch", lambda audio, sr: _track())
    monkeypatch.setattr(
        service,
        "analyze_prosody",
        lambda audio, sr, pitch_track=None: ProsodyReport(
            duration=1.0,
            mean_pitch_hz=180,
            pitch_range_semitones=4,
            mean_intensity=0.1,
            pause_count=0,
            estimated_speech_rate=2,
            monotony_score=0.2,
        ),
    )
    monkeypatch.setattr(
        service,
        "analyze_quality",
        lambda *args, **kwargs: AudioQualityReport(
            duration=1.0,
            rms=0.1,
            peak=0.2,
            clipping_ratio=0,
            estimated_noise_floor=0,
            voiced_coverage=0.8,
        ),
    )
    monkeypatch.setattr(service, "compare_speech_files", lambda *args: None)
    reports = iter([_transcript(["hello", "world"]), _transcript(["hello", "there"])])
    monkeypatch.setattr(service, "transcribe_file_local", lambda *args, **kwargs: next(reports))
    result = service.coach_speech_file(
        "user.wav",
        reference_path="reference.wav",
        backend="faster-whisper",
        ai_enabled=True,
    )
    assert result.metadata.actual_backend == "faster-whisper"
    assert result.transcript and result.reference_transcript and result.word_alignment
    assert result.word_alignment.substituted_words == [("there", "world")]


def test_ast_mapping_confidence_and_unknown_thresholds() -> None:
    estimate, candidates, margin = map_ast_labels([("Violin, fiddle", 0.82), ("Flute", 0.15)])
    assert estimate.label == "violin" and candidates[0].raw_label == "Violin, fiddle" and margin > 0.5
    unknown, _, _ = map_ast_labels([("Violin, fiddle", 0.30), ("Flute", 0.28)])
    assert unknown.label == "unknown"


def test_local_generative_coach_rejects_remote_and_parses_local(monkeypatch: pytest.MonkeyPatch) -> None:
    prosody = ProsodyReport(
        duration=2,
        mean_pitch_hz=180,
        pitch_range_semitones=3,
        mean_intensity=0.1,
        pause_count=1,
        estimated_speech_rate=2,
        monotony_score=0.5,
    )
    with pytest.raises(ValueError, match="localhost"):
        generate_local_coaching(
            prosody=prosody,
            transcript=None,
            alignment=None,
            goal="balanced",
            endpoint="https://example.com",
            model="model",
            ai_enabled=True,
        )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"message": {"content": "Try one focused pause."}}).encode()

    monkeypatch.setattr("audiotrainer.ml.generative.urlopen", lambda request, timeout: Response())
    message = generate_local_coaching(
        prosody=prosody,
        transcript=None,
        alignment=None,
        goal="balanced",
        endpoint="http://127.0.0.1:11434",
        model="local",
        ai_enabled=True,
    )
    assert message == "Try one focused pause."


def test_api_returns_503_for_disabled_explicit_backend() -> None:
    from fastapi.testclient import TestClient
    from app.fastapi_app import app

    output = io.BytesIO()
    sf.write(output, np.zeros(8_000), 8_000, format="WAV")
    response = TestClient(app).post(
        "/speech?backend=faster-whisper&ai=false",
        files={"file": ("speech.wav", output.getvalue(), "audio/wav")},
    )
    assert response.status_code == 503


def test_cli_and_api_can_disable_ai_persistently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient
    from app.fastapi_app import app
    from audiotrainer.cli import app as cli_app

    data_dir = tmp_path / "settings-surfaces"
    monkeypatch.setenv("AUDIOTRAINER_DATA_DIR", str(data_dir))
    runner = CliRunner()
    assert runner.invoke(cli_app, ["models", "enable"]).exit_code == 0
    assert get_ai_settings(SessionRepository(data_dir)).enabled is True
    assert runner.invoke(cli_app, ["models", "disable"]).exit_code == 0
    assert get_ai_settings(SessionRepository(data_dir)).enabled is False

    client = TestClient(app)
    settings = AISettings(enabled=True, generative_coaching_enabled=True, generative_model="local")
    response = client.put("/ai-settings", json=settings.model_dump(mode="json"))
    assert response.status_code == 200 and response.json()["enabled"] is True
    assert client.get("/ai-settings").json()["generative_coaching_enabled"] is True
