import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from app.fastapi_app import app
from audiotrainer.cli import app as cli_app


def wav_bytes(frequency: float = 440.0, sr: int = 8_000) -> bytes:
    time = np.arange(sr // 2, dtype=np.float64) / sr
    audio = 0.2 * np.sin(2 * np.pi * frequency * time)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    return buffer.getvalue()


def speech_bytes(sr: int = 8_000, offset: float = 0.0) -> bytes:
    first_time = np.arange(round(0.55 * sr), dtype=np.float64) / sr
    second_time = np.arange(round(0.65 * sr), dtype=np.float64) / sr
    audio = np.concatenate(
        [
            0.16 * np.sin(2 * np.pi * (175 + offset) * first_time),
            np.zeros(round(0.18 * sr)),
            0.18 * np.sin(2 * np.pi * (205 + offset) * second_time),
        ]
    )
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    return buffer.getvalue()


def test_capabilities_and_typed_pitch_api() -> None:
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    client = TestClient(app)
    capabilities = client.get("/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.json()["pitch_backends"] == ["yin"]
    assert capabilities.json()["speech_backends"] == ["baseline"]
    response = client.post("/pitch?target_note=A4", files={"file": ("tone.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 200
    assert response.json()["score"]["target_note"] == "A4"


def test_api_rejects_invalid_note_and_file_type() -> None:
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/pitch-exercise?targets=not-a-note", files={"file": ("tone.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 422
    response = client.post("/quality", files={"file": ("bad.txt", b"text", "text/plain")})
    assert response.status_code == 400


def test_api_transcription_contract_and_opt_in_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    monkeypatch.setenv("AUDIOTRAINER_DATA_DIR", str(tmp_path / "api-data"))
    client = TestClient(app)
    response = client.post(
        "/transcribe?time_signature=6/8&quantization=4&persist=true",
        files={"file": ("tone.wav", wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["score"]["beats_per_measure"] == 6
    assert body["metadata"]["actual_backend"] == "yin"
    from audiotrainer.history import SessionRepository

    sessions = SessionRepository(tmp_path / "api-data").list()
    assert len(sessions) == 1
    assert sessions[0].audio_path is None


def test_api_requires_persistence_before_audio_retention() -> None:
    from fastapi.testclient import TestClient

    response = TestClient(app).post(
        "/pitch?target_note=A4&retain_audio=true",
        files={"file": ("tone.wav", wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 422


def test_cli_score_exports_and_deterministic_capabilities(tmp_path: Path) -> None:
    input_path = tmp_path / "tone.wav"
    input_path.write_bytes(wav_bytes())
    runner = CliRunner()
    xml_path = tmp_path / "score.musicxml"
    json_path = tmp_path / "score.json"
    result = runner.invoke(cli_app, ["transcribe", str(input_path), "--musicxml-out", str(xml_path), "--json-out", str(json_path)])
    assert result.exit_code == 0, result.output
    assert xml_path.exists() and json_path.exists()
    result = runner.invoke(cli_app, ["capabilities"])
    assert result.exit_code == 0
    assert '"pitch_backends": [' in result.output
    assert '"yin"' in result.output


def test_cli_save_is_opt_in_and_can_retain_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "tone.wav"
    input_path.write_bytes(wav_bytes())
    data_dir = tmp_path / "data"
    monkeypatch.setenv("AUDIOTRAINER_DATA_DIR", str(data_dir))
    runner = CliRunner()

    result = runner.invoke(cli_app, ["pitch", str(input_path), "--target", "A4"])
    assert result.exit_code == 0, result.output
    from audiotrainer.history import SessionRepository

    assert SessionRepository(data_dir).list() == []
    result = runner.invoke(
        cli_app,
        ["pitch", str(input_path), "--target", "A4", "--save", "--retain-audio"],
    )
    assert result.exit_code == 0, result.output
    sessions = SessionRepository(data_dir).list()
    assert len(sessions) == 1
    assert sessions[0].audio_path and Path(sessions[0].audio_path).is_file()


def test_streamlit_app_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("AUDIOTRAINER_DATA_DIR", str(tmp_path / "app-data"))
    test = AppTest.from_file("app/streamlit_app.py", default_timeout=10).run()
    assert not test.exception
    assert test.title[0].value == "AudioTrainer"


def test_streamlit_all_pages_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("AUDIOTRAINER_DATA_DIR", str(tmp_path / "page-data"))
    test = AppTest.from_file("app/streamlit_app.py", default_timeout=10).run()
    expected = {
        "Dashboard": "Practice dashboard",
        "Pitch": "Pitch trainer",
        "Score": "Monophonic score creator",
        "Speech": "Speech coach",
        "Voice": "Voice profile",
        "Instruments": "Instrument lab",
        "Privacy": "Privacy & data",
    }
    for page, heading in expected.items():
        test.sidebar.radio[0].set_value(page).run()
        assert not test.exception
        assert any(item.value == heading for item in test.header)


@pytest.mark.parametrize(
    ("page", "uploader_index", "button_label", "metric_label"),
    [
        ("Pitch", 1, "Analyze recording", "Accuracy"),
        ("Score", 0, "Create score", None),
        ("Voice", 1, "Create voice profile", "Stable span"),
        ("Instruments", 0, "Analyze instrument", "Estimate"),
    ],
)
def test_streamlit_recorded_workflows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    page: str,
    uploader_index: int,
    button_label: str,
    metric_label: str | None,
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("AUDIOTRAINER_DATA_DIR", str(tmp_path / f"{page}-data"))
    test = AppTest.from_file("app/streamlit_app.py", default_timeout=20).run()
    test.sidebar.radio[0].set_value(page).run()
    test.file_uploader[uploader_index].upload("tone.wav", wav_bytes(sr=16_000), "audio/wav").run()
    next(button for button in test.button if button.label == button_label).click().run(timeout=20)
    assert not test.exception
    assert not test.error
    if metric_label:
        assert any(metric.label == metric_label for metric in test.metric)
    else:
        assert len(test.dataframe) >= 1


def test_streamlit_speech_reference_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("AUDIOTRAINER_DATA_DIR", str(tmp_path / "speech-data"))
    test = AppTest.from_file("app/streamlit_app.py", default_timeout=20).run()
    test.sidebar.radio[0].set_value("Speech").run()
    test.file_uploader[0].upload("user.wav", speech_bytes(8_000), "audio/wav").run()
    test.file_uploader[1].upload("reference.wav", speech_bytes(16_000, 2.0), "audio/wav").run()
    next(button for button in test.button if button.label == "Analyze speech").click().run(timeout=20)
    assert not test.exception
    assert not test.error
    assert any(metric.label == "Prosody similarity" for metric in test.metric)


test_streamlit_app_smoke = pytest.mark.ui(test_streamlit_app_smoke)
test_streamlit_all_pages_smoke = pytest.mark.ui(test_streamlit_all_pages_smoke)
test_streamlit_recorded_workflows = pytest.mark.ui(test_streamlit_recorded_workflows)
test_streamlit_speech_reference_workflow = pytest.mark.ui(test_streamlit_speech_reference_workflow)
