"""End-to-end offline release acceptance using generated audio only."""

from __future__ import annotations

import io
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from app.fastapi_app import app as fastapi_app
from audiotrainer.api.service import (
    analyze_audio_quality,
    analyze_instrument_file,
    analyze_pitch_file,
    analyze_voice_profile_details,
    coach_speech_file,
    create_score_file,
    run_pitch_exercise,
)
from audiotrainer.cli import app as cli_app
from audiotrainer.history import SessionRepository
from audiotrainer.transcription import (
    export_midi,
    export_notes_csv,
    export_score_musicxml,
    score_document_to_note_events,
)


def _tone(frequency: float, duration: float, sr: int, amplitude: float = 0.22) -> np.ndarray:
    time = np.arange(round(duration * sr), dtype=np.float64) / sr
    edge = max(1, round(0.02 * sr))
    envelope = np.ones_like(time)
    envelope[:edge] = np.linspace(0.0, 1.0, edge)
    envelope[-edge:] = np.linspace(1.0, 0.0, edge)
    return amplitude * np.sin(2 * np.pi * frequency * time) * envelope


def _sequence(frequencies: list[float], sr: int = 16_000) -> np.ndarray:
    return np.concatenate([_tone(frequency, 0.65, sr) for frequency in frequencies])


def _speech_like(sr: int, offset: float = 0.0) -> np.ndarray:
    return np.concatenate(
        [
            _tone(175 + offset, 0.55, sr, 0.16),
            np.zeros(round(0.18 * sr)),
            _tone(205 + offset, 0.65, sr, 0.18),
            np.zeros(round(0.22 * sr)),
            _tone(185 + offset, 0.55, sr, 0.15),
        ]
    )


def _write(path: Path, audio: np.ndarray, sr: int) -> Path:
    sf.write(path, audio, sr)
    return path


def _wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    output = io.BytesIO()
    sf.write(output, audio, sr, format="WAV")
    return output.getvalue()


def test_all_library_workflows_and_exports(tmp_path: Path) -> None:
    pitch_path = _write(tmp_path / "a4.wav", _tone(440.0, 1.4, 16_000), 16_000)
    sequence_path = _write(tmp_path / "sequence.wav", _sequence([261.63, 293.66, 329.63]), 16_000)
    speech_path = _write(tmp_path / "speech.wav", _speech_like(8_000), 8_000)
    reference_path = _write(tmp_path / "reference.wav", _speech_like(16_000, 2.0), 16_000)
    voice_path = _write(tmp_path / "voice.wav", _sequence([130.81, 196.00, 261.63, 392.00]), 16_000)

    quality = analyze_audio_quality(pitch_path)
    track, pitch_score, feedback = analyze_pitch_file(pitch_path, "A4")
    exercise = run_pitch_exercise(sequence_path, ["C4", "D4", "E4"])
    score_track, notes, score, metadata = create_score_file(
        sequence_path,
        bpm=120,
        time_signature="6/8",
        quantization=4,
    )
    speech = coach_speech_file(speech_path, reference_path=reference_path, goal="presenter presence")
    voice_track, vocal_range, voice_type, voice_feedback, voice_quality, voice_metadata = analyze_voice_profile_details(
        voice_path
    )
    instrument = analyze_instrument_file(pitch_path)

    assert quality.duration > 1.3 and quality.peak < 0.3
    assert track.frames and pitch_score.target_note == "A4" and pitch_score.accuracy > 0.7 and feedback
    assert exercise.metadata.actual_backend == "yin"
    assert [item.target_note for item in exercise.notes] == ["C4", "D4", "E4"]
    assert exercise.overall_accuracy > 0.45
    assert score_track.frames and notes and score.time_signature == "6/8" and score.events
    assert metadata.actual_backend == "yin"
    assert speech.metadata.actual_backend == "baseline" and speech.reference_comparison is not None
    assert speech.reference_comparison.overall_score > 0.5 and speech.feedback
    assert voice_track.frames and vocal_range.lowest_note and vocal_range.highest_note
    assert voice_type.primary_type and voice_feedback and voice_quality.voiced_coverage > 0.2
    assert voice_metadata.actual_backend == "yin"
    assert instrument.metadata.actual_backend == "baseline" and instrument.candidates

    exported_notes = score_document_to_note_events(score)
    csv_path = export_notes_csv(exported_notes, tmp_path / "score.csv")
    midi_path = export_midi(exported_notes, tmp_path / "score.mid", bpm=score.bpm)
    xml_path = export_score_musicxml(score, tmp_path / "score.musicxml")
    json_path = tmp_path / "score.json"
    json_path.write_text(score.model_dump_json(indent=2), encoding="utf-8")
    assert csv_path.read_text(encoding="utf-8").startswith("start_time,end_time")
    assert midi_path.read_bytes().startswith(b"MThd")
    assert ET.parse(xml_path).getroot().tag == "score-partwise"
    assert '"beats_per_measure": 6' in json_path.read_text(encoding="utf-8")


def test_all_upload_endpoints_and_history_lifecycle(tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    data_dir = tmp_path / "api-data"
    monkeypatch.setenv("AUDIOTRAINER_DATA_DIR", str(data_dir))
    client = TestClient(fastapi_app)
    tone = _wav_bytes(_tone(440.0, 1.2, 16_000), 16_000)
    sequence = _wav_bytes(_sequence([261.63, 293.66, 329.63]), 16_000)
    speech = _wav_bytes(_speech_like(8_000), 8_000)
    reference = _wav_bytes(_speech_like(16_000, 2.0), 16_000)

    responses = [
        client.post("/quality", files={"file": ("tone.wav", tone, "audio/wav")}),
        client.post("/pitch?target_note=A4", files={"file": ("tone.wav", tone, "audio/wav")}),
        client.post(
            "/pitch-exercise?targets=C4,D4,E4",
            files={"file": ("sequence.wav", sequence, "audio/wav")},
        ),
        client.post(
            "/transcribe?time_signature=3/4&persist=true",
            files={"file": ("sequence.wav", sequence, "audio/wav")},
        ),
        client.post(
            "/speech?goal=presenter%20presence",
            files={
                "file": ("speech.wav", speech, "audio/wav"),
                "reference": ("reference.wav", reference, "audio/wav"),
            },
        ),
        client.post("/voice-profile", files={"file": ("voice.wav", sequence, "audio/wav")}),
        client.post("/instrument", files={"file": ("tone.wav", tone, "audio/wav")}),
        client.post("/instrument-analysis", files={"file": ("tone.wav", tone, "audio/wav")}),
    ]
    assert [response.status_code for response in responses] == [200] * len(responses)
    assert responses[3].json()["score"]["beats_per_measure"] == 3
    assert responses[4].json()["reference_comparison"] is not None

    history = client.get("/history")
    assert history.status_code == 200 and len(history.json()) == 1
    session_id = history.json()[0]["session_id"]
    assert client.get(f"/history/{session_id}").status_code == 200
    assert client.delete(f"/history/{session_id}").status_code == 204
    assert client.get(f"/history/{session_id}").status_code == 404


def test_cli_workflows_exports_and_persistence(tmp_path: Path, monkeypatch) -> None:
    input_path = _write(tmp_path / "tone.wav", _tone(440.0, 1.2, 16_000), 16_000)
    speech_path = _write(tmp_path / "speech.wav", _speech_like(8_000), 8_000)
    reference_path = _write(tmp_path / "reference.wav", _speech_like(16_000, 2.0), 16_000)
    data_dir = tmp_path / "cli-data"
    monkeypatch.setenv("AUDIOTRAINER_DATA_DIR", str(data_dir))
    runner = CliRunner()
    paths = {
        "csv": tmp_path / "score.csv",
        "midi": tmp_path / "score.mid",
        "xml": tmp_path / "score.musicxml",
        "json": tmp_path / "score.json",
    }
    commands = [
        ["capabilities"],
        ["pitch", str(input_path), "--target", "A4", "--json", "--save"],
        ["pitch", str(input_path), "--targets", "A4", "--json"],
        [
            "transcribe",
            str(input_path),
            "--csv-out",
            str(paths["csv"]),
            "--midi-out",
            str(paths["midi"]),
            "--musicxml-out",
            str(paths["xml"]),
            "--json-out",
            str(paths["json"]),
        ],
        ["speech", str(speech_path), "--reference", str(reference_path), "--json"],
        ["voice-profile", str(input_path), "--json"],
        ["instrument", str(input_path), "--json"],
        ["history", "list", "--json"],
        ["history", "export", str(tmp_path / "history.json")],
        ["history", "export", str(tmp_path / "history.csv")],
    ]
    for command in commands:
        result = runner.invoke(cli_app, command)
        assert result.exit_code == 0, f"{' '.join(command)}\n{result.output}\n{result.exception}"
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    assert (tmp_path / "history.json").is_file() and (tmp_path / "history.csv").is_file()
    sessions = SessionRepository(data_dir).list()
    assert len(sessions) == 1 and sessions[0].mode == "pitch"
