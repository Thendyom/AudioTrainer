"""AudioTrainer command line interface."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from audiotrainer.api.service import (
    analyze_audio_quality,
    analyze_instrument_file,
    analyze_pitch_file,
    analyze_voice_profile_file,
    coach_speech_file,
    create_score_file,
    run_pitch_exercise,
)
from audiotrainer.backends import capabilities
from audiotrainer.history import SessionRepository
from audiotrainer.transcription.midi_export import export_midi
from audiotrainer.transcription.note_events import export_notes_csv
from audiotrainer.transcription.score_export import export_score_musicxml
from audiotrainer.transcription.score_document import score_document_to_note_events

app = typer.Typer(help="AudioTrainer audio coaching tools.")
console = Console()
history_app = typer.Typer(help="Inspect and manage local practice history.")
app.add_typer(history_app, name="history")


@app.command()
def pitch(
    file: Path = typer.Argument(..., exists=True, readable=True, help="Audio file to analyze."),
    target: str | None = typer.Option(None, "--target", "-t", help="Optional target note such as A4."),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    targets: str | None = typer.Option(None, "--targets", help="Comma-separated ordered exercise notes."),
    save: bool = typer.Option(False, "--save", help="Save metrics to local practice history."),
    retain_audio: bool = typer.Option(False, "--retain-audio", help="Retain a managed copy (requires --save)."),
) -> None:
    """Detect pitch and score pitch accuracy."""

    if targets:
        result = run_pitch_exercise(file, [item.strip() for item in targets.split(",") if item.strip()])
        _save_if_requested(
            save=save,
            retain_audio=retain_audio,
            mode="pitch",
            file=file,
            duration=result.quality.duration,
            backend=result.metadata.actual_backend,
            settings={"targets": result.target_notes},
            result=result,
            feedback=result.feedback,
        )
        if json_output:
            _print_json({"exercise": result})
        else:
            _print_model("Exercise", {"targets": ", ".join(result.target_notes), "accuracy": f"{result.overall_accuracy:.1%}", "backend": result.metadata.actual_backend})
            _print_feedback(result.feedback)
        return
    track, score, feedback = analyze_pitch_file(file, target)
    quality = analyze_audio_quality(file) if save else None
    _save_if_requested(
        save=save,
        retain_audio=retain_audio,
        mode="pitch",
        file=file,
        duration=quality.duration if quality else 0.0,
        backend="yin",
        settings={"target": target},
        result={"track": track, "score": score, "quality": quality},
        feedback=feedback,
    )
    if json_output:
        _print_json({"track": track, "score": score, "feedback": feedback})
        return

    console.print(f"[bold]Pitch frames:[/bold] {len(track.frames)}")
    _print_model("Pitch score", score.model_dump())
    _print_feedback(feedback)


@app.command()
def transcribe(
    file: Path = typer.Argument(..., exists=True, readable=True, help="Audio file to transcribe."),
    csv_out: Path | None = typer.Option(None, "--csv-out", help="Optional CSV output path."),
    midi_out: Path | None = typer.Option(None, "--midi-out", help="Optional MIDI output path."),
    musicxml_out: Path | None = typer.Option(None, "--musicxml-out", help="Optional MusicXML output path."),
    json_out: Path | None = typer.Option(None, "--json-out", help="Optional score JSON output path."),
    bpm: int = typer.Option(120, "--bpm", min=1, max=400),
    time_signature: str = typer.Option("4/4", "--time-signature"),
    quantization: int = typer.Option(4, "--quantization", min=1, max=32),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    save: bool = typer.Option(False, "--save", help="Save metrics to local practice history."),
    retain_audio: bool = typer.Option(False, "--retain-audio", help="Retain a managed copy (requires --save)."),
) -> None:
    """Convert detected pitch into note events."""

    track, events, document, metadata = create_score_file(
        file,
        bpm=bpm,
        time_signature=time_signature,
        quantization=quantization,
    )
    _save_if_requested(
        save=save,
        retain_audio=retain_audio,
        mode="score",
        file=file,
        duration=_track_duration(track),
        backend=metadata.actual_backend,
        settings={"bpm": bpm, "time_signature": time_signature, "quantization": quantization},
        result={"events": events, "score": document, "metadata": metadata},
        feedback=[],
    )
    if csv_out:
        export_notes_csv(events, csv_out)
    if midi_out:
        export_midi(score_document_to_note_events(document), midi_out, bpm=bpm)
    if musicxml_out:
        export_score_musicxml(document, musicxml_out)
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    if json_output:
        _print_json({"events": events, "score": document})
        return

    table = Table(title="Detected notes")
    for column in ["start", "end", "note", "frequency", "confidence"]:
        table.add_column(column)
    for event in events:
        table.add_row(
            f"{event.start_time:.2f}",
            f"{event.end_time:.2f}",
            event.note,
            f"{event.frequency_hz:.1f}",
            f"{event.confidence:.2f}",
        )
    console.print(table)


@app.command()
def speech(
    file: Path = typer.Argument(..., exists=True, readable=True, help="Speech file to analyze."),
    reference: Path | None = typer.Option(None, "--reference", "-r", exists=True, readable=True),
    goal: str = typer.Option("balanced", "--goal", help="balanced, clear pronunciation, or presenter presence."),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    save: bool = typer.Option(False, "--save", help="Save metrics to local practice history."),
    retain_audio: bool = typer.Option(False, "--retain-audio", help="Retain a managed copy (requires --save)."),
) -> None:
    """Analyze speech prosody and optional reference similarity."""

    result = coach_speech_file(file, reference_path=reference, goal=goal)
    _save_if_requested(
        save=save,
        retain_audio=retain_audio,
        mode="speech",
        file=file,
        duration=result.quality.duration,
        backend=result.metadata.actual_backend,
        settings={"goal": goal, "reference": bool(reference)},
        result=result,
        feedback=result.feedback,
    )
    report, feedback, comparison = result.prosody, result.feedback, result.reference_comparison
    if json_output:
        _print_json({"coaching": result})
        return

    _print_model("Prosody", report.model_dump())
    if comparison:
        _print_model("Reference comparison", comparison.model_dump())
    _print_feedback(feedback)


@app.command("voice-profile")
def voice_profile(
    file: Path = typer.Argument(..., exists=True, readable=True, help="Voice recording to profile."),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    save: bool = typer.Option(False, "--save", help="Save metrics to local practice history."),
    retain_audio: bool = typer.Option(False, "--retain-audio", help="Retain a managed copy (requires --save)."),
) -> None:
    """Estimate vocal range and rough voice type."""

    vocal_range, estimate, feedback = analyze_voice_profile_file(file)
    quality = analyze_audio_quality(file) if save else None
    _save_if_requested(
        save=save,
        retain_audio=retain_audio,
        mode="voice",
        file=file,
        duration=quality.duration if quality else 0.0,
        backend="yin",
        settings={"workflow": "guided-range"},
        result={"range": vocal_range, "estimate": estimate, "quality": quality},
        feedback=feedback,
    )
    if json_output:
        _print_json({"range": vocal_range, "estimate": estimate, "feedback": feedback})
        return
    _print_model("Vocal range", vocal_range.model_dump())
    _print_model("Voice type estimate", estimate.model_dump())
    _print_feedback(feedback)


@app.command()
def instrument(
    file: Path = typer.Argument(..., exists=True, readable=True, help="Instrument clip to classify."),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    save: bool = typer.Option(False, "--save", help="Save metrics to local practice history."),
    retain_audio: bool = typer.Option(False, "--retain-audio", help="Retain a managed copy (requires --save)."),
) -> None:
    """Estimate likely instrument class."""

    result = analyze_instrument_file(file)
    _save_if_requested(
        save=save,
        retain_audio=retain_audio,
        mode="instrument",
        file=file,
        duration=result.quality.duration,
        backend=result.metadata.actual_backend,
        settings={},
        result=result,
        feedback=[],
    )
    estimate = result.estimate
    if json_output:
        _print_json({"analysis": result})
        return
    _print_model("Instrument estimate", estimate.model_dump())


@app.command("capabilities")
def capabilities_command() -> None:
    """Show the built-in offline capabilities."""

    _print_json(capabilities())


@history_app.command("list")
def history_list(mode: str | None = typer.Option(None, "--mode"), json_output: bool = typer.Option(False, "--json")) -> None:
    """List local practice sessions."""

    sessions = SessionRepository().list(mode=mode)
    if json_output:
        _print_json({"sessions": sessions})
        return
    table = Table(title="Practice history")
    for column in ["id", "created", "mode", "duration", "backend"]:
        table.add_column(column)
    for session in sessions:
        table.add_row(session.session_id, session.created_at.isoformat(timespec="seconds"), session.mode, f"{session.duration:.1f}s", session.backend)
    console.print(table)


@history_app.command("delete")
def history_delete(session_id: str = typer.Argument(...)) -> None:
    """Delete one practice session and its retained recording."""

    if not SessionRepository().delete(session_id):
        raise typer.BadParameter("session was not found")
    console.print("Deleted.")


@history_app.command("export")
def history_export(output: Path = typer.Argument(...)) -> None:
    """Export history as JSON or CSV based on the output suffix."""

    repository = SessionRepository()
    if output.suffix.lower() == ".csv":
        repository.export_csv(output)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(repository.export_json(), encoding="utf-8")
    console.print(f"Exported to {output}")


@app.command("app")
def run_app(
    backend: str = typer.Option("streamlit", "--backend", help="streamlit or fastapi."),
    host: str = typer.Option("127.0.0.1", "--host", help="Host for FastAPI."),
    port: int = typer.Option(8501, "--port", help="Port for the app."),
) -> None:
    """Launch the demo application."""

    root = Path(__file__).resolve().parents[2]
    if backend == "streamlit":
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(root / "app" / "streamlit_app.py"), "--server.port", str(port)],
            check=False,
        )
        return
    if backend == "fastapi":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.fastapi_app:app",
                "--host",
                host,
                "--port",
                str(port),
            ],
            cwd=root,
            check=False,
        )
        return
    raise typer.BadParameter("backend must be 'streamlit' or 'fastapi'")


def _print_model(title: str, data: dict[str, Any]) -> None:
    table = Table(title=title)
    table.add_column("field")
    table.add_column("value")
    for key, value in data.items():
        table.add_row(str(key), str(value))
    console.print(table)


def _print_feedback(feedback) -> None:
    table = Table(title="Feedback")
    table.add_column("severity")
    table.add_column("category")
    table.add_column("message")
    table.add_column("suggestion")
    for item in feedback:
        table.add_row(item.severity, item.category, item.message, item.suggestion)
    console.print(table)


def _print_json(data: dict[str, Any]) -> None:
    def encode(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        return value

    console.print_json(data=encode(data))


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _save_if_requested(
    *,
    save: bool,
    retain_audio: bool,
    mode: str,
    file: Path,
    duration: float,
    backend: str,
    settings: dict[str, Any],
    result: Any,
    feedback: list[Any],
) -> None:
    if retain_audio and not save:
        raise typer.BadParameter("--retain-audio requires --save")
    if not save:
        return
    session = SessionRepository().save(
        mode=mode,
        duration=duration,
        source=file.name,
        backend=backend,
        settings=_jsonable(settings),
        result=_jsonable(result),
        feedback=_jsonable(feedback),
        audio_source=file,
        retain_audio=retain_audio,
    )
    console.print(f"Saved practice session {session.session_id}")


def _track_duration(track: Any) -> float:
    if not track.frames:
        return 0.0
    if len(track.frames) == 1:
        return track.frames[0].time
    return track.frames[-1].time + max(0.0, track.frames[1].time - track.frames[0].time)


if __name__ == "__main__":
    app()
