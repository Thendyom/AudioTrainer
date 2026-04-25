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
    analyze_pitch_file,
    analyze_speech_file,
    analyze_voice_profile_file,
    classify_instrument_file,
    compare_speech_files,
    transcribe_file,
)
from audiotrainer.transcription.midi_export import export_midi
from audiotrainer.transcription.note_events import export_notes_csv

app = typer.Typer(help="AudioTrainer audio coaching tools.")
console = Console()


@app.command()
def pitch(
    file: Path = typer.Argument(..., exists=True, readable=True, help="Audio file to analyze."),
    target: str | None = typer.Option(None, "--target", "-t", help="Optional target note such as A4."),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Detect pitch and score pitch accuracy."""

    track, score, feedback = analyze_pitch_file(file, target)
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
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Convert detected pitch into note events."""

    _, events = transcribe_file(file)
    if csv_out:
        export_notes_csv(events, csv_out)
    if midi_out:
        export_midi(events, midi_out)
    if json_output:
        _print_json({"events": events})
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
) -> None:
    """Analyze speech prosody and optional reference similarity."""

    report, feedback = analyze_speech_file(file, goal=goal)
    comparison = compare_speech_files(file, reference) if reference else None
    if json_output:
        _print_json({"prosody": report, "comparison": comparison, "feedback": feedback})
        return

    _print_model("Prosody", report.model_dump())
    if comparison:
        _print_model("Reference comparison", comparison.model_dump())
    _print_feedback(feedback)


@app.command("voice-profile")
def voice_profile(
    file: Path = typer.Argument(..., exists=True, readable=True, help="Voice recording to profile."),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Estimate vocal range and rough voice type."""

    vocal_range, estimate, feedback = analyze_voice_profile_file(file)
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
) -> None:
    """Estimate likely instrument class."""

    estimate = classify_instrument_file(file)
    if json_output:
        _print_json({"estimate": estimate})
        return
    _print_model("Instrument estimate", estimate.model_dump())


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
