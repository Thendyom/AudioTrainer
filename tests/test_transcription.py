from pathlib import Path

import pytest

from audiotrainer.api.schemas import PitchFrame, PitchTrack
from audiotrainer.transcription import export_midi, export_notes_csv, pitch_track_to_notes
from audiotrainer.transcription.score_export import export_musicxml, note_events_to_score_text


def make_track() -> PitchTrack:
    frames = [
        PitchFrame(time=0.00, frequency_hz=440.0, confidence=0.9, note="A4", cents=0.0),
        PitchFrame(time=0.05, frequency_hz=441.0, confidence=0.9, note="A4", cents=4.0),
        PitchFrame(time=0.10, frequency_hz=442.0, confidence=0.8, note="A4", cents=8.0),
        PitchFrame(time=0.15, frequency_hz=None, confidence=0.0, note=None, cents=None),
        PitchFrame(time=0.20, frequency_hz=493.9, confidence=0.8, note="B4", cents=0.0),
        PitchFrame(time=0.25, frequency_hz=494.5, confidence=0.7, note="B4", cents=2.0),
    ]
    return PitchTrack(sample_rate=22_050, frames=frames)


def test_pitch_track_to_notes_segments_contiguous_notes() -> None:
    events = pitch_track_to_notes(make_track(), min_duration=0.05)
    assert [event.note for event in events] == ["A4", "B4"]
    assert events[0].start_time == 0.0
    assert events[0].end_time == pytest.approx(0.15)
    assert events[0].frequency_hz == 441.0


def test_export_notes_csv_writes_header_and_rows(tmp_path: Path) -> None:
    events = pitch_track_to_notes(make_track(), min_duration=0.05)
    output = export_notes_csv(events, tmp_path / "notes.csv")
    text = output.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "start_time,end_time,frequency_hz,note,confidence"
    assert "A4" in text


def test_export_midi_writes_standard_header(tmp_path: Path) -> None:
    events = pitch_track_to_notes(make_track(), min_duration=0.05)
    output = export_midi(events, tmp_path / "notes.mid")
    data = output.read_bytes()
    assert data.startswith(b"MThd")
    assert b"MTrk" in data


def test_score_text_and_musicxml_export(tmp_path: Path) -> None:
    events = pitch_track_to_notes(make_track(), min_duration=0.05)
    assert "A4" in note_events_to_score_text(events)
    output = export_musicxml(events, tmp_path / "score.musicxml")
    text = output.read_text(encoding="utf-8")
    assert "<score-partwise" in text
    assert "<step>A</step>" in text
