from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from audiotrainer.api.schemas import NoteEvent, ScoreDocument, ScoreEvent
from audiotrainer.audio.quality import analyze_audio_quality
from audiotrainer.audio.live import RollingAudioBuffer
from audiotrainer.transcription.score_document import create_score_document, parse_time_signature, score_document_to_note_events
from audiotrainer.transcription.score_export import export_score_musicxml


def test_audio_quality_flags_low_and_clipped_recordings() -> None:
    low = analyze_audio_quality(np.zeros(8_000), 8_000)
    clipped = analyze_audio_quality(np.ones(8_000), 8_000)
    assert any("low" in warning for warning in low.warnings)
    assert clipped.clipping_ratio > 0.9
    assert any("Clipping" in warning for warning in clipped.warnings)


def test_rolling_buffer_resamples_and_returns_requested_window() -> None:
    buffer = RollingAudioBuffer(target_sr=8_000, max_seconds=1.0)
    buffer.append(np.ones(4_000), 4_000)
    latest = buffer.latest(0.35)
    assert latest.shape == (2_800,)
    assert np.allclose(latest, 1.0)


def test_score_document_preserves_rests_and_splits_tied_measure() -> None:
    events = [
        NoteEvent(start_time=0.5, end_time=2.5, frequency_hz=440.0, note="A4", confidence=0.9),
    ]
    document = create_score_document(events, bpm=120, time_signature="3/4", quantization=4)
    assert document.events[0].kind == "rest"
    note_fragments = [event for event in document.events if event.kind == "note"]
    assert len(note_fragments) == 2
    assert note_fragments[0].tie_start is True
    assert note_fragments[1].tie_stop is True
    merged = score_document_to_note_events(document)
    assert len(merged) == 1
    assert merged[0].start_time == pytest.approx(0.5)
    assert merged[0].end_time == pytest.approx(2.5)


def test_score_document_rejects_overlaps_and_bad_signatures() -> None:
    with pytest.raises(ValueError, match="overlap"):
        ScoreDocument(
            events=[
                ScoreEvent(kind="note", start_beat=0.0, duration_beats=2.0, note="C4"),
                ScoreEvent(kind="note", start_beat=1.0, duration_beats=1.0, note="D4"),
            ]
        )
    with pytest.raises(ValueError, match="signature"):
        parse_time_signature("4/3")
    with pytest.raises(ValueError, match="valid note"):
        ScoreEvent(kind="note", start_beat=0.0, duration_beats=1.0, note="not-a-note")


def test_multi_measure_musicxml_is_valid_and_contains_rests_and_ties(tmp_path: Path) -> None:
    event = NoteEvent(start_time=0.5, end_time=2.5, frequency_hz=440.0, note="A4", confidence=0.9)
    document = create_score_document([event], bpm=120, time_signature="3/4")
    output = export_score_musicxml(document, tmp_path / "score.musicxml")
    root = ET.parse(output).getroot()
    assert len(root.findall(".//measure")) == 2
    assert root.find(".//rest") is not None
    assert root.find('.//tie[@type="start"]') is not None


@pytest.mark.parametrize("signature", ["3/4", "6/8"])
def test_supported_ui_signatures_export_measure_attributes(tmp_path: Path, signature: str) -> None:
    event = NoteEvent(start_time=0.0, end_time=0.75, frequency_hz=261.63, note="C4", confidence=0.9)
    document = create_score_document([event], bpm=120, time_signature=signature)
    root = ET.parse(export_score_musicxml(document, tmp_path / f"score-{signature.replace('/', '-')}.musicxml")).getroot()
    assert root.findtext(".//time/beats") == signature.split("/")[0]
    assert root.findtext(".//time/beat-type") == signature.split("/")[1]
