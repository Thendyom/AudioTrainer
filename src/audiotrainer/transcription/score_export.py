"""Simple score export helpers for note events."""

from __future__ import annotations

import html
from pathlib import Path

from audiotrainer.api.schemas import NoteEvent, ScoreDocument, ScoreEvent
from audiotrainer.transcription.score_document import create_score_document


def note_events_to_score_text(events: list[NoteEvent], *, bpm: int = 120) -> str:
    """Render note events as a compact text score with rough durations."""

    if not events:
        return "No notes detected."
    tokens = []
    for event in events:
        tokens.append(f"{event.note}{_duration_symbol(event.end_time - event.start_time, bpm)}")
    bars = [" ".join(tokens[index : index + 4]) for index in range(0, len(tokens), 4)]
    return " | ".join(bars)


def export_musicxml(events: list[NoteEvent], path: str | Path, *, bpm: int = 120) -> Path:
    """Export note events as MusicXML through a quantized score document."""

    document = create_score_document(events, bpm=bpm)
    return export_score_musicxml(document, path)


def export_score_musicxml(document: ScoreDocument, path: str | Path) -> Path:
    """Export a score document with rests, measures, and ties."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    divisions = 480
    measures: dict[int, list[ScoreEvent]] = {}
    for event in document.events:
        measures.setdefault(event.measure, []).append(event)
    if not measures:
        measures[1] = []
    measure_xml = []
    for number, events in sorted(measures.items()):
        attributes = ""
        direction = ""
        if number == 1:
            attributes = f"""      <attributes>
        <divisions>{divisions}</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>{document.beats_per_measure}</beats><beat-type>{document.beat_unit}</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>"""
            direction = f"""      <direction placement="above">
        <direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>{document.bpm}</per-minute></metronome></direction-type>
        <sound tempo="{document.bpm}"/>
      </direction>"""
        notes_xml = "\n".join(_score_event_to_musicxml(event, divisions=divisions) for event in events)
        measure_xml.append(f"""    <measure number="{number}">
{attributes}
{direction}
{notes_xml}
    </measure>""")
    score = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1">
      <part-name>AudioTrainer Notes</part-name>
    </score-part>
  </part-list>
  <part id="P1">
{chr(10).join(measure_xml)}
  </part>
</score-partwise>
"""
    output_path.write_text(score, encoding="utf-8")
    return output_path


def _score_event_to_musicxml(event: ScoreEvent, *, divisions: int) -> str:
    duration = max(1, int(round(event.duration_beats * divisions)))
    note_type = _beat_duration_type(event.duration_beats)
    if event.kind == "rest":
        return f"""      <note>
        <rest/>
        <duration>{duration}</duration>
        <type>{note_type}</type>
      </note>"""
    step, alter, octave = _parse_note(event.note or "C4")
    alter_xml = "" if alter == 0 else f"\n          <alter>{alter}</alter>"
    ties = ""
    notations = []
    if event.tie_stop:
        ties += '\n        <tie type="stop"/>'
        notations.append('<tied type="stop"/>')
    if event.tie_start:
        ties += '\n        <tie type="start"/>'
        notations.append('<tied type="start"/>')
    notation_xml = f"\n        <notations>{''.join(notations)}</notations>" if notations else ""
    return f"""      <note>
        <pitch>
          <step>{html.escape(step)}</step>{alter_xml}
          <octave>{octave}</octave>
        </pitch>
        <duration>{duration}</duration>{ties}
        <type>{note_type}</type>{notation_xml}
      </note>"""


def _event_to_musicxml(event: NoteEvent, *, bpm: int, divisions: int) -> str:
    step, alter, octave = _parse_note(event.note)
    duration = max(1, int(round((event.end_time - event.start_time) * (bpm / 60.0) * divisions)))
    note_type = _duration_type(event.end_time - event.start_time, bpm)
    alter_xml = "" if alter == 0 else f"\n        <alter>{alter}</alter>"
    return f"""      <note>
        <pitch>
          <step>{html.escape(step)}</step>{alter_xml}
          <octave>{octave}</octave>
        </pitch>
        <duration>{duration}</duration>
        <type>{note_type}</type>
      </note>"""


def _parse_note(note: str) -> tuple[str, int, int]:
    step = note[0].upper()
    rest = note[1:]
    alter = 0
    if rest.startswith("#"):
        alter = 1
        rest = rest[1:]
    elif rest.startswith("b"):
        alter = -1
        rest = rest[1:]
    return step, alter, int(rest)


def _duration_symbol(seconds: float, bpm: int) -> str:
    duration_type = _duration_type(seconds, bpm)
    return {
        "whole": "w",
        "half": "h",
        "quarter": "q",
        "eighth": "e",
        "16th": "s",
    }[duration_type]


def _duration_type(seconds: float, bpm: int) -> str:
    quarter_notes = max(0.0, seconds * (bpm / 60.0))
    candidates = [(4.0, "whole"), (2.0, "half"), (1.0, "quarter"), (0.5, "eighth"), (0.25, "16th")]
    return min(candidates, key=lambda item: abs(item[0] - quarter_notes))[1]


def _beat_duration_type(beats: float) -> str:
    candidates = [(4.0, "whole"), (2.0, "half"), (1.0, "quarter"), (0.5, "eighth"), (0.25, "16th"), (0.125, "32nd")]
    return min(candidates, key=lambda item: abs(item[0] - beats))[1]
