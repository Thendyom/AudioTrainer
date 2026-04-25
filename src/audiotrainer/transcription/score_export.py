"""Simple score export helpers for note events."""

from __future__ import annotations

import html
from pathlib import Path

from audiotrainer.api.schemas import NoteEvent


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
    """Export note events as a small MusicXML score."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    divisions = 480
    notes_xml = "\n".join(_event_to_musicxml(event, bpm=bpm, divisions=divisions) for event in events)
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
    <measure number="1">
      <attributes>
        <divisions>{divisions}</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <direction placement="above">
        <direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>{bpm}</per-minute></metronome></direction-type>
        <sound tempo="{bpm}"/>
      </direction>
{notes_xml}
    </measure>
  </part>
</score-partwise>
"""
    output_path.write_text(score, encoding="utf-8")
    return output_path


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
