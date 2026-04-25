"""Convert pitch tracks into note events."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import median

from audiotrainer.api.schemas import NoteEvent, PitchFrame, PitchTrack


def pitch_track_to_notes(
    track: PitchTrack,
    *,
    min_duration: float = 0.08,
    min_confidence: float = 0.3,
    max_gap: float = 0.08,
) -> list[NoteEvent]:
    """Segment a pitch track into stable note events."""

    if not track.frames:
        return []
    frame_step = _estimate_frame_step(track.frames)
    groups: list[list[PitchFrame]] = []
    current: list[PitchFrame] = []
    last_time: float | None = None
    last_note: str | None = None

    for frame in track.frames:
        voiced = frame.frequency_hz is not None and frame.note is not None and frame.confidence >= min_confidence
        if not voiced:
            if current:
                groups.append(current)
                current = []
            last_time = None
            last_note = None
            continue

        gap = 0.0 if last_time is None else frame.time - last_time
        same_note = frame.note == last_note
        if current and (not same_note or gap > max_gap + frame_step):
            groups.append(current)
            current = []

        current.append(frame)
        last_time = frame.time
        last_note = frame.note

    if current:
        groups.append(current)

    events: list[NoteEvent] = []
    for group in groups:
        event = _group_to_event(group, frame_step)
        if event.end_time - event.start_time >= min_duration:
            events.append(event)
    return events


def export_notes_csv(events: list[NoteEvent], path: str | Path) -> Path:
    """Write note events to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["start_time", "end_time", "frequency_hz", "note", "confidence"],
        )
        writer.writeheader()
        for event in events:
            writer.writerow(event.model_dump())
    return output_path


def _estimate_frame_step(frames: list[PitchFrame]) -> float:
    if len(frames) < 2:
        return 0.0
    steps = [right.time - left.time for left, right in zip(frames, frames[1:]) if right.time > left.time]
    return float(median(steps)) if steps else 0.0


def _group_to_event(frames: list[PitchFrame], frame_step: float) -> NoteEvent:
    frequencies = [frame.frequency_hz for frame in frames if frame.frequency_hz is not None]
    confidences = [frame.confidence for frame in frames]
    notes = [frame.note for frame in frames if frame.note is not None]
    note = max(set(notes), key=notes.count)
    return NoteEvent(
        start_time=frames[0].time,
        end_time=frames[-1].time + frame_step,
        frequency_hz=float(median(frequencies)),
        note=note,
        confidence=float(sum(confidences) / len(confidences)),
    )
