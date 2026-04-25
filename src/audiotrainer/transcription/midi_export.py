"""Minimal Standard MIDI File export for note events."""

from __future__ import annotations

from pathlib import Path

from audiotrainer.api.schemas import NoteEvent
from audiotrainer.pitch.notes import note_name_to_midi


def export_midi(
    events: list[NoteEvent],
    path: str | Path,
    *,
    bpm: int = 120,
    ticks_per_beat: int = 480,
    velocity: int = 88,
) -> Path:
    """Export note events to a type-0 MIDI file without extra dependencies."""

    if bpm <= 0:
        raise ValueError("bpm must be positive")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tempo_microseconds = int(60_000_000 / bpm)

    track_data = bytearray()
    track_data.extend(_varlen(0))
    track_data.extend(b"\xff\x51\x03")
    track_data.extend(tempo_microseconds.to_bytes(3, "big"))

    midi_messages: list[tuple[int, bytes]] = []
    for event in events:
        start_tick = _seconds_to_ticks(event.start_time, bpm, ticks_per_beat)
        end_tick = _seconds_to_ticks(event.end_time, bpm, ticks_per_beat)
        note = note_name_to_midi(event.note)
        midi_messages.append((start_tick, bytes([0x90, note, velocity])))
        midi_messages.append((max(start_tick + 1, end_tick), bytes([0x80, note, 0])))

    previous_tick = 0
    for tick, message in sorted(midi_messages, key=lambda item: (item[0], item[1][0])):
        track_data.extend(_varlen(max(0, tick - previous_tick)))
        track_data.extend(message)
        previous_tick = tick

    track_data.extend(_varlen(0))
    track_data.extend(b"\xff\x2f\x00")

    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big")
    header += (1).to_bytes(2, "big") + ticks_per_beat.to_bytes(2, "big")
    track = b"MTrk" + len(track_data).to_bytes(4, "big") + bytes(track_data)
    output_path.write_bytes(header + track)
    return output_path


def _seconds_to_ticks(seconds: float, bpm: int, ticks_per_beat: int) -> int:
    return int(round(max(0.0, seconds) * (bpm / 60.0) * ticks_per_beat))


def _varlen(value: int) -> bytes:
    if value < 0:
        raise ValueError("variable-length quantity cannot be negative")
    buffer = value & 0x7F
    value >>= 7
    while value:
        buffer <<= 8
        buffer |= ((value & 0x7F) | 0x80)
        value >>= 7

    output = bytearray()
    while True:
        output.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(output)
