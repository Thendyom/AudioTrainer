"""Musical note conversion helpers."""

from __future__ import annotations

import math
import re

from audiotrainer.api.schemas import Note

A4_MIDI = 69
A4_HZ = 440.0
NOTE_NAMES_SHARP = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_TO_SHARP = {
    "DB": "C#",
    "EB": "D#",
    "GB": "F#",
    "AB": "G#",
    "BB": "A#",
}
NOTE_PATTERN = re.compile(r"^([A-Ga-g])([#bB]?)(-?\d+)$")


def frequency_to_midi(frequency_hz: float) -> float:
    """Convert frequency in hertz to a fractional MIDI note number."""

    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    return A4_MIDI + 12.0 * math.log2(frequency_hz / A4_HZ)


def midi_to_frequency(midi_note: float) -> float:
    """Convert a MIDI note number to equal-tempered frequency in hertz."""

    return A4_HZ * (2.0 ** ((midi_note - A4_MIDI) / 12.0))


def midi_to_note_name(midi_note: int) -> str:
    """Convert an integer MIDI note to a note name such as A4."""

    octave = (midi_note // 12) - 1
    return f"{NOTE_NAMES_SHARP[midi_note % 12]}{octave}"


def note_name_to_midi(note_name: str) -> int:
    """Parse a note name such as C4, C#4, or Db4 into an integer MIDI note."""

    match = NOTE_PATTERN.match(note_name.strip())
    if not match:
        raise ValueError(f"Invalid note name: {note_name!r}")

    letter, accidental, octave_text = match.groups()
    canonical = letter.upper()
    if accidental:
        canonical = f"{canonical}{accidental.upper()}"
    canonical = FLAT_TO_SHARP.get(canonical, canonical)
    if canonical not in NOTE_NAMES_SHARP:
        raise ValueError(f"Invalid note name: {note_name!r}")

    return (int(octave_text) + 1) * 12 + NOTE_NAMES_SHARP.index(canonical)


def hz_to_note(frequency_hz: float) -> Note:
    """Convert a frequency to the nearest equal-tempered note."""

    fractional_midi = frequency_to_midi(frequency_hz)
    nearest_midi = int(round(fractional_midi))
    nearest_frequency = midi_to_frequency(nearest_midi)
    return Note(
        name=midi_to_note_name(nearest_midi),
        midi=nearest_midi,
        frequency_hz=nearest_frequency,
        cents=1200.0 * math.log2(frequency_hz / nearest_frequency),
    )


def _target_to_frequency(target_note: str | int | float | None, frequency_hz: float) -> float:
    if target_note is None:
        return midi_to_frequency(int(round(frequency_to_midi(frequency_hz))))
    if isinstance(target_note, str):
        return midi_to_frequency(note_name_to_midi(target_note))
    return midi_to_frequency(float(target_note))


def cents_error(frequency_hz: float, target_note: str | int | float | None = None) -> float:
    """Return cents deviation from the nearest note or a target note."""

    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    target_frequency = _target_to_frequency(target_note, frequency_hz)
    return 1200.0 * math.log2(frequency_hz / target_frequency)
