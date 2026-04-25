"""Musical note conversion helpers."""

from audiotrainer.api.schemas import Note


def frequency_to_midi(frequency_hz: float) -> float:
    """Convert frequency in hertz to a fractional MIDI note number."""

    raise NotImplementedError


def midi_to_frequency(midi_note: float) -> float:
    """Convert a MIDI note number to equal-tempered frequency in hertz."""

    raise NotImplementedError


def hz_to_note(frequency_hz: float) -> Note:
    """Convert a frequency to the nearest equal-tempered note."""

    raise NotImplementedError


def cents_error(frequency_hz: float, target_note: str | int | float | None = None) -> float:
    """Return cents deviation from the nearest note or a target note."""

    raise NotImplementedError
