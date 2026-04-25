import pytest

from audiotrainer.pitch.notes import (
    cents_error,
    frequency_to_midi,
    hz_to_note,
    midi_to_frequency,
    note_name_to_midi,
)


def test_a4_frequency_converts_to_midi_69() -> None:
    assert frequency_to_midi(440.0) == pytest.approx(69.0)
    assert midi_to_frequency(69) == pytest.approx(440.0)


def test_hz_to_note_reports_nearest_note_and_cents() -> None:
    note = hz_to_note(445.0)
    assert note.name == "A4"
    assert note.midi == 69
    assert note.cents == pytest.approx(19.56, abs=0.1)


def test_cents_error_accepts_note_names_and_midi_targets() -> None:
    assert cents_error(440.0, "A4") == pytest.approx(0.0)
    assert cents_error(466.1637615, "A#4") == pytest.approx(0.0, abs=1e-4)
    assert cents_error(440.0, 69) == pytest.approx(0.0)


def test_note_name_parser_supports_flats() -> None:
    assert note_name_to_midi("Db4") == note_name_to_midi("C#4")


def test_invalid_frequency_rejected() -> None:
    with pytest.raises(ValueError):
        frequency_to_midi(0)
