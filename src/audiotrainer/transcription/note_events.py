"""Convert pitch tracks into note events."""

from pathlib import Path

from audiotrainer.api.schemas import NoteEvent, PitchTrack


def pitch_track_to_notes(track: PitchTrack) -> list[NoteEvent]:
    """Segment a pitch track into stable note events."""

    raise NotImplementedError


def export_notes_csv(events: list[NoteEvent], path: str | Path) -> Path:
    """Write note events to CSV."""

    raise NotImplementedError
