"""Note-event plotting."""

from __future__ import annotations

from audiotrainer.api.schemas import NoteEvent
from audiotrainer.pitch.notes import note_name_to_midi


def plot_piano_roll(events: list[NoteEvent], *, title: str = "Detected notes"):
    """Return a matplotlib piano-roll style figure for note events."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig, ax = plt.subplots(figsize=(9, 3.5))
    for event in events:
        midi = note_name_to_midi(event.note)
        ax.broken_barh([(event.start_time, event.end_time - event.start_time)], (midi - 0.4, 0.8), facecolors="#16a34a")
        ax.text(event.start_time, midi + 0.5, event.note, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MIDI note")
    ax.grid(True, alpha=0.25)
    return fig
