"""Simple staff-style score plotting."""

from __future__ import annotations

from audiotrainer.api.schemas import NoteEvent
from audiotrainer.pitch.notes import note_name_to_midi


def plot_score(events: list[NoteEvent], *, title: str = "Score sketch"):
    """Return a matplotlib figure with a simple treble staff and note heads."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig, ax = plt.subplots(figsize=(10, 3.2))
    if not events:
        ax.text(0.5, 0.5, "No notes detected", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    midi_values = [note_name_to_midi(event.note) for event in events]
    low = min(min(midi_values) - 2, 60)
    high = max(max(midi_values) + 2, 72)
    staff_lines = [64, 67, 71, 74, 77]
    for line in staff_lines:
        ax.hlines(line, 0, max(event.end_time for event in events) + 0.5, color="#374151", linewidth=0.8)

    for event, midi in zip(events, midi_values, strict=True):
        x = (event.start_time + event.end_time) / 2.0
        ax.scatter([x], [midi], s=170, color="#111827", zorder=3)
        if event.end_time - event.start_time <= 0.35:
            ax.vlines(x + 0.035, midi, midi + 5, color="#111827", linewidth=1.1)
        ax.text(x, midi - 1.6, event.note, ha="center", va="top", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch")
    ax.set_ylim(low, high)
    ax.set_yticks(sorted(set(staff_lines + midi_values)))
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    return fig
