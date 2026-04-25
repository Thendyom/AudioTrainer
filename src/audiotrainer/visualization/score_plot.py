"""Staff-style score plotting."""

from __future__ import annotations

import math

from audiotrainer.api.schemas import NoteEvent

STEP_INDEX = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
TREBLE_BOTTOM_LINE = STEP_INDEX["E"] + 4 * 7
STAFF_LINES = [0, 1, 2, 3, 4]
STAFF_TOP = STAFF_LINES[-1]
STAFF_BOTTOM = STAFF_LINES[0]


def plot_score(events: list[NoteEvent], *, title: str = "Score", bpm: int = 120):
    """Return a matplotlib figure with a readable treble-staff score sketch."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    if not events:
        ax.text(0.5, 0.5, "No notes detected", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    layout = _layout_events(events, bpm=bpm)
    width = max(8.0, layout[-1]["end_x"] + 1.2)
    left_margin = 0.9
    right_edge = width + 0.2
    for line in STAFF_LINES:
        ax.hlines(line, left_margin, right_edge, color="#1f2937", linewidth=1.2)

    ax.text(0.28, 1.85, "G", fontsize=30, fontweight="bold", color="#111827", ha="center", va="center")
    ax.vlines(left_margin, STAFF_BOTTOM, STAFF_TOP, color="#111827", linewidth=1.5)
    _draw_barlines(ax, layout, left_margin=left_margin, right_edge=right_edge)

    for item in layout:
        event = item["event"]
        x = item["x"]
        y = _staff_y(event.note)
        duration_name = _duration_name(event.end_time - event.start_time, bpm)
        _draw_ledger_lines(ax, x, y)
        head = Ellipse(
            (x, y),
            width=0.34,
            height=0.24,
            angle=-18,
            facecolor="#111827",
            edgecolor="#111827",
            linewidth=1.0,
            zorder=4,
        )
        ax.add_patch(head)
        _draw_stem(ax, x, y, duration_name)
        label_y = -1.2 if y >= 2.3 else 5.25
        ax.text(x, label_y, event.note, fontsize=9, color="#334155", ha="center", va="center")
        ax.text(x, label_y - 0.35 if label_y < 0 else label_y + 0.35, duration_name, fontsize=7, color="#64748b", ha="center")

    ax.set_title(title)
    ax.set_xlim(0, right_edge + 0.25)
    ax.set_ylim(-1.8, 5.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.margins(x=0.02)
    fig.tight_layout()
    return fig


def _layout_events(events: list[NoteEvent], *, bpm: int) -> list[dict]:
    cursor = 1.35
    measure_start = cursor
    layout = []
    for event in events:
        quarter_notes = max(0.5, (event.end_time - event.start_time) * bpm / 60.0)
        visual_width = _visual_width(quarter_notes)
        x = cursor + visual_width / 2.0
        layout.append(
            {
                "event": event,
                "x": x,
                "start_x": cursor,
                "end_x": cursor + visual_width,
                "measure_start": measure_start,
                "quarter_notes": quarter_notes,
            }
        )
        cursor += visual_width + 0.22
        if cursor - measure_start >= 4.2:
            measure_start = cursor
    return layout


def _visual_width(quarter_notes: float) -> float:
    if quarter_notes >= 3.2:
        return 1.65
    if quarter_notes >= 1.6:
        return 1.25
    if quarter_notes >= 0.75:
        return 0.95
    return 0.72


def _draw_barlines(ax, layout: list[dict], *, left_margin: float, right_edge: float) -> None:
    ax.vlines(left_margin, STAFF_BOTTOM, STAFF_TOP, color="#111827", linewidth=1.6)
    previous_measure = layout[0]["measure_start"]
    for item in layout[1:]:
        if not math.isclose(item["measure_start"], previous_measure):
            ax.vlines(item["start_x"] - 0.1, STAFF_BOTTOM, STAFF_TOP, color="#334155", linewidth=1.0)
            previous_measure = item["measure_start"]
    ax.vlines(right_edge, STAFF_BOTTOM, STAFF_TOP, color="#111827", linewidth=1.8)
    ax.vlines(right_edge - 0.08, STAFF_BOTTOM, STAFF_TOP, color="#111827", linewidth=0.8)


def _draw_stem(ax, x: float, y: float, duration_name: str) -> None:
    if duration_name in {"whole"}:
        return
    upward = y < 2.5
    stem_x = x + 0.15 if upward else x - 0.15
    stem_end = y + 2.2 if upward else y - 2.2
    ax.vlines(stem_x, y, stem_end, color="#111827", linewidth=1.2, zorder=3)
    if duration_name in {"eighth", "16th"}:
        flag_y = stem_end
        direction = -1 if upward else 1
        ax.plot(
            [stem_x, stem_x + 0.42],
            [flag_y, flag_y + direction * 0.35],
            color="#111827",
            linewidth=1.0,
            zorder=3,
        )
    if duration_name == "16th":
        flag_y = stem_end - 0.28 if upward else stem_end + 0.28
        direction = -1 if upward else 1
        ax.plot(
            [stem_x, stem_x + 0.36],
            [flag_y, flag_y + direction * 0.28],
            color="#111827",
            linewidth=1.0,
            zorder=3,
        )


def _draw_ledger_lines(ax, x: float, y: float) -> None:
    if y <= STAFF_BOTTOM - 1:
        for line in range(math.floor(y), STAFF_BOTTOM):
            ax.hlines(line, x - 0.28, x + 0.28, color="#1f2937", linewidth=1.0, zorder=2)
    elif y >= STAFF_TOP + 1:
        for line in range(STAFF_TOP + 1, math.ceil(y) + 1):
            ax.hlines(line, x - 0.28, x + 0.28, color="#1f2937", linewidth=1.0, zorder=2)


def _staff_y(note: str) -> float:
    step = note[0].upper()
    octave_text = note[1:]
    if octave_text.startswith("#") or octave_text.startswith("b"):
        octave_text = octave_text[1:]
    staff_index = STEP_INDEX[step] + int(octave_text) * 7
    return (staff_index - TREBLE_BOTTOM_LINE) / 2.0


def _duration_name(seconds: float, bpm: int) -> str:
    quarter_notes = max(0.0, seconds * bpm / 60.0)
    candidates = [(4.0, "whole"), (2.0, "half"), (1.0, "quarter"), (0.5, "eighth"), (0.25, "16th")]
    return min(candidates, key=lambda item: abs(item[0] - quarter_notes))[1]
