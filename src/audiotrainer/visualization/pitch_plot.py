"""Pitch curve plotting."""

from __future__ import annotations

from audiotrainer.api.schemas import PitchTrack


def plot_pitch_track(track: PitchTrack, *, title: str = "Pitch track"):
    """Return a matplotlib figure for a pitch track."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    times = [frame.time for frame in track.frames if frame.frequency_hz is not None]
    frequencies = [frame.frequency_hz for frame in track.frames if frame.frequency_hz is not None]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(times, frequencies, color="#2563eb", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(True, alpha=0.25)
    return fig
