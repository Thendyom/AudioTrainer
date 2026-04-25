"""Spectrogram plotting."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def plot_spectrogram(audio: NDArray[np.floating], sr: int, *, title: str = "Spectrogram"):
    """Return a matplotlib spectrogram figure."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.specgram(np.asarray(audio, dtype=float), Fs=sr, NFFT=1024, noverlap=512, cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    return fig
