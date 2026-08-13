"""Thread-safe rolling audio buffer for browser microphone streams."""

from __future__ import annotations

from collections import deque
from threading import Lock

import numpy as np
from numpy.typing import NDArray

from audiotrainer.audio.preprocessing import resample_audio, to_mono


class RollingAudioBuffer:
    """Collect streaming chunks and expose a fixed recent analysis window."""

    def __init__(self, *, target_sr: int = 16_000, max_seconds: float = 3.0) -> None:
        if target_sr <= 0 or max_seconds <= 0:
            raise ValueError("target_sr and max_seconds must be positive")
        self.target_sr = target_sr
        self.max_samples = int(target_sr * max_seconds)
        self._chunks: deque[NDArray[np.float64]] = deque()
        self._sample_count = 0
        self._lock = Lock()

    def append(self, audio: NDArray[np.floating], sr: int) -> None:
        signal = to_mono(audio)
        if sr != self.target_sr:
            signal = resample_audio(signal, sr, self.target_sr)
        with self._lock:
            self._chunks.append(np.asarray(signal, dtype=np.float64))
            self._sample_count += signal.size
            while self._chunks and self._sample_count - self._chunks[0].size >= self.max_samples:
                removed = self._chunks.popleft()
                self._sample_count -= removed.size

    def latest(self, seconds: float = 0.35) -> NDArray[np.float64]:
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        requested = min(self.max_samples, int(round(seconds * self.target_sr)))
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float64)
            combined = np.concatenate(list(self._chunks))
        return combined[-requested:].copy()

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._sample_count = 0
