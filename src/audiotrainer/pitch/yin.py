"""YIN pitch detection baseline."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from audiotrainer.api.schemas import PitchFrame, PitchTrack
from audiotrainer.audio.framing import frame_audio, frame_times
from audiotrainer.audio.preprocessing import remove_dc, rms_energy
from audiotrainer.pitch.notes import cents_error, hz_to_note
from audiotrainer.pitch.smoothing import median_smooth_pitch_track


def detect_pitch(
    audio: NDArray[np.floating],
    sr: int,
    *,
    frame_length: int = 2_048,
    hop_length: int = 256,
    fmin: float = 50.0,
    fmax: float = 1_200.0,
    trough_threshold: float = 0.18,
    min_rms: float = 1e-4,
    smooth: bool = True,
) -> PitchTrack:
    """Estimate a framewise monophonic pitch track using a compact YIN baseline."""

    if sr <= 0:
        raise ValueError("sr must be positive")
    signal = remove_dc(audio)
    frames = frame_audio(signal, frame_length=frame_length, hop_length=hop_length)
    times = frame_times(frames.shape[0], sr, hop_length)

    output_frames: list[PitchFrame] = []
    global_rms = rms_energy(signal)
    energy_floor = max(min_rms, global_rms * 0.02)
    for time, frame in zip(times, frames, strict=True):
        frequency, confidence = _yin_frame(
            frame,
            sr,
            fmin=fmin,
            fmax=fmax,
            trough_threshold=trough_threshold,
            min_rms=energy_floor,
        )
        if frequency is None:
            output_frames.append(
                PitchFrame(time=float(time), frequency_hz=None, confidence=confidence, note=None, cents=None)
            )
            continue

        note = hz_to_note(frequency)
        output_frames.append(
            PitchFrame(
                time=float(time),
                frequency_hz=frequency,
                confidence=confidence,
                note=note.name,
                cents=cents_error(frequency),
            )
        )

    track = PitchTrack(sample_rate=sr, frames=output_frames)
    return median_smooth_pitch_track(track) if smooth else track


def _yin_frame(
    frame: NDArray[np.floating],
    sr: int,
    *,
    fmin: float,
    fmax: float,
    trough_threshold: float,
    min_rms: float,
) -> tuple[float | None, float]:
    signal = np.asarray(frame, dtype=np.float64)
    signal = signal * np.hanning(signal.size)
    if rms_energy(signal) < min_rms:
        return None, 0.0

    min_tau = max(2, int(sr / fmax))
    max_tau = min(signal.size // 2, int(sr / fmin))
    if max_tau <= min_tau:
        return None, 0.0

    difference = _difference_function(signal, max_tau)
    cmnd = _cumulative_mean_normalized_difference(difference)
    tau = _absolute_threshold(cmnd, min_tau, max_tau, trough_threshold)
    if tau is None:
        search = cmnd[min_tau : max_tau + 1]
        tau = int(np.argmin(search)) + min_tau

    refined_tau = _parabolic_refine(cmnd, tau)
    if refined_tau <= 0:
        return None, 0.0
    confidence = float(np.clip(1.0 - cmnd[tau], 0.0, 1.0))
    if confidence < 0.1:
        return None, confidence
    return float(sr / refined_tau), confidence


def _difference_function(signal: NDArray[np.float64], max_tau: int) -> NDArray[np.float64]:
    difference = np.zeros(max_tau + 1, dtype=np.float64)
    for tau in range(1, max_tau + 1):
        delta = signal[:-tau] - signal[tau:]
        difference[tau] = float(np.dot(delta, delta))
    return difference


def _cumulative_mean_normalized_difference(difference: NDArray[np.float64]) -> NDArray[np.float64]:
    cmnd = np.ones_like(difference)
    cumulative = 0.0
    for tau in range(1, difference.size):
        cumulative += difference[tau]
        cmnd[tau] = difference[tau] * tau / cumulative if cumulative > 0 else 1.0
    return cmnd


def _absolute_threshold(
    cmnd: NDArray[np.float64],
    min_tau: int,
    max_tau: int,
    threshold: float,
) -> int | None:
    for tau in range(min_tau, max_tau + 1):
        if cmnd[tau] >= threshold:
            continue
        while tau + 1 <= max_tau and cmnd[tau + 1] < cmnd[tau]:
            tau += 1
        return tau
    return None


def _parabolic_refine(values: NDArray[np.float64], index: int) -> float:
    if index <= 0 or index >= values.size - 1:
        return float(index)
    left, center, right = values[index - 1], values[index], values[index + 1]
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return float(index)
    return float(index + 0.5 * (left - right) / denominator)
