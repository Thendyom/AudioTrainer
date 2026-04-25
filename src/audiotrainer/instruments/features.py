"""Instrument feature extraction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from audiotrainer.api.schemas import InstrumentFeatureVector
from audiotrainer.audio.preprocessing import remove_dc, rms_energy
from audiotrainer.pitch.autocorrelation import estimate_frequency_autocorrelation


def extract_instrument_features(audio: NDArray[np.floating], sr: int) -> InstrumentFeatureVector:
    """Extract compact spectral and temporal instrument features."""

    if sr <= 0:
        raise ValueError("sr must be positive")
    signal = remove_dc(audio)
    if signal.size == 0:
        return InstrumentFeatureVector(
            spectral_centroid=0.0,
            spectral_bandwidth=0.0,
            spectral_rolloff=0.0,
            zero_crossing_rate=0.0,
            rms=0.0,
            harmonic_ratio=0.0,
            mfcc=[0.0] * 13,
        )

    max_samples = min(signal.size, sr * 5)
    signal = signal[:max_samples]
    windowed = signal * np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sr)
    total = float(np.sum(spectrum))
    if total <= 1e-12:
        centroid = bandwidth = rolloff = 0.0
    else:
        centroid = float(np.sum(frequencies * spectrum) / total)
        bandwidth = float(np.sqrt(np.sum(np.square(frequencies - centroid) * spectrum) / total))
        cumulative = np.cumsum(spectrum)
        rolloff = float(frequencies[min(np.searchsorted(cumulative, 0.85 * cumulative[-1]), frequencies.size - 1)])

    zero_crossing_rate = _zero_crossing_rate(signal)
    harmonic_ratio = _harmonic_ratio(signal, sr)
    mfcc = _mfcc_like_coefficients(signal, sr)

    return InstrumentFeatureVector(
        spectral_centroid=centroid,
        spectral_bandwidth=bandwidth,
        spectral_rolloff=rolloff,
        zero_crossing_rate=zero_crossing_rate,
        rms=rms_energy(signal),
        harmonic_ratio=harmonic_ratio,
        mfcc=mfcc,
    )


def _zero_crossing_rate(signal: NDArray[np.float64]) -> float:
    if signal.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(np.signbit(signal))).astype(np.float64)))


def _harmonic_ratio(signal: NDArray[np.float64], sr: int) -> float:
    frame_size = min(signal.size, max(1024, int(0.08 * sr)))
    if frame_size < 32:
        return 0.0
    _, confidence = estimate_frequency_autocorrelation(signal[:frame_size], sr, fmin=50, fmax=1_500)
    return confidence


def _mfcc_like_coefficients(signal: NDArray[np.float64], sr: int, coefficient_count: int = 13) -> list[float]:
    power = np.square(np.abs(np.fft.rfft(signal * np.hanning(signal.size))))
    filters = _mel_filterbank(sr, signal.size, filter_count=24)
    energies = np.maximum(filters @ power, 1e-12)
    log_energies = np.log(energies)
    try:
        from scipy.fftpack import dct

        coeffs = dct(log_energies, type=2, norm="ortho")[:coefficient_count]
    except ImportError:  # pragma: no cover - scipy is a core dependency
        coeffs = np.fft.rfft(log_energies).real[:coefficient_count]
    if coeffs.size < coefficient_count:
        coeffs = np.pad(coeffs, (0, coefficient_count - coeffs.size))
    return [float(value) for value in coeffs]


def _mel_filterbank(sr: int, fft_size: int, filter_count: int) -> NDArray[np.float64]:
    min_mel = _hz_to_mel(50.0)
    max_mel = _hz_to_mel(sr / 2.0)
    mel_points = np.linspace(min_mel, max_mel, filter_count + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((fft_size + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, fft_size // 2)
    filters = np.zeros((filter_count, fft_size // 2 + 1), dtype=np.float64)
    for index in range(filter_count):
        left, center, right = bins[index], bins[index + 1], bins[index + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        filters[index, left:center] = np.linspace(0.0, 1.0, max(1, center - left), endpoint=False)
        filters[index, center:right] = np.linspace(1.0, 0.0, max(1, right - center), endpoint=False)
    return filters


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: NDArray[np.float64]) -> NDArray[np.float64]:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)
