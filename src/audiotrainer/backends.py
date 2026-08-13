"""Stable capability discovery for the deterministic AudioTrainer release."""

from __future__ import annotations

def capabilities() -> dict[str, object]:
    """Return stable public capabilities for UI, CLI, and API clients."""

    return {
        "version": "0.2.0",
        "offline_core": True,
        "pitch_backends": ["yin"],
        "speech_backends": ["baseline"],
        "instrument_backends": ["baseline"],
        "monophonic_transcription": True,
        "phoneme_scoring": False,
    }
