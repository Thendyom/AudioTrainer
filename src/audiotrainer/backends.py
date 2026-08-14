"""Capability discovery without importing heavy optional runtimes."""

from __future__ import annotations

from pathlib import Path

from audiotrainer.ml.manager import AISettings, get_ai_settings, get_model_capabilities


def capabilities(
    settings: AISettings | None = None,
    *,
    data_dir: str | Path | None = None,
) -> dict[str, object]:
    """Return supported, installed, and enabled backend information."""

    selected = settings or get_ai_settings()
    statuses = get_model_capabilities(selected, data_dir=data_dir)
    ready = {status.backend for status in statuses if status.available and status.enabled}
    return {
        "version": "0.2.0",
        "offline_core": True,
        "pitch_backends": ["yin", *(["pyin"] if "pyin" in ready else [])],
        "speech_backends": ["baseline", *(["faster-whisper"] if "faster-whisper-small" in ready else [])],
        "instrument_backends": ["baseline", *(["ast"] if "ast" in ready else [])],
        "supported_pitch_backends": ["yin", "pyin", "auto"],
        "supported_speech_backends": ["baseline", "faster-whisper", "auto"],
        "supported_instrument_backends": ["baseline", "ast", "auto"],
        "monophonic_transcription": True,
        "phoneme_scoring": False,
        "ai_supported": True,
        "ai_enabled": selected.enabled,
        "model_capabilities": statuses,
    }
