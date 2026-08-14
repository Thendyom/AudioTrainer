"""Lazy Faster-Whisper transcription and language-aware word metrics."""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
import re

from audiotrainer.api.schemas import TranscriptReport, TranscriptWord
from audiotrainer.audio.io import load_audio
from audiotrainer.ml.manager import BackendUnavailableError, model_path, require_feature

ENGLISH_FILLERS = {"ah", "actually", "basically", "erm", "hmm", "like", "literally", "so", "uh", "um", "well"}


def transcribe_file_local(
    path: str | Path,
    *,
    language: str | None = None,
    ai_enabled: bool,
    data_dir: str | Path | None = None,
) -> TranscriptReport:
    """Transcribe with cached local weights; never initiates a download."""

    require_feature(ai_enabled, "speech", "faster-whisper")
    if find_spec("faster_whisper") is None:
        raise BackendUnavailableError('Faster-Whisper is unavailable; install with pip install ".[ml-speech]"')
    weights = model_path("speech", data_dir)
    if not weights.is_dir() or not any(item.is_file() for item in weights.rglob("*")):
        raise BackendUnavailableError("Faster-Whisper weights are not downloaded in Models & Privacy")
    model = _load_model(str(weights.resolve()))
    segments, info = model.transcribe(
        str(path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    words: list[TranscriptWord] = []
    texts: list[str] = []
    for segment in segments:
        texts.append(str(getattr(segment, "text", "")).strip())
        for item in getattr(segment, "words", None) or []:
            token = str(getattr(item, "word", "")).strip()
            if not token:
                continue
            words.append(
                TranscriptWord(
                    word=token,
                    start=max(0.0, float(getattr(item, "start", 0.0) or 0.0)),
                    end=max(0.0, float(getattr(item, "end", 0.0) or 0.0)),
                    confidence=float(max(0.0, min(1.0, getattr(item, "probability", 0.0) or 0.0))),
                )
            )
    audio, sr = load_audio(path)
    duration = float(len(audio) / sr) if sr else 0.0
    detected_language = getattr(info, "language", None) or language
    return build_transcript_report(
        text=" ".join(value for value in texts if value).strip(),
        language=detected_language,
        language_probability=getattr(info, "language_probability", None),
        duration=duration,
        words=words,
    )


def build_transcript_report(
    *,
    text: str,
    language: str | None,
    language_probability: float | None,
    duration: float,
    words: list[TranscriptWord],
) -> TranscriptReport:
    """Derive stable word-based metrics from adapter output."""

    normalized = [_normalize_word(item.word) for item in words]
    repetitions = [
        normalized[index]
        for index in range(1, len(normalized))
        if normalized[index] and normalized[index] == normalized[index - 1]
    ]
    pauses = [
        (previous.end, current.start)
        for previous, current in zip(words, words[1:])
        if current.start - previous.end >= 0.35
    ]
    language_code = (language or "").lower().split("-")[0]
    fillers = [word for word in normalized if word in ENGLISH_FILLERS] if language_code == "en" else None
    confidence = sum(item.confidence for item in words) / len(words) if words else None
    wpm = 60.0 * len(words) / duration if duration > 0 and words else None
    return TranscriptReport(
        text=text,
        language=language,
        language_probability=language_probability,
        confidence=confidence,
        duration=duration,
        words=words,
        word_count=len(words),
        words_per_minute=wpm,
        repetitions=repetitions,
        filler_words=fillers,
        pause_positions=pauses,
    )


@lru_cache(maxsize=2)
def _load_model(weights_path: str):
    from faster_whisper import WhisperModel

    return WhisperModel(weights_path, device="cpu", compute_type="int8")


def _normalize_word(value: str) -> str:
    return re.sub(r"[^\w']+", "", value.lower(), flags=re.UNICODE)
