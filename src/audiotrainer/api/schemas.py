"""Pydantic result models used by the public AudioTrainer API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Note(BaseModel):
    """Equal-tempered musical note metadata."""

    name: str
    midi: int
    frequency_hz: float
    cents: float = 0.0


class PitchFrame(BaseModel):
    """Pitch estimate for one analysis frame."""

    time: float
    frequency_hz: float | None
    confidence: float = Field(ge=0.0, le=1.0)
    note: str | None
    cents: float | None


class PitchTrack(BaseModel):
    """Framewise pitch estimates for an audio signal."""

    sample_rate: int
    frames: list[PitchFrame]


class PitchScore(BaseModel):
    """Pitch accuracy and stability summary."""

    accuracy: float = Field(ge=0.0, le=1.0)
    stability: float = Field(ge=0.0, le=1.0)
    mean_abs_cents: float | None
    voiced_frame_count: int
    target_note: str | None = None


class NoteEvent(BaseModel):
    """A stable note segment inferred from a pitch track."""

    start_time: float
    end_time: float
    frequency_hz: float
    note: str
    confidence: float = Field(ge=0.0, le=1.0)


class PauseReport(BaseModel):
    """Pause timing summary for speech-like audio."""

    pause_count: int
    total_pause_time: float
    mean_pause_time: float | None
    pauses: list[tuple[float, float]]


class ProsodyReport(BaseModel):
    """Prosody features for speech coaching."""

    duration: float
    mean_pitch_hz: float | None
    pitch_range_semitones: float | None
    mean_intensity: float
    pause_count: int
    estimated_speech_rate: float | None
    monotony_score: float = Field(ge=0.0, le=1.0)


class PronunciationReport(BaseModel):
    """Reference-comparison report without phoneme-level claims."""

    duration_ratio: float | None
    pitch_similarity: float | None
    energy_similarity: float | None
    pause_similarity: float | None
    overall_score: float = Field(ge=0.0, le=1.0)
    explanation: str


class VocalRange(BaseModel):
    """Estimated stable vocal range."""

    lowest_note: str | None
    highest_note: str | None
    stable_range_semitones: float
    confidence: float = Field(ge=0.0, le=1.0)


class VoiceTypeEstimate(BaseModel):
    """Probabilistic voice type estimate."""

    primary_type: str
    secondary_type: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


class InstrumentFeatureVector(BaseModel):
    """Compact feature vector for lightweight instrument recognition."""

    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    zero_crossing_rate: float
    rms: float
    harmonic_ratio: float
    mfcc: list[float]


class InstrumentEstimate(BaseModel):
    """Rule-based instrument estimate."""

    label: Literal["voice", "piano", "guitar", "violin", "flute", "saxophone", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


class FeedbackItem(BaseModel):
    """Actionable coaching feedback."""

    severity: Literal["info", "warning", "critical"]
    category: Literal["pitch", "timing", "pronunciation", "voice", "instrument"]
    message: str
    suggestion: str
