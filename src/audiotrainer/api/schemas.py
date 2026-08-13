"""Pydantic result models used by the public AudioTrainer API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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
    pitch_contour: list[tuple[float, float]] = Field(default_factory=list)
    intensity_contour: list[tuple[float, float]] = Field(default_factory=list)
    pauses: list[tuple[float, float]] = Field(default_factory=list)


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


class AnalysisMetadata(BaseModel):
    """Provenance and runtime details for an analysis operation."""

    requested_backend: str = "auto"
    actual_backend: str
    processing_time_ms: float = Field(ge=0.0)
    fallback_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)


class AudioQualityReport(BaseModel):
    """Input quality measurements used to qualify coaching results."""

    duration: float = Field(ge=0.0)
    rms: float = Field(ge=0.0)
    peak: float = Field(ge=0.0)
    clipping_ratio: float = Field(ge=0.0, le=1.0)
    estimated_noise_floor: float = Field(ge=0.0)
    voiced_coverage: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class PitchExerciseNoteResult(BaseModel):
    """Result for one requested target in a pitch exercise."""

    target_note: str
    detected_note: str | None
    accuracy: float = Field(ge=0.0, le=1.0)
    stability: float = Field(ge=0.0, le=1.0)
    mean_abs_cents: float | None
    voiced_duration: float = Field(ge=0.0)
    missed: bool


class PitchExerciseResult(BaseModel):
    """Complete sustained-note or note-sequence exercise result."""

    metadata: AnalysisMetadata
    quality: AudioQualityReport
    track: PitchTrack
    target_notes: list[str]
    notes: list[PitchExerciseNoteResult]
    overall_accuracy: float = Field(ge=0.0, le=1.0)
    feedback: list[FeedbackItem] = Field(default_factory=list)


class ScoreEvent(BaseModel):
    """Beat-aligned monophonic score event."""

    kind: Literal["note", "rest"]
    start_beat: float = Field(ge=0.0)
    duration_beats: float = Field(gt=0.0)
    note: str | None = None
    frequency_hz: float | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    measure: int = Field(default=1, ge=1)
    tie_start: bool = False
    tie_stop: bool = False

    @model_validator(mode="after")
    def validate_kind(self) -> "ScoreEvent":
        if self.kind == "note" and self.note is None:
            raise ValueError("note events require a note name")
        if self.kind == "note" and not _is_note_name(self.note):
            raise ValueError("note events require a valid note name such as C4 or F#3")
        if self.kind == "rest" and self.note is not None:
            raise ValueError("rest events cannot contain a note name")
        return self


class ScoreDocument(BaseModel):
    """Editable, quantized monophonic score."""

    bpm: int = Field(default=120, gt=0, le=400)
    beats_per_measure: int = Field(default=4, gt=0, le=32)
    beat_unit: int = Field(default=4)
    quantization: int = Field(default=4, ge=1, le=32, description="Subdivisions per quarter note")
    events: list[ScoreEvent] = Field(default_factory=list)
    suggested_bpm: int | None = Field(default=None, gt=0, le=400)

    @model_validator(mode="after")
    def validate_document(self) -> "ScoreDocument":
        if self.beat_unit not in {1, 2, 4, 8, 16, 32}:
            raise ValueError("beat_unit must be a power-of-two notation value")
        previous_end = 0.0
        for event in sorted(self.events, key=lambda item: item.start_beat):
            if event.start_beat < previous_end - 1e-7:
                raise ValueError("score events must not overlap")
            previous_end = event.start_beat + event.duration_beats
        return self

    @property
    def time_signature(self) -> str:
        return f"{self.beats_per_measure}/{self.beat_unit}"


class SpeechCoachingResult(BaseModel):
    """Prosody, reference comparison, and actionable feedback."""

    metadata: AnalysisMetadata
    quality: AudioQualityReport
    prosody: ProsodyReport
    reference_comparison: PronunciationReport | None = None
    feedback: list[FeedbackItem] = Field(default_factory=list)


class InstrumentCandidate(BaseModel):
    """One normalized instrument candidate."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    raw_label: str | None = None


class InstrumentAnalysisResult(BaseModel):
    """Quality-qualified experimental instrument classification."""

    metadata: AnalysisMetadata
    quality: AudioQualityReport
    estimate: InstrumentEstimate
    candidates: list[InstrumentCandidate] = Field(default_factory=list)
    confidence_margin: float = Field(ge=0.0, le=1.0)
    experimental: bool = True


class PracticeSession(BaseModel):
    """Persisted local practice result. Audio is optional and opt-in."""

    session_id: str
    mode: str
    created_at: datetime
    duration: float = Field(ge=0.0)
    source: str
    backend: str
    settings: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    feedback: list[dict[str, Any]] = Field(default_factory=list)
    audio_path: str | None = None


class PitchAnalysisResponse(BaseModel):
    track: PitchTrack
    score: PitchScore
    feedback: list[FeedbackItem]


class TranscriptionResponse(BaseModel):
    events: list[NoteEvent]
    score: ScoreDocument | None = None
    metadata: AnalysisMetadata | None = None


class VoiceProfileResponse(BaseModel):
    range: VocalRange
    estimate: VoiceTypeEstimate
    feedback: list[FeedbackItem]


class InstrumentResponse(BaseModel):
    estimate: InstrumentEstimate


class CapabilitiesResponse(BaseModel):
    version: str
    offline_core: bool
    pitch_backends: list[str]
    speech_backends: list[str]
    instrument_backends: list[str]
    monophonic_transcription: bool
    phoneme_scoring: bool


def _is_note_name(value: str | None) -> bool:
    if not value or len(value) < 2:
        return False
    letter = value[0].upper()
    remainder = value[1:]
    if remainder.startswith(("#", "b", "B")):
        remainder = remainder[1:]
    if remainder.startswith("-"):
        remainder = remainder[1:]
    return letter in "ABCDEFG" and remainder.isdigit()
