"""Application-facing service functions built on the core library."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from audiotrainer.api.schemas import (
    AnalysisMetadata,
    AudioQualityReport,
    FeedbackItem,
    InstrumentAnalysisResult,
    InstrumentCandidate,
    InstrumentEstimate,
    NoteEvent,
    PitchExerciseNoteResult,
    PitchExerciseResult,
    PitchScore,
    PitchTrack,
    PronunciationReport,
    ProsodyReport,
    ScoreDocument,
    SpeechCoachingResult,
    VocalRange,
    VoiceTypeEstimate,
)
from audiotrainer.audio.io import load_audio
from audiotrainer.audio.quality import analyze_audio_quality as analyze_quality
from audiotrainer.coaching.feedback import (
    generate_pitch_feedback,
    generate_speech_feedback,
    generate_voice_feedback,
)
from audiotrainer.coaching.scoring import infer_target_note, score_pitch_accuracy
from audiotrainer.instruments.classifier import classify_instrument, rank_instrument_candidates
from audiotrainer.instruments.features import extract_instrument_features
from audiotrainer.pitch.yin import detect_pitch
from audiotrainer.speech.pronunciation import compare_reference_speech
from audiotrainer.speech.prosody import analyze_prosody, detect_speech_pitch
from audiotrainer.speech.voice_profile import classify_voice_type, estimate_vocal_range
from audiotrainer.transcription.note_events import pitch_track_to_notes
from audiotrainer.transcription.score_document import create_score_document


def analyze_pitch_file(path: str | Path, target_note: str | None = None) -> tuple[PitchTrack, PitchScore, list[FeedbackItem]]:
    """Load a file, detect pitch, score it, and return feedback."""

    audio, sr = load_audio(path)
    track = detect_pitch(audio, sr)
    selected_target = target_note or infer_target_note(track)
    score = score_pitch_accuracy(track, selected_target)
    return track, score, generate_pitch_feedback(score)


def transcribe_file(path: str | Path) -> tuple[PitchTrack, list[NoteEvent]]:
    """Load a file and convert detected pitch to note events."""

    audio, sr = load_audio(path)
    track = detect_pitch(audio, sr)
    return track, pitch_track_to_notes(track)


def analyze_speech_file(path: str | Path, goal: str = "balanced") -> tuple[ProsodyReport, list[FeedbackItem]]:
    """Load a speech file, analyze prosody, and return feedback."""

    audio, sr = load_audio(path)
    report = analyze_prosody(audio, sr)
    return report, generate_speech_feedback(report, goal=goal)


def compare_speech_files(user_path: str | Path, reference_path: str | Path) -> PronunciationReport:
    """Compare two speech files at the prosody level."""

    user_audio, user_sr = load_audio(user_path)
    ref_audio, ref_sr = load_audio(reference_path)
    if user_sr != ref_sr:
        from audiotrainer.audio.preprocessing import resample_audio

        user_audio = resample_audio(user_audio, user_sr, ref_sr)
        user_sr = ref_sr
    return compare_reference_speech(user_audio, ref_audio, user_sr)


def analyze_voice_profile_file(path: str | Path) -> tuple[VocalRange, VoiceTypeEstimate, list[FeedbackItem]]:
    """Estimate vocal range, likely voice type, and feedback from a file."""

    _, vocal_range, estimate, feedback, _, _ = analyze_voice_profile_details(path)
    return vocal_range, estimate, feedback


def analyze_voice_profile_details(
    path: str | Path,
) -> tuple[PitchTrack, VocalRange, VoiceTypeEstimate, list[FeedbackItem], AudioQualityReport, AnalysisMetadata]:
    """Return a complete voice profile without repeating pitch detection in the UI."""

    started = perf_counter()
    audio, sr = load_audio(path)
    track = detect_pitch(audio, sr, fmin=55.0, fmax=1_100.0)
    vocal_range = estimate_vocal_range(track)
    prosody = analyze_prosody(audio, sr, pitch_track=track)
    estimate = classify_voice_type(vocal_range, speaking_pitch=prosody.mean_pitch_hz)
    feedback = [*generate_voice_feedback(vocal_range), *generate_voice_feedback(estimate)]
    quality = analyze_quality(audio, sr, pitch_track=track)
    metadata = AnalysisMetadata(
        requested_backend="yin",
        actual_backend="yin",
        processing_time_ms=(perf_counter() - started) * 1000,
        warnings=list(quality.warnings),
    )
    return track, vocal_range, estimate, feedback, quality, metadata


def classify_instrument_file(path: str | Path) -> InstrumentEstimate:
    """Load a clip and classify likely instrument using the rule-based baseline."""

    audio, sr = load_audio(path)
    features = extract_instrument_features(audio, sr)
    return classify_instrument(features)


def analyze_instrument_file(
    path: str | Path,
    *,
    backend: str = "baseline",
    confidence_threshold: float = 0.30,
    margin_threshold: float = 0.08,
) -> InstrumentAnalysisResult:
    """Classify an instrument with the quality-qualified rule-based baseline."""

    _require_backend(backend, actual="baseline", kind="instrument")
    started = perf_counter()
    audio, sr = load_audio(path)
    quality = analyze_quality(audio, sr)
    features = extract_instrument_features(audio, sr)
    estimate = classify_instrument(features)
    candidates = rank_instrument_candidates(features)
    top = candidates[0] if candidates else InstrumentCandidate(label="unknown", confidence=0.0)
    runner_up = candidates[1].confidence if len(candidates) > 1 else 0.0
    margin = max(0.0, top.confidence - runner_up)
    if top.confidence < confidence_threshold or margin < margin_threshold or quality.rms < 1e-5:
        estimate = InstrumentEstimate(
            label="unknown",
            confidence=top.confidence,
            explanation="Experimental classifier confidence or separation was insufficient.",
        )
    warnings = ["Instrument classification is experimental.", *quality.warnings]
    return InstrumentAnalysisResult(
        metadata=AnalysisMetadata(
            requested_backend=backend,
            actual_backend="baseline",
            processing_time_ms=(perf_counter() - started) * 1000,
            warnings=warnings,
        ),
        quality=quality,
        estimate=estimate,
        candidates=candidates[:5],
        confidence_margin=margin,
    )


def analyze_audio_quality(path: str | Path) -> AudioQualityReport:
    """Analyze file recording quality without persisting data."""

    audio, sr = load_audio(path)
    track = detect_pitch(audio, sr)
    return analyze_quality(audio, sr, pitch_track=track)


def run_pitch_exercise(
    path: str | Path,
    target_notes: list[str],
    *,
    backend: str = "yin",
) -> PitchExerciseResult:
    """Score sustained-note or ordered note-sequence practice."""

    if not target_notes:
        raise ValueError("at least one target note is required")
    _require_backend(backend, actual="yin", kind="pitch")
    from audiotrainer.pitch.notes import note_name_to_midi

    for target in target_notes:
        note_name_to_midi(target)
    started = perf_counter()
    audio, sr = load_audio(path)
    track = detect_pitch(audio, sr)
    quality = analyze_quality(audio, sr, pitch_track=track)
    chunks = _split_frames(track.frames, len(target_notes))
    results: list[PitchExerciseNoteResult] = []
    feedback: list[FeedbackItem] = []
    for target, frames in zip(target_notes, chunks, strict=True):
        partial = PitchTrack(sample_rate=sr, frames=frames)
        score = score_pitch_accuracy(partial, target)
        frame_step = _frame_step(track)
        voiced_duration = score.voiced_frame_count * frame_step
        detected = infer_target_note(partial)
        missed = score.voiced_frame_count < 2
        if missed:
            score = score.model_copy(
                update={"accuracy": 0.0, "stability": 0.0, "mean_abs_cents": None}
            )
        results.append(
            PitchExerciseNoteResult(
                target_note=target,
                detected_note=detected,
                accuracy=score.accuracy,
                stability=score.stability,
                mean_abs_cents=score.mean_abs_cents,
                voiced_duration=voiced_duration,
                missed=missed,
            )
        )
        feedback.extend(generate_pitch_feedback(score))
    return PitchExerciseResult(
        metadata=AnalysisMetadata(
            requested_backend=backend,
            actual_backend="yin",
            processing_time_ms=(perf_counter() - started) * 1000,
            warnings=list(quality.warnings),
        ),
        quality=quality,
        track=track,
        target_notes=target_notes,
        notes=results,
        overall_accuracy=sum(item.accuracy for item in results) / len(results),
        feedback=feedback,
    )


def create_score_file(
    path: str | Path,
    *,
    bpm: int = 120,
    time_signature: str = "4/4",
    quantization: int = 4,
    backend: str = "yin",
) -> tuple[PitchTrack, list[NoteEvent], ScoreDocument, AnalysisMetadata]:
    """Transcribe a file into editable, quantized monophonic notation."""

    _require_backend(backend, actual="yin", kind="pitch")
    started = perf_counter()
    audio, sr = load_audio(path)
    track = detect_pitch(audio, sr)
    events = pitch_track_to_notes(track)
    document = create_score_document(events, bpm=bpm, time_signature=time_signature, quantization=quantization)
    return track, events, document, AnalysisMetadata(
        requested_backend=backend,
        actual_backend="yin",
        processing_time_ms=(perf_counter() - started) * 1000,
    )


def coach_speech_file(
    path: str | Path,
    *,
    reference_path: str | Path | None = None,
    goal: str = "balanced",
    language: str | None = None,
    backend: str = "baseline",
) -> SpeechCoachingResult:
    """Run deterministic prosody coaching and optional reference comparison.

    ``language`` remains accepted for source compatibility but is not used because
    this release intentionally performs no speech-to-text processing.
    """

    del language
    _require_backend(backend, actual="baseline", kind="speech")
    started = perf_counter()
    audio, sr = load_audio(path)
    speech_pitch = detect_speech_pitch(audio, sr)
    prosody = analyze_prosody(audio, sr, pitch_track=speech_pitch)
    quality = analyze_quality(audio, sr, pitch_track=speech_pitch)
    comparison = compare_speech_files(path, reference_path) if reference_path else None
    feedback = generate_speech_feedback(prosody, goal=goal)
    return SpeechCoachingResult(
        metadata=AnalysisMetadata(
            requested_backend=backend,
            actual_backend="baseline",
            processing_time_ms=(perf_counter() - started) * 1000,
            warnings=list(quality.warnings),
        ),
        quality=quality,
        prosody=prosody,
        reference_comparison=comparison,
        feedback=feedback,
    )


def _split_frames(frames, count: int):
    sizes = [len(frames) // count + (1 if index < len(frames) % count else 0) for index in range(count)]
    chunks, cursor = [], 0
    for size in sizes:
        chunks.append(frames[cursor : cursor + size])
        cursor += size
    return chunks


def _frame_step(track: PitchTrack) -> float:
    if len(track.frames) < 2:
        return 0.0
    return max(0.0, track.frames[1].time - track.frames[0].time)


def _require_backend(requested: str, *, actual: str, kind: str) -> None:
    """Accept the deterministic engine and the legacy ``auto`` alias only."""

    if requested.lower().strip() not in {actual, "auto"}:
        raise ValueError(f"{kind} backend must be {actual}; optional model backends are not part of this release")
