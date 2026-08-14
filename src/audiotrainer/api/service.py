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
from audiotrainer.ml.alignment import align_transcripts
from audiotrainer.ml.generative import generate_local_coaching
from audiotrainer.ml.instruments import classify_ast
from audiotrainer.ml.manager import BackendUnavailableError
from audiotrainer.ml.pitch import detect_pitch_backend
from audiotrainer.ml.speech import transcribe_file_local
from audiotrainer.pitch.yin import detect_pitch
from audiotrainer.speech.pronunciation import compare_reference_speech
from audiotrainer.speech.prosody import analyze_prosody, detect_speech_pitch
from audiotrainer.speech.voice_profile import classify_voice_type, estimate_vocal_range
from audiotrainer.transcription.note_events import pitch_track_to_notes
from audiotrainer.transcription.score_document import create_score_document


def analyze_pitch_file(
    path: str | Path,
    target_note: str | None = None,
    *,
    backend: str = "yin",
    ai_enabled: bool = False,
) -> tuple[PitchTrack, PitchScore, list[FeedbackItem]]:
    """Load a file, detect pitch, score it, and return feedback."""

    track, score, feedback, _, _ = analyze_pitch_file_details(path, target_note, backend=backend, ai_enabled=ai_enabled)
    return track, score, feedback


def analyze_pitch_file_details(
    path: str | Path,
    target_note: str | None = None,
    *,
    backend: str = "yin",
    ai_enabled: bool = False,
) -> tuple[PitchTrack, PitchScore, list[FeedbackItem], AudioQualityReport, AnalysisMetadata]:
    """Pitch analysis with quality and complete backend provenance."""

    started = perf_counter()
    audio, sr = load_audio(path)
    _validate_ai_request(backend, optional="pyin", ai_enabled=ai_enabled, feature="pitch")
    track, actual, fallback = _run_pitch_backend(audio, sr, backend=backend, ai_enabled=ai_enabled)
    selected_target = target_note or infer_target_note(track)
    score = score_pitch_accuracy(track, selected_target)
    feedback = generate_pitch_feedback(score)
    quality = analyze_quality(audio, sr, pitch_track=track)
    return (
        track,
        score,
        feedback,
        quality,
        AnalysisMetadata(
            requested_backend=backend,
            actual_backend=actual,
            fallback_reason=fallback,
            processing_time_ms=(perf_counter() - started) * 1000,
            warnings=list(quality.warnings),
        ),
    )


def transcribe_file(
    path: str | Path, *, backend: str = "yin", ai_enabled: bool = False
) -> tuple[PitchTrack, list[NoteEvent]]:
    """Load a file and convert detected pitch to note events."""

    audio, sr = load_audio(path)
    _validate_ai_request(backend, optional="pyin", ai_enabled=ai_enabled, feature="pitch")
    track, _, _ = _run_pitch_backend(audio, sr, backend=backend, ai_enabled=ai_enabled)
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


def analyze_voice_profile_file(
    path: str | Path, *, backend: str = "yin", ai_enabled: bool = False
) -> tuple[VocalRange, VoiceTypeEstimate, list[FeedbackItem]]:
    """Estimate vocal range, likely voice type, and feedback from a file."""

    _, vocal_range, estimate, feedback, _, _ = analyze_voice_profile_details(
        path, backend=backend, ai_enabled=ai_enabled
    )
    return vocal_range, estimate, feedback


def analyze_voice_profile_details(
    path: str | Path,
    *,
    backend: str = "yin",
    ai_enabled: bool = False,
) -> tuple[PitchTrack, VocalRange, VoiceTypeEstimate, list[FeedbackItem], AudioQualityReport, AnalysisMetadata]:
    """Return a complete voice profile without repeating pitch detection in the UI."""

    started = perf_counter()
    audio, sr = load_audio(path)
    _validate_ai_request(backend, optional="pyin", ai_enabled=ai_enabled, feature="pitch")
    track, actual, fallback = _run_pitch_backend(
        audio, sr, backend=backend, ai_enabled=ai_enabled, fmin=55.0, fmax=1_100.0
    )
    vocal_range = estimate_vocal_range(track)
    prosody = analyze_prosody(audio, sr, pitch_track=track)
    estimate = classify_voice_type(vocal_range, speaking_pitch=prosody.mean_pitch_hz)
    feedback = [*generate_voice_feedback(vocal_range), *generate_voice_feedback(estimate)]
    quality = analyze_quality(audio, sr, pitch_track=track)
    metadata = AnalysisMetadata(
        requested_backend=backend,
        actual_backend=actual,
        processing_time_ms=(perf_counter() - started) * 1000,
        fallback_reason=fallback,
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
    ai_enabled: bool = False,
    data_dir: str | Path | None = None,
) -> InstrumentAnalysisResult:
    """Classify with optional local AST and deterministic fallback."""

    requested = backend.lower().strip()
    if requested not in {"baseline", "auto", "ast"}:
        raise ValueError("instrument backend must be baseline, ast, or auto")
    _validate_ai_request(requested, optional="ast", ai_enabled=ai_enabled, feature="instruments")
    started = perf_counter()
    audio, sr = load_audio(path)
    quality = analyze_quality(audio, sr)
    actual = "baseline"
    fallback = None
    if requested in {"auto", "ast"}:
        try:
            estimate, candidates, margin = classify_ast(
                audio,
                sr,
                ai_enabled=ai_enabled,
                data_dir=data_dir,
                confidence_threshold=confidence_threshold,
                margin_threshold=margin_threshold,
            )
            actual = "ast"
        except BackendUnavailableError as exc:
            if requested == "ast":
                raise
            fallback = f"{exc}; used the deterministic instrument baseline."
        except RuntimeError as exc:
            if requested == "ast":
                raise
            fallback = f"{exc}; used the deterministic instrument baseline."
    if actual == "baseline":
        features = extract_instrument_features(audio, sr)
        estimate = classify_instrument(features)
        candidates = rank_instrument_candidates(features)
        top = candidates[0] if candidates else InstrumentCandidate(label="unknown", confidence=0.0)
        runner_up = candidates[1].confidence if len(candidates) > 1 else 0.0
        margin = max(0.0, top.confidence - runner_up)
    else:
        top = candidates[0] if candidates else InstrumentCandidate(label="unknown", confidence=0.0)
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
            actual_backend=actual,
            processing_time_ms=(perf_counter() - started) * 1000,
            fallback_reason=fallback,
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
    track, _, _ = _run_pitch_backend(audio, sr, backend="yin", ai_enabled=False)
    return analyze_quality(audio, sr, pitch_track=track)


def run_pitch_exercise(
    path: str | Path,
    target_notes: list[str],
    *,
    backend: str = "yin",
    ai_enabled: bool = False,
) -> PitchExerciseResult:
    """Score sustained-note or ordered note-sequence practice."""

    if not target_notes:
        raise ValueError("at least one target note is required")
    from audiotrainer.pitch.notes import note_name_to_midi

    for target in target_notes:
        note_name_to_midi(target)
    _validate_ai_request(backend, optional="pyin", ai_enabled=ai_enabled, feature="pitch")
    started = perf_counter()
    audio, sr = load_audio(path)
    track, actual, fallback = _run_pitch_backend(audio, sr, backend=backend, ai_enabled=ai_enabled)
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
            score = score.model_copy(update={"accuracy": 0.0, "stability": 0.0, "mean_abs_cents": None})
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
            actual_backend=actual,
            processing_time_ms=(perf_counter() - started) * 1000,
            fallback_reason=fallback,
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
    ai_enabled: bool = False,
) -> tuple[PitchTrack, list[NoteEvent], ScoreDocument, AnalysisMetadata]:
    """Transcribe a file into editable, quantized monophonic notation."""

    _validate_ai_request(backend, optional="pyin", ai_enabled=ai_enabled, feature="pitch")
    started = perf_counter()
    audio, sr = load_audio(path)
    track, actual, fallback = _run_pitch_backend(audio, sr, backend=backend, ai_enabled=ai_enabled)
    events = pitch_track_to_notes(track)
    document = create_score_document(events, bpm=bpm, time_signature=time_signature, quantization=quantization)
    return (
        track,
        events,
        document,
        AnalysisMetadata(
            requested_backend=backend,
            actual_backend=actual,
            processing_time_ms=(perf_counter() - started) * 1000,
            fallback_reason=fallback,
        ),
    )


def coach_speech_file(
    path: str | Path,
    *,
    reference_path: str | Path | None = None,
    goal: str = "balanced",
    language: str | None = None,
    backend: str = "baseline",
    ai_enabled: bool = False,
    data_dir: str | Path | None = None,
    generative_coaching: bool = False,
    generative_ai_enabled: bool | None = None,
    generative_endpoint: str = "http://127.0.0.1:11434",
    generative_model: str = "",
) -> SpeechCoachingResult:
    """Run prosody coaching with optional local ASR, word alignment, and local coaching."""

    requested = backend.lower().strip()
    if requested not in {"baseline", "auto", "faster-whisper"}:
        raise ValueError("speech backend must be baseline, faster-whisper, or auto")
    _validate_ai_request(requested, optional="faster-whisper", ai_enabled=ai_enabled, feature="speech")
    started = perf_counter()
    audio, sr = load_audio(path)
    speech_pitch = detect_speech_pitch(audio, sr)
    prosody = analyze_prosody(audio, sr, pitch_track=speech_pitch)
    quality = analyze_quality(audio, sr, pitch_track=speech_pitch)
    comparison = compare_speech_files(path, reference_path) if reference_path else None
    feedback = generate_speech_feedback(prosody, goal=goal)
    actual = "baseline"
    fallback = None
    transcript = None
    reference_transcript = None
    alignment = None
    warnings = list(quality.warnings)
    if requested in {"auto", "faster-whisper"}:
        try:
            transcript = transcribe_file_local(path, language=language, ai_enabled=ai_enabled, data_dir=data_dir)
            if reference_path:
                reference_transcript = transcribe_file_local(
                    reference_path, language=language, ai_enabled=ai_enabled, data_dir=data_dir
                )
                alignment = align_transcripts(transcript, reference_transcript)
            actual = "faster-whisper"
        except RuntimeError as exc:
            if requested == "faster-whisper":
                raise
            fallback = f"{exc}; used language-neutral delivery metrics only."
    ai_message = None
    ai_backend = None
    if generative_coaching:
        try:
            ai_message = generate_local_coaching(
                prosody=prosody,
                transcript=transcript,
                alignment=alignment,
                goal=goal,
                endpoint=generative_endpoint,
                model=generative_model,
                ai_enabled=ai_enabled if generative_ai_enabled is None else generative_ai_enabled,
            )
            ai_backend = f"local:{generative_model}"
        except RuntimeError as exc:
            warnings.append(str(exc))
    return SpeechCoachingResult(
        metadata=AnalysisMetadata(
            requested_backend=backend,
            actual_backend=actual,
            processing_time_ms=(perf_counter() - started) * 1000,
            fallback_reason=fallback,
            warnings=warnings,
        ),
        quality=quality,
        prosody=prosody,
        reference_comparison=comparison,
        transcript=transcript,
        reference_transcript=reference_transcript,
        word_alignment=alignment,
        ai_coaching_message=ai_message,
        ai_coaching_backend=ai_backend,
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


def _run_pitch_backend(audio, sr: int, *, backend: str, ai_enabled: bool, **options):
    """Keep the long-standing service-level YIN seam available to callers and tests."""

    if backend.lower().strip() in {"yin", "baseline"}:
        return detect_pitch(audio, sr, **options), "yin", None
    return detect_pitch_backend(audio, sr, backend=backend, ai_enabled=ai_enabled, **options)


def _validate_ai_request(requested: str, *, optional: str, ai_enabled: bool, feature: str) -> None:
    normalized = requested.lower().strip()
    if normalized == optional and not ai_enabled:
        from audiotrainer.ml.manager import BackendDisabledError

        raise BackendDisabledError(
            f"{feature} backend {optional} is disabled; explicitly enable local AI for this analysis"
        )
