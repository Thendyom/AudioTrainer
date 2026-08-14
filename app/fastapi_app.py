"""Typed FastAPI application for AudioTrainer v0.2."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

# Support running the app directly from a source checkout without an editable install.
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from audiotrainer.api.schemas import (
    AudioQualityReport,
    CapabilitiesResponse,
    InstrumentAnalysisResult,
    InstrumentResponse,
    PitchAnalysisResponse,
    PitchExerciseResult,
    PracticeSession,
    SpeechCoachingResult,
    TranscriptionResponse,
    VoiceProfileResponse,
)
from audiotrainer.api.service import (
    analyze_audio_quality,
    analyze_instrument_file,
    analyze_pitch_file_details,
    analyze_voice_profile_details,
    classify_instrument_file,
    coach_speech_file,
    create_score_file,
    run_pitch_exercise,
)
from audiotrainer.backends import capabilities
from audiotrainer.history import SessionRepository
from audiotrainer.ml.manager import (
    AISettings,
    BackendDisabledError,
    BackendUnavailableError,
    get_ai_settings,
    save_ai_settings,
)

app = FastAPI(title="AudioTrainer", version="0.2.0")
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.2.0"}


@app.get("/capabilities", response_model=CapabilitiesResponse)
def get_capabilities() -> dict[str, object]:
    return capabilities()


@app.get("/ai-settings", response_model=AISettings)
def ai_settings() -> AISettings:
    return get_ai_settings()


@app.put("/ai-settings", response_model=AISettings)
def update_ai_settings(settings: AISettings) -> AISettings:
    save_ai_settings(settings)
    return settings


@app.post("/quality", response_model=AudioQualityReport)
async def quality(file: UploadFile = File(...)) -> AudioQualityReport:
    path = await _save_upload(file)
    try:
        return _translate_errors(lambda: analyze_audio_quality(path))
    finally:
        path.unlink(missing_ok=True)


@app.post("/pitch", response_model=PitchAnalysisResponse)
async def pitch(
    file: UploadFile = File(...),
    target_note: str | None = None,
    persist: bool = Query(False),
    retain_audio: bool = Query(False),
    backend: str = Query("yin"),
    ai: bool = Query(False, description="Allow an explicitly selected optional local backend"),
) -> PitchAnalysisResponse:
    source = file.filename or "audio.wav"
    path = await _save_upload(file)
    try:
        track, score, feedback, quality_report, metadata = _translate_errors(
            lambda: analyze_pitch_file_details(path, target_note, backend=backend, ai_enabled=ai)
        )
        result = PitchAnalysisResponse(
            track=track, score=score, feedback=feedback, quality=quality_report, metadata=metadata
        )
        _persist_if_requested(
            persist,
            retain_audio,
            "pitch",
            path,
            source,
            metadata.actual_backend,
            {"target_note": target_note, "ai": ai},
            result,
            feedback,
            _track_duration(track),
        )
        return result
    finally:
        path.unlink(missing_ok=True)


@app.post("/pitch-exercise", response_model=PitchExerciseResult)
async def pitch_exercise(
    file: UploadFile = File(...),
    targets: str = Query(..., description="Comma-separated note names"),
    persist: bool = Query(False),
    retain_audio: bool = Query(False),
    backend: str = Query("yin"),
    ai: bool = Query(False),
) -> PitchExerciseResult:
    source = file.filename or "audio.wav"
    path = await _save_upload(file)
    try:
        target_notes = [value.strip() for value in targets.split(",") if value.strip()]
        result = _translate_errors(lambda: run_pitch_exercise(path, target_notes, backend=backend, ai_enabled=ai))
        _persist_if_requested(
            persist,
            retain_audio,
            "pitch",
            path,
            source,
            result.metadata.actual_backend,
            {"targets": target_notes},
            result,
            result.feedback,
            result.quality.duration,
        )
        return result
    finally:
        path.unlink(missing_ok=True)


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    bpm: int = Query(120, ge=1, le=400),
    time_signature: str = Query("4/4"),
    quantization: int = Query(4, ge=1, le=32),
    persist: bool = Query(False),
    retain_audio: bool = Query(False),
    backend: str = Query("yin"),
    ai: bool = Query(False),
) -> TranscriptionResponse:
    source = file.filename or "audio.wav"
    path = await _save_upload(file)
    try:
        track, events, score, metadata = _translate_errors(
            lambda: create_score_file(
                path,
                bpm=bpm,
                time_signature=time_signature,
                quantization=quantization,
                backend=backend,
                ai_enabled=ai,
            )
        )
        result = TranscriptionResponse(events=events, score=score, metadata=metadata)
        _persist_if_requested(
            persist,
            retain_audio,
            "score",
            path,
            source,
            metadata.actual_backend,
            {"bpm": bpm, "time_signature": time_signature, "quantization": quantization},
            result,
            [],
            _track_duration(track),
        )
        return result
    finally:
        path.unlink(missing_ok=True)


@app.post("/speech", response_model=SpeechCoachingResult)
async def speech(
    file: UploadFile = File(...),
    reference: UploadFile | None = File(None),
    goal: str = Query("balanced"),
    language: str | None = Query(None),
    backend: str = Query("baseline"),
    ai: bool = Query(False),
    generative_coaching: bool = Query(False),
    generative_endpoint: str = Query("http://127.0.0.1:11434"),
    generative_model: str = Query(""),
    persist: bool = Query(False),
    retain_audio: bool = Query(False),
) -> SpeechCoachingResult:
    source = file.filename or "audio.wav"
    path = await _save_upload(file)
    reference_path = None
    try:
        reference_path = await _save_upload(reference) if reference else None
        result = _translate_errors(
            lambda: coach_speech_file(
                path,
                reference_path=reference_path,
                goal=goal,
                language=language,
                backend=backend,
                ai_enabled=ai,
                generative_coaching=generative_coaching,
                generative_endpoint=generative_endpoint,
                generative_model=generative_model,
            )
        )
        _persist_if_requested(
            persist,
            retain_audio,
            "speech",
            path,
            source,
            result.metadata.actual_backend,
            {
                "goal": goal,
                "reference": reference is not None,
                "language": language,
                "ai": ai,
                "generative_coaching": generative_coaching,
            },
            result,
            result.feedback,
            result.quality.duration,
        )
        return result
    finally:
        path.unlink(missing_ok=True)
        if reference_path:
            reference_path.unlink(missing_ok=True)


@app.post("/voice-profile", response_model=VoiceProfileResponse)
async def voice_profile(
    file: UploadFile = File(...),
    persist: bool = Query(False),
    retain_audio: bool = Query(False),
    backend: str = Query("yin"),
    ai: bool = Query(False),
) -> VoiceProfileResponse:
    source = file.filename or "audio.wav"
    path = await _save_upload(file)
    try:
        _, vocal_range, estimate, feedback, quality_report, metadata = _translate_errors(
            lambda: analyze_voice_profile_details(path, backend=backend, ai_enabled=ai)
        )
        result = VoiceProfileResponse(
            range=vocal_range,
            estimate=estimate,
            feedback=feedback,
            quality=quality_report,
            metadata=metadata,
        )
        _persist_if_requested(
            persist,
            retain_audio,
            "voice",
            path,
            source,
            metadata.actual_backend,
            {"workflow": "guided-range"},
            result,
            feedback,
            quality_report.duration,
        )
        return result
    finally:
        path.unlink(missing_ok=True)


@app.post("/instrument", response_model=InstrumentResponse)
async def instrument(file: UploadFile = File(...)) -> InstrumentResponse:
    """Compatibility endpoint returning the original estimate shape."""

    path = await _save_upload(file)
    try:
        estimate = _translate_errors(lambda: classify_instrument_file(path))
        return InstrumentResponse(estimate=estimate)
    finally:
        path.unlink(missing_ok=True)


@app.post("/instrument-analysis", response_model=InstrumentAnalysisResult)
async def instrument_analysis(
    file: UploadFile = File(...),
    persist: bool = Query(False),
    retain_audio: bool = Query(False),
    backend: str = Query("baseline"),
    ai: bool = Query(False),
) -> InstrumentAnalysisResult:
    source = file.filename or "audio.wav"
    path = await _save_upload(file)
    try:
        result = _translate_errors(lambda: analyze_instrument_file(path, backend=backend, ai_enabled=ai))
        _persist_if_requested(
            persist,
            retain_audio,
            "instrument",
            path,
            source,
            result.metadata.actual_backend,
            {},
            result,
            [],
            result.quality.duration,
        )
        return result
    finally:
        path.unlink(missing_ok=True)


@app.get("/history", response_model=list[PracticeSession])
def history(mode: str | None = None, limit: int = Query(100, ge=1, le=1000)) -> list[PracticeSession]:
    return SessionRepository().list(mode=mode, limit=limit)


@app.get("/history/{session_id}", response_model=PracticeSession)
def history_detail(session_id: str) -> PracticeSession:
    session = SessionRepository().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Practice session not found")
    return session


@app.delete("/history/{session_id}", status_code=204)
def history_delete(session_id: str) -> None:
    if not SessionRepository().delete(session_id):
        raise HTTPException(status_code=404, detail="Practice session not found")


async def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
    if suffix not in {".wav", ".flac", ".ogg", ".aiff", ".aif"}:
        raise HTTPException(status_code=400, detail="Unsupported audio file type")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio upload exceeds 100 MB")
    if not data:
        raise HTTPException(status_code=400, detail="Audio upload is empty")
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(data)
        return Path(temp.name)


def _translate_errors(operation):
    try:
        return operation()
    except (BackendUnavailableError, BackendDisabledError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        status = (
            422
            if "note" in str(exc).lower() or "backend" in str(exc).lower() or "signature" in str(exc).lower()
            else 400
        )
        raise HTTPException(status_code=status, detail=str(exc)) from exc


def _persist_if_requested(
    persist: bool,
    retain_audio: bool,
    mode: str,
    path: Path,
    source: str,
    backend: str,
    settings: dict[str, object],
    result,
    feedback,
    duration: float,
) -> None:
    if retain_audio and not persist:
        raise HTTPException(status_code=422, detail="retain_audio=true requires persist=true")
    if not persist:
        return
    SessionRepository().save(
        mode=mode,
        duration=duration,
        source=source,
        backend=backend,
        settings=settings,
        result=_jsonable(result),
        feedback=_jsonable(feedback),
        audio_source=path,
        retain_audio=retain_audio,
    )


def _jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _track_duration(track) -> float:
    if not track.frames:
        return 0.0
    if len(track.frames) == 1:
        return track.frames[0].time
    return track.frames[-1].time + max(0.0, track.frames[1].time - track.frames[0].time)
