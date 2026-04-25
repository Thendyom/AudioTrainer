"""FastAPI application for AudioTrainer."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, UploadFile

from audiotrainer.api.service import (
    analyze_pitch_file,
    analyze_speech_file,
    analyze_voice_profile_file,
    classify_instrument_file,
    transcribe_file,
)

app = FastAPI(title="AudioTrainer", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/pitch")
async def pitch(file: UploadFile = File(...), target_note: str | None = None) -> dict:
    path = await _save_upload(file)
    try:
        track, score, feedback = analyze_pitch_file(path, target_note)
        return {
            "track": track.model_dump(mode="json"),
            "score": score.model_dump(mode="json"),
            "feedback": [item.model_dump(mode="json") for item in feedback],
        }
    finally:
        path.unlink(missing_ok=True)


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    path = await _save_upload(file)
    try:
        _, events = transcribe_file(path)
        return {"events": [event.model_dump(mode="json") for event in events]}
    finally:
        path.unlink(missing_ok=True)


@app.post("/speech")
async def speech(file: UploadFile = File(...)) -> dict:
    path = await _save_upload(file)
    try:
        report, feedback = analyze_speech_file(path)
        return {
            "prosody": report.model_dump(mode="json"),
            "feedback": [item.model_dump(mode="json") for item in feedback],
        }
    finally:
        path.unlink(missing_ok=True)


@app.post("/voice-profile")
async def voice_profile(file: UploadFile = File(...)) -> dict:
    path = await _save_upload(file)
    try:
        vocal_range, estimate, feedback = analyze_voice_profile_file(path)
        return {
            "range": vocal_range.model_dump(mode="json"),
            "estimate": estimate.model_dump(mode="json"),
            "feedback": [item.model_dump(mode="json") for item in feedback],
        }
    finally:
        path.unlink(missing_ok=True)


@app.post("/instrument")
async def instrument(file: UploadFile = File(...)) -> dict:
    path = await _save_upload(file)
    try:
        estimate = classify_instrument_file(path)
        return {"estimate": estimate.model_dump(mode="json")}
    finally:
        path.unlink(missing_ok=True)


async def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await file.read())
        return Path(temp.name)
