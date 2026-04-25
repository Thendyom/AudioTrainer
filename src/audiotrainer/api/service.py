"""Application-facing service functions built on the core library."""

from __future__ import annotations

from pathlib import Path

from audiotrainer.api.schemas import (
    FeedbackItem,
    InstrumentEstimate,
    NoteEvent,
    PitchScore,
    PitchTrack,
    PronunciationReport,
    ProsodyReport,
    VocalRange,
    VoiceTypeEstimate,
)
from audiotrainer.audio.io import load_audio
from audiotrainer.coaching.feedback import (
    generate_pitch_feedback,
    generate_speech_feedback,
    generate_voice_feedback,
)
from audiotrainer.coaching.scoring import score_pitch_accuracy
from audiotrainer.instruments.classifier import classify_instrument
from audiotrainer.instruments.features import extract_instrument_features
from audiotrainer.pitch.yin import detect_pitch
from audiotrainer.speech.pronunciation import compare_reference_speech
from audiotrainer.speech.prosody import analyze_prosody
from audiotrainer.speech.voice_profile import classify_voice_type, estimate_vocal_range
from audiotrainer.transcription.note_events import pitch_track_to_notes


def analyze_pitch_file(path: str | Path, target_note: str | None = None) -> tuple[PitchTrack, PitchScore, list[FeedbackItem]]:
    """Load a file, detect pitch, score it, and return feedback."""

    audio, sr = load_audio(path)
    track = detect_pitch(audio, sr)
    score = score_pitch_accuracy(track, target_note)
    return track, score, generate_pitch_feedback(score)


def transcribe_file(path: str | Path) -> tuple[PitchTrack, list[NoteEvent]]:
    """Load a file and convert detected pitch to note events."""

    audio, sr = load_audio(path)
    track = detect_pitch(audio, sr)
    return track, pitch_track_to_notes(track)


def analyze_speech_file(path: str | Path) -> tuple[ProsodyReport, list[FeedbackItem]]:
    """Load a speech file, analyze prosody, and return feedback."""

    audio, sr = load_audio(path)
    report = analyze_prosody(audio, sr)
    return report, generate_speech_feedback(report)


def compare_speech_files(user_path: str | Path, reference_path: str | Path) -> PronunciationReport:
    """Compare two speech files at the prosody level."""

    user_audio, user_sr = load_audio(user_path)
    ref_audio, ref_sr = load_audio(reference_path)
    if user_sr != ref_sr:
        raise ValueError("Reference comparison currently requires matching sample rates")
    return compare_reference_speech(user_audio, ref_audio, user_sr)


def analyze_voice_profile_file(path: str | Path) -> tuple[VocalRange, VoiceTypeEstimate, list[FeedbackItem]]:
    """Estimate vocal range, likely voice type, and feedback from a file."""

    audio, sr = load_audio(path)
    track = detect_pitch(audio, sr, fmin=55.0, fmax=1_100.0)
    vocal_range = estimate_vocal_range(track)
    estimate = classify_voice_type(vocal_range)
    feedback = [*generate_voice_feedback(vocal_range), *generate_voice_feedback(estimate)]
    return vocal_range, estimate, feedback


def classify_instrument_file(path: str | Path) -> InstrumentEstimate:
    """Load a clip and classify likely instrument using the rule-based baseline."""

    audio, sr = load_audio(path)
    features = extract_instrument_features(audio, sr)
    return classify_instrument(features)
