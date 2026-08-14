"""Local-first Streamlit product for AudioTrainer v0.2."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import streamlit as st

# Support running the app directly from a source checkout without an editable install.
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from audiotrainer.api.schemas import ScoreDocument, ScoreEvent
from audiotrainer.api.service import (
    analyze_audio_quality,
    analyze_instrument_file,
    analyze_pitch_file_details,
    analyze_voice_profile_details,
    coach_speech_file,
    create_score_file,
    run_pitch_exercise,
)
from audiotrainer.audio.io import load_audio
from audiotrainer.audio.quality import analyze_audio_quality as analyze_quality_signal
from audiotrainer.history import SessionRepository
from audiotrainer.ml.manager import (
    AISettings,
    BackendDisabledError,
    BackendUnavailableError,
    download_model,
    get_ai_settings,
    get_model_capabilities,
    remove_model,
    save_ai_settings,
)
from audiotrainer.pitch.notes import midi_to_frequency, note_name_to_midi
from audiotrainer.transcription import (
    export_midi,
    export_notes_csv,
    export_score_musicxml,
    score_document_to_note_events,
)

st.set_page_config(page_title="AudioTrainer", page_icon="🎧", layout="wide")

PAGES = ["Dashboard", "Pitch", "Score", "Speech", "Voice", "Instruments", "Models & Privacy"]
INSTRUMENT_TRANSPOSITIONS = {
    "Concert pitch": 0,
    "Tenor/soprano saxophone, trumpet, clarinet (Bb)": 2,
    "Alto/baritone saxophone (Eb)": -3,
    "French horn (F)": 7,
}


def main() -> None:
    st.title("AudioTrainer")
    st.caption("Fast, private audio practice with deterministic analysis and optional local AI")
    ai_settings = get_ai_settings(get_repository())
    ai_enabled = st.sidebar.toggle(
        "Enable optional local AI",
        value=ai_settings.enabled,
        help="Off guarantees that only the deterministic engines run.",
    )
    if ai_enabled != ai_settings.enabled:
        ai_settings = ai_settings.model_copy(update={"enabled": ai_enabled})
        save_ai_settings(ai_settings, get_repository())
    page = st.sidebar.radio("Workspace", PAGES)
    st.sidebar.toggle(
        "Retain the next analyzed recording",
        value=bool(get_repository().get_setting("retain_audio_default", False)),
        key="retain-next-analysis",
        help="Off keeps metrics and reports but deletes the temporary recording after analysis.",
    )
    st.sidebar.caption("v0.2.0 · " + ("Optional local AI enabled" if ai_settings.enabled else "Deterministic mode"))
    if page == "Dashboard":
        dashboard_page()
    elif page == "Pitch":
        pitch_page()
    elif page == "Score":
        score_page()
    elif page == "Speech":
        speech_page()
    elif page == "Voice":
        voice_page()
    elif page == "Instruments":
        instruments_page()
    else:
        models_privacy_page()


def dashboard_page() -> None:
    st.header("Practice dashboard", anchor="practice-dashboard")
    repository = get_repository()
    controls = st.columns(3)
    mode = controls[0].selectbox("Mode", ["All", "pitch", "score", "speech", "voice", "instrument"])
    period = controls[1].selectbox("Period", ["All time", "7 days", "30 days", "90 days"])
    limit = controls[2].selectbox("Maximum rows", [50, 100, 500], index=1)
    days = None if period == "All time" else int(period.split()[0])
    since = datetime.now(timezone.utc) - timedelta(days=days) if days else None
    sessions = repository.list(mode=None if mode == "All" else mode, since=since, limit=limit)
    if not sessions:
        st.info("Complete an analysis to start building private local practice history.")
        return
    metrics = st.columns(4)
    metrics[0].metric("Sessions", len(sessions))
    metrics[1].metric("Practice time", f"{sum(item.duration for item in sessions) / 60:.1f} min")
    metrics[2].metric("Modes", len({item.mode for item in sessions}))
    metrics[3].metric("Audio retained", sum(bool(item.audio_path) for item in sessions))
    rows = [
        {
            "created": session.created_at.isoformat(timespec="minutes"),
            "mode": session.mode,
            "duration_s": round(session.duration, 2),
            "backend": session.backend,
            "score": _session_score(session.result),
            "id": session.session_id,
        }
        for session in reversed(sessions)
    ]
    trend_rows = [row for row in rows if row["score"] is not None]
    if trend_rows:
        st.subheader("Comparable score trend")
        st.line_chart(trend_rows, x="created", y="score", color="mode")
    st.dataframe(list(reversed(rows)), width="stretch", hide_index=True)
    selected = st.selectbox(
        "Session details",
        [session.session_id for session in sessions],
        format_func=lambda value: _session_label(value, sessions),
    )
    detail = repository.get(selected)
    if detail:
        st.json(detail.model_dump(mode="json"), expanded=False)
        if st.button("Delete selected session"):
            repository.delete(selected)
            st.rerun()
    export_cols = st.columns(3)
    export_cols[0].download_button(
        "Download history JSON", repository.export_json(), "audiotrainer-history.json", "application/json"
    )
    with NamedTemporaryFile(suffix=".csv") as temp:
        csv_path = repository.export_csv(temp.name)
        export_cols[1].download_button(
            "Download history CSV", csv_path.read_bytes(), "audiotrainer-history.csv", "text/csv"
        )
    confirm = export_cols[2].checkbox("Confirm clear all", key="clear-history-confirm")
    if export_cols[2].button("Clear all history", disabled=not confirm):
        repository.clear()
        st.rerun()


def pitch_page() -> None:
    st.header("Pitch trainer", anchor="pitch-trainer")
    ai_settings = get_ai_settings(get_repository())
    backend = backend_selector("pitch", ai_settings, "yin", "pyin")
    render_microphone_check("pitch-quality")
    mode = st.radio(
        "Practice mode",
        ["Free tuner", "Sustained note", "Note sequence"],
        horizontal=True,
    )
    instrument = st.selectbox("Notation view", list(INSTRUMENT_TRANSPOSITIONS))
    transposition = INSTRUMENT_TRANSPOSITIONS[instrument]
    if mode == "Free tuner":
        render_live_pitch(instrument, transposition)
        uploaded = audio_input("pitch-free")
        target = st.text_input("Written target (optional)", placeholder="Leave blank to infer, or enter A4")
        if uploaded and st.button("Analyze recording", type="primary", key="analyze-pitch-free"):
            with temp_audio(uploaded) as path, st.spinner("Analyzing pitch…"):
                try:
                    concert_target = _concert_note(target, transposition) if target else None
                    track, score, feedback, quality, metadata = analyze_pitch_file_details(
                        path,
                        concert_target,
                        backend=backend,
                        ai_enabled=ai_settings.feature_enabled("pitch"),
                    )
                    payload = {
                        "track": track,
                        "score": score,
                        "feedback": feedback,
                        "quality": quality,
                        "metadata": metadata,
                        "instrument": instrument,
                    }
                    st.session_state["pitch-free-result"] = payload
                    save_once("pitch", path, uploaded, metadata.actual_backend, score.model_dump(mode="json"), feedback)
                except (ValueError, BackendUnavailableError, BackendDisabledError) as exc:
                    st.error(str(exc))
        payload = st.session_state.get("pitch-free-result")
        if payload:
            st.pyplot(get_plot_pitch_track()(payload["track"]), clear_figure=True)
            render_pitch_score(payload["score"], payload["instrument"])
            render_quality(payload["quality"])
            render_metadata(payload["metadata"])
            feedback_table(payload["feedback"])
        return

    default_targets = "A4" if mode == "Sustained note" else "C4,D4,E4,F4,G4"
    targets_text = st.text_input("Written target note(s)", value=default_targets)
    targets = [value.strip() for value in targets_text.split(",") if value.strip()]
    try:
        concert_targets = [_concert_note(value, transposition) for value in targets]
    except ValueError as exc:
        concert_targets = []
        st.error(f"Check the target notes: {exc}")
    if targets:
        tones = generate_reference_tones(concert_targets)
        st.audio(tones, sample_rate=22_050)
        st.caption("Reference tones play each target for 0.65 seconds with a short gap.")
    uploaded = audio_input("pitch-exercise")
    if uploaded and st.button("Score exercise", type="primary") and concert_targets:
        with temp_audio(uploaded) as path, st.spinner("Scoring exercise…"):
            try:
                result = run_pitch_exercise(
                    path,
                    concert_targets,
                    backend=backend,
                    ai_enabled=ai_settings.feature_enabled("pitch"),
                )
                st.session_state["pitch-exercise-result"] = (result, targets)
                save_once(
                    "pitch",
                    path,
                    uploaded,
                    result.metadata.actual_backend,
                    result.model_dump(mode="json"),
                    result.feedback,
                )
            except (ValueError, BackendUnavailableError, BackendDisabledError) as exc:
                st.error(str(exc))
    exercise_payload = st.session_state.get("pitch-exercise-result")
    if exercise_payload:
        result, written_targets = exercise_payload
        display_rows = [
            {**item.model_dump(), "target_note": written}
            for written, item in zip(written_targets, result.notes, strict=True)
        ]
        st.metric("Overall accuracy", f"{result.overall_accuracy:.0%}")
        st.dataframe(display_rows, width="stretch", hide_index=True)
        st.pyplot(get_plot_pitch_track()(result.track), clear_figure=True)
        render_quality(result.quality)
        render_metadata(result.metadata)
        feedback_table(result.feedback)


def score_page() -> None:
    st.header("Monophonic score creator", anchor="score-creator")
    ai_settings = get_ai_settings(get_repository())
    backend = backend_selector("pitch", ai_settings, "yin", "pyin", key="score-pitch-backend")
    st.caption("Creates one melody line. Chords and polyphonic audio are intentionally not claimed.")
    settings = st.columns(3)
    bpm = settings[0].number_input("Tempo (BPM)", 30, 300, 120)
    signature = settings[1].selectbox("Time signature", ["4/4", "3/4", "6/8"])
    quantization_label = settings[2].selectbox(
        "Smallest grid", ["Quarter", "Eighth", "Sixteenth", "Thirty-second"], index=2
    )
    quantization = {"Quarter": 1, "Eighth": 2, "Sixteenth": 4, "Thirty-second": 8}[quantization_label]
    uploaded = audio_input("score")
    if uploaded and st.button("Create score", type="primary"):
        with temp_audio(uploaded) as path, st.spinner("Detecting and quantizing notes…"):
            try:
                track, _, document, metadata = create_score_file(
                    path,
                    bpm=int(bpm),
                    time_signature=signature,
                    quantization=quantization,
                    backend=backend,
                    ai_enabled=ai_settings.feature_enabled("pitch"),
                )
                audio, sr = load_audio(path)
                quality = analyze_quality_signal(audio, sr, pitch_track=track)
                st.session_state["score-analysis"] = {
                    "document": document,
                    "metadata": metadata,
                    "quality": quality,
                }
                save_once("score", path, uploaded, metadata.actual_backend, document.model_dump(mode="json"), [])
            except (ValueError, BackendUnavailableError, BackendDisabledError) as exc:
                st.error(str(exc))
    score_payload = st.session_state.get("score-analysis")
    if score_payload:
        document = score_payload["document"]
        if document.suggested_bpm:
            st.info(f"Onset-based tempo suggestion: {document.suggested_bpm} BPM. It was not applied automatically.")
        edited = edit_score_document(document)
        score_payload["document"] = edited
        render_score_document(edited)
        render_metadata(score_payload["metadata"])
        render_quality(score_payload["quality"])


def speech_page() -> None:
    st.header("Speech coach", anchor="speech-coach")
    st.caption("Pitch, intensity, pauses, pace, and reference delivery comparison. This is not phoneme scoring.")
    goal = st.selectbox(
        "Speaking goal",
        ["balanced", "clear pronunciation", "presenter presence", "charismatic delivery"],
    )
    ai_settings = get_ai_settings(get_repository())
    backend = backend_selector("speech", ai_settings, "baseline", "faster-whisper")
    language = st.text_input(
        "Transcription language (optional)",
        placeholder="Leave blank to detect automatically, or enter en, de, fr…",
        disabled=backend == "baseline",
    )
    use_generative = st.toggle(
        "Add local generative coaching",
        value=False,
        disabled=not ai_settings.feature_enabled("generative_coaching"),
        help="Sends measured metrics and the transcript—not audio—to your configured localhost model.",
    )
    uploaded = audio_input("speech-user", label="Your recording")
    reference = audio_input("speech-reference", label="Optional reference")
    if uploaded and st.button("Analyze speech", type="primary"):
        with temp_audio(uploaded) as path, st.spinner("Analyzing delivery…"):
            if reference:
                with temp_audio(reference) as reference_path:
                    analyze_speech_result(
                        path, uploaded, reference_path, goal, backend, language, use_generative, ai_settings
                    )
            else:
                analyze_speech_result(path, uploaded, None, goal, backend, language, use_generative, ai_settings)
    speech_result = st.session_state.get("speech-result")
    if speech_result:
        render_speech_result(speech_result)


def voice_page() -> None:
    st.header("Voice profile", anchor="voice-profile")
    ai_settings = get_ai_settings(get_repository())
    backend = backend_selector("pitch", ai_settings, "yin", "pyin", key="voice-pitch-backend")
    st.info(
        "Record a slow comfortable scale or siren from low to high. Stop before straining. This is not a medical or definitive voice classification."
    )
    render_microphone_check("voice-quality")
    uploaded = audio_input("voice-profile")
    if uploaded and st.button("Create voice profile", type="primary"):
        with temp_audio(uploaded) as path, st.spinner("Building voice profile…"):
            try:
                track, vocal_range, estimate, feedback, quality, metadata = analyze_voice_profile_details(
                    path,
                    backend=backend,
                    ai_enabled=ai_settings.feature_enabled("pitch"),
                )
                st.session_state["voice-result"] = (track, vocal_range, estimate, feedback, quality, metadata)
                result = {"range": vocal_range.model_dump(mode="json"), "estimate": estimate.model_dump(mode="json")}
                save_once("voice", path, uploaded, metadata.actual_backend, result, feedback)
            except (ValueError, BackendUnavailableError, BackendDisabledError) as exc:
                st.error(str(exc))
    voice_result = st.session_state.get("voice-result")
    if voice_result:
        track, vocal_range, estimate, feedback, quality, metadata = voice_result
        metrics = st.columns(5)
        metrics[0].metric("Lowest", vocal_range.lowest_note or "n/a")
        metrics[1].metric("Highest", vocal_range.highest_note or "n/a")
        metrics[2].metric("Stable span", f"{vocal_range.stable_range_semitones:.1f} st")
        metrics[3].metric("Likely overlap", estimate.primary_type)
        metrics[4].metric("Confidence", f"{estimate.confidence:.0%}")
        st.caption(estimate.explanation)
        st.pyplot(get_plot_pitch_track()(track), clear_figure=True)
        render_quality(quality)
        render_metadata(metadata)
        feedback_table(feedback)
        previous = get_repository().list(mode="voice", limit=5)
        if previous:
            st.caption(f"Compared with {len(previous)} recent voice session(s) in your dashboard.")


def instruments_page() -> None:
    st.header("Instrument lab", anchor="instrument-lab")
    ai_settings = get_ai_settings(get_repository())
    backend = backend_selector("instruments", ai_settings, "baseline", "ast")
    st.warning("Experimental: this is a broad estimate, not definitive instrument identification.")
    st.caption("The selected local classifier may return unknown when confidence or separation is insufficient.")
    uploaded = audio_input("instrument")
    if uploaded and st.button("Analyze instrument", type="primary"):
        with temp_audio(uploaded) as path, st.spinner("Classifying the clip…"):
            try:
                result = analyze_instrument_file(
                    path,
                    backend=backend,
                    ai_enabled=ai_settings.feature_enabled("instruments"),
                )
                st.session_state["instrument-result"] = result
                save_once(
                    "instrument", path, uploaded, result.metadata.actual_backend, result.model_dump(mode="json"), []
                )
            except (ValueError, BackendUnavailableError, BackendDisabledError) as exc:
                st.error(str(exc))
    result = st.session_state.get("instrument-result")
    if result:
        metrics = st.columns(4)
        metrics[0].metric("Estimate", result.estimate.label)
        metrics[1].metric("Confidence", f"{result.estimate.confidence:.0%}")
        metrics[2].metric("Top-two margin", f"{result.confidence_margin:.0%}")
        metrics[3].metric("Engine", result.metadata.actual_backend)
        st.caption(result.estimate.explanation)
        st.dataframe([item.model_dump() for item in result.candidates], width="stretch", hide_index=True)
        render_quality(result.quality)
        render_metadata(result.metadata)


def models_privacy_page() -> None:
    st.header("Models & Privacy", anchor="models-privacy")
    repository = get_repository()
    settings = get_ai_settings(repository)
    st.success("All analysis stays on this computer. Optional models are disabled globally with one switch.")
    st.subheader("Local AI controls")
    enabled = settings.enabled
    st.info(
        "Local AI is currently "
        + (
            "enabled. Use the master switch in the sidebar to turn every optional backend off."
            if enabled
            else "disabled. Use the master switch in the sidebar to enable optional backends."
        )
    )
    feature_cols = st.columns(4)
    pitch_enabled = feature_cols[0].toggle("pYIN pitch", value=settings.pitch_enabled, disabled=not enabled)
    speech_enabled = feature_cols[1].toggle("Speech transcription", value=settings.speech_enabled, disabled=not enabled)
    instruments_enabled = feature_cols[2].toggle(
        "AST instruments", value=settings.instruments_enabled, disabled=not enabled
    )
    generative_enabled = feature_cols[3].toggle(
        "Generative coaching", value=settings.generative_coaching_enabled, disabled=not enabled
    )
    endpoint = st.text_input("Local coaching endpoint", value=settings.generative_endpoint, disabled=not enabled)
    model = st.text_input(
        "Local coaching model name",
        value=settings.generative_model,
        placeholder="For example: llama3.2:3b",
        disabled=not (enabled and generative_enabled),
    )
    try:
        updated_settings = AISettings(
            enabled=enabled,
            pitch_enabled=pitch_enabled,
            speech_enabled=speech_enabled,
            instruments_enabled=instruments_enabled,
            generative_coaching_enabled=generative_enabled,
            speech_model_id=settings.speech_model_id,
            instrument_model_id=settings.instrument_model_id,
            generative_endpoint=endpoint,
            generative_model=model,
        )
        if updated_settings != settings:
            save_ai_settings(updated_settings, repository)
            settings = updated_settings
    except ValueError as exc:
        st.error(str(exc))
    statuses = get_model_capabilities(settings)
    st.dataframe(
        [
            {
                "feature": item.feature,
                "backend": item.backend,
                "enabled": item.enabled,
                "dependency": "installed" if item.dependency_installed else "missing",
                "weights": (
                    "not required"
                    if item.feature == "pitch"
                    else "installed"
                    if item.weights_installed
                    else "not installed"
                ),
                "disk": format_bytes(item.disk_usage_bytes),
                "status": item.status,
            }
            for item in statuses
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption("Model weights are never downloaded during startup or analysis. Use the explicit controls below.")
    for feature, title, url in [
        ("speech", "Faster-Whisper small", "https://huggingface.co/Systran/faster-whisper-small"),
        ("instruments", "AudioSet AST", "https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593"),
    ]:
        status = next(item for item in statuses if item.feature == feature)
        with st.expander(title):
            st.markdown(f"Model card and license: [{status.model_id}]({url})")
            st.code(status.install_command or "", language="bash")
            actions = st.columns(3)
            if actions[0].button(
                f"Download {title}", disabled=not status.dependency_installed, key=f"download-{feature}"
            ):
                with st.spinner(f"Downloading {title}…"):
                    try:
                        destination = download_model(feature, settings=settings)
                        st.success(f"Installed in {destination}")
                        st.rerun()
                    except (OSError, RuntimeError) as exc:
                        st.error(str(exc))
            confirm = actions[1].checkbox("Confirm removal", key=f"confirm-remove-{feature}")
            if actions[2].button(
                f"Remove {title}", disabled=not (status.weights_installed and confirm), key=f"remove-{feature}"
            ):
                remove_model(feature)
                st.success(f"Removed {title}")
                st.rerun()
    st.write(
        "Recordings are temporary by default. Retaining audio is opt-in for each analysis; metrics and reports are saved locally in SQLite."
    )
    retain_default = repository.get_setting("retain_audio_default", False)
    updated = st.toggle("Default to retaining analyzed audio", value=bool(retain_default))
    if updated != retain_default:
        repository.set_setting("retain_audio_default", updated)
    st.subheader("What is stored")
    st.write(
        "Completed metrics and coaching reports are saved in a local SQLite database so the dashboard survives restarts."
    )
    st.write("Recordings are deleted after analysis unless you explicitly enable retention before analyzing them.")
    st.subheader("Privacy boundaries")
    st.write(
        "No accounts, telemetry, cloud inference APIs, or remote storage are used. Generative coaching only accepts a localhost endpoint and never sends audio."
    )


def render_live_pitch(instrument: str, transposition: int) -> None:
    st.subheader("Quick tuner")
    st.info(
        "Record or upload a short tone below, then choose Analyze recording. This reliable workflow avoids extra streaming components."
    )


def render_microphone_check(key: str) -> None:
    with st.expander("Microphone setup check"):
        sample = audio_input(key, label="Short setup sample")
        if sample and st.button("Check microphone", key=f"{key}-analyze"):
            with temp_audio(sample) as path, st.spinner("Checking level and noise…"):
                try:
                    st.session_state[f"{key}-result"] = analyze_audio_quality(path)
                except ValueError as exc:
                    st.error(str(exc))
        if st.session_state.get(f"{key}-result"):
            render_quality(st.session_state[f"{key}-result"])


def analyze_speech_result(
    path: Path,
    uploaded,
    reference_path: Path | None,
    goal: str,
    backend: str,
    language: str,
    use_generative: bool,
    ai_settings: AISettings,
) -> None:
    """Run and persist speech analysis; rendering happens outside the action block."""

    try:
        result = coach_speech_file(
            path,
            reference_path=reference_path,
            goal=goal,
            language=language.strip() or None,
            backend=backend,
            ai_enabled=ai_settings.feature_enabled("speech"),
            generative_coaching=use_generative,
            generative_ai_enabled=ai_settings.feature_enabled("generative_coaching"),
            generative_endpoint=ai_settings.generative_endpoint,
            generative_model=ai_settings.generative_model,
        )
        st.session_state["speech-result"] = result
        save_once(
            "speech", path, uploaded, result.metadata.actual_backend, result.model_dump(mode="json"), result.feedback
        )
    except (ValueError, BackendUnavailableError, BackendDisabledError) as exc:
        st.error(str(exc))


def render_speech_result(result) -> None:
    report = result.prosody
    metrics = st.columns(5)
    metrics[0].metric(
        "Pitch range", "n/a" if report.pitch_range_semitones is None else f"{report.pitch_range_semitones:.1f} st"
    )
    metrics[1].metric("Monotony", f"{report.monotony_score:.0%}")
    metrics[2].metric("Pauses", report.pause_count)
    metrics[3].metric(
        "Pace proxy", "n/a" if report.estimated_speech_rate is None else f"{report.estimated_speech_rate:.1f}/s"
    )
    metrics[4].metric("Mean pitch", "n/a" if report.mean_pitch_hz is None else f"{report.mean_pitch_hz:.0f} Hz")
    charts = st.tabs(["Pitch", "Intensity", "Pauses"])
    with charts[0]:
        if report.pitch_contour:
            st.line_chart(
                [{"time": time, "pitch_hz": value} for time, value in report.pitch_contour], x="time", y="pitch_hz"
            )
        else:
            st.info("No stable speech pitch contour was detected.")
    with charts[1]:
        if report.intensity_contour:
            st.line_chart(
                [{"time": time, "intensity": value} for time, value in report.intensity_contour],
                x="time",
                y="intensity",
            )
    with charts[2]:
        if report.pauses:
            st.dataframe(
                [{"start": start, "end": end, "duration": end - start} for start, end in report.pauses],
                width="stretch",
                hide_index=True,
            )
    if result.reference_comparison:
        st.subheader("Reference comparison")
        st.metric("Prosody similarity", f"{result.reference_comparison.overall_score:.0%}")
        st.caption(result.reference_comparison.explanation)
    if result.transcript:
        st.subheader("Local transcript")
        transcript_metrics = st.columns(5)
        transcript_metrics[0].metric("Language", result.transcript.language or "unknown")
        transcript_metrics[1].metric("Words", result.transcript.word_count)
        transcript_metrics[2].metric(
            "Real WPM",
            "n/a" if result.transcript.words_per_minute is None else f"{result.transcript.words_per_minute:.0f}",
        )
        transcript_metrics[3].metric("Confidence", _percent(result.transcript.confidence))
        transcript_metrics[4].metric("Repetitions", len(result.transcript.repetitions))
        st.write(result.transcript.text or "No words were confidently transcribed.")
        if result.transcript.filler_words is None:
            st.caption("Filler-word rules are currently available for English only.")
        else:
            st.caption(
                "English filler words: "
                + (", ".join(result.transcript.filler_words) if result.transcript.filler_words else "none detected")
            )
        with st.expander("Word timestamps"):
            st.dataframe([item.model_dump() for item in result.transcript.words], width="stretch", hide_index=True)
    if result.word_alignment:
        st.subheader("Word-level reference alignment")
        alignment_metrics = st.columns(5)
        alignment_metrics[0].metric("Match", f"{result.word_alignment.match_score:.0%}")
        alignment_metrics[1].metric("Matched", result.word_alignment.matched_words)
        alignment_metrics[2].metric("Omitted", len(result.word_alignment.omitted_words))
        alignment_metrics[3].metric("Substituted", len(result.word_alignment.substituted_words))
        alignment_metrics[4].metric("Added", len(result.word_alignment.added_words))
        st.caption(result.word_alignment.explanation)
        st.json(result.word_alignment.model_dump(mode="json"), expanded=False)
    if result.ai_coaching_message:
        st.subheader("Local generative coaching")
        st.write(result.ai_coaching_message)
        st.caption(f"Generated by {result.ai_coaching_backend}; measurements and transcript only, never audio.")
    render_quality(result.quality)
    render_metadata(result.metadata)
    feedback_table(result.feedback)


def edit_score_document(document: ScoreDocument) -> ScoreDocument:
    st.subheader("Edit quantized events")
    rows = [event.model_dump() for event in document.events]
    edited = st.data_editor(rows, width="stretch", hide_index=True, num_rows="dynamic")
    try:
        events = [ScoreEvent.model_validate(row) for row in edited]
        return ScoreDocument.model_validate({**document.model_dump(), "events": events})
    except ValueError as exc:
        st.error(f"Score edit is invalid: {exc}")
        return document


def render_score_document(document: ScoreDocument) -> None:
    notes = score_document_to_note_events(document)
    preview = st.tabs(["Staff", "Piano roll", "Data"])
    with preview[0]:
        st.pyplot(get_plot_score()(notes, bpm=document.bpm))
    with preview[1]:
        st.pyplot(get_plot_piano_roll()(notes))
    with preview[2]:
        st.json(document.model_dump(mode="json"), expanded=False)
    downloads = st.columns(4)
    downloads[0].download_button("Download JSON", document.model_dump_json(indent=2), "score.json", "application/json")
    with NamedTemporaryFile(suffix=".csv") as temp:
        csv_path = export_notes_csv(notes, temp.name)
        downloads[1].download_button("Download CSV", csv_path.read_bytes(), "score.csv", "text/csv")
    with NamedTemporaryFile(suffix=".mid") as temp:
        midi_path = export_midi(notes, temp.name, bpm=document.bpm)
        downloads[2].download_button("Download MIDI", midi_path.read_bytes(), "score.mid", "audio/midi")
    with NamedTemporaryFile(suffix=".musicxml") as temp:
        xml_path = export_score_musicxml(document, temp.name)
        downloads[3].download_button(
            "Download MusicXML", xml_path.read_bytes(), "score.musicxml", "application/vnd.recordare.musicxml+xml"
        )


def render_quality(report) -> None:
    st.subheader("Recording quality")
    cols = st.columns(5)
    cols[0].metric("Duration", f"{report.duration:.1f}s")
    cols[1].metric("RMS", f"{report.rms:.4f}")
    cols[2].metric("Peak", f"{report.peak:.3f}")
    cols[3].metric("Clipping", f"{report.clipping_ratio:.2%}")
    cols[4].metric("Voiced", f"{report.voiced_coverage:.0%}")
    for warning in report.warnings:
        st.warning(warning)


def render_metadata(metadata) -> None:
    st.caption(
        f"Requested {metadata.requested_backend} · ran {metadata.actual_backend} · "
        f"completed locally in {metadata.processing_time_ms:.0f} ms"
    )
    if metadata.fallback_reason:
        st.info(metadata.fallback_reason)
    for warning in metadata.warnings:
        st.warning(warning)


def render_pitch_score(score, instrument: str) -> None:
    metrics = st.columns(5)
    metrics[0].metric("View", instrument)
    metrics[1].metric("Target", score.target_note or "nearest note")
    metrics[2].metric("Accuracy", f"{score.accuracy:.0%}")
    metrics[3].metric("Stability", f"{score.stability:.0%}")
    metrics[4].metric("Mean cents", "n/a" if score.mean_abs_cents is None else f"{score.mean_abs_cents:.1f}")


def feedback_table(feedback) -> None:
    if feedback:
        st.subheader("Coaching")
        st.dataframe([item.model_dump() for item in feedback], width="stretch", hide_index=True)


def audio_input(key: str, *, label: str = "Audio"):
    columns = st.columns(2)
    captured = (
        columns[0].audio_input(f"Record {label.lower()}", key=f"{key}-record")
        if hasattr(columns[0], "audio_input")
        else None
    )
    uploaded = columns[1].file_uploader(
        f"Upload {label.lower()}", type=["wav", "flac", "ogg", "aiff", "aif"], key=f"{key}-upload"
    )
    return captured or uploaded


def backend_selector(
    feature: str,
    settings: AISettings,
    baseline: str,
    optional: str,
    *,
    key: str | None = None,
) -> str:
    """Render an explicit backend selector while making the master off-state unambiguous."""

    enabled = settings.feature_enabled(feature)  # type: ignore[arg-type]
    options = [baseline, "auto", optional] if enabled else [baseline]
    selected = st.selectbox(
        "Analysis engine",
        options,
        key=key or f"{feature}-backend",
        help="Auto tries the optional local backend and reports any fallback.",
    )
    if not enabled:
        st.caption("Optional local AI is off for this feature; the deterministic engine will run.")
    return selected


@contextmanager
def temp_audio(uploaded):
    suffix = Path(getattr(uploaded, "name", "audio.wav")).suffix or ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(uploaded.getvalue())
        path = Path(temp.name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def get_repository() -> SessionRepository:
    return _get_repository(os.environ.get("AUDIOTRAINER_DATA_DIR"))


@st.cache_resource
def _get_repository(data_dir: str | None) -> SessionRepository:
    return SessionRepository(data_dir)


@st.cache_resource(show_spinner=False)
def get_plot_pitch_track():
    from audiotrainer.visualization.pitch_plot import plot_pitch_track

    return plot_pitch_track


@st.cache_resource(show_spinner=False)
def get_plot_score():
    from audiotrainer.visualization.score_plot import plot_score

    return plot_score


@st.cache_resource(show_spinner=False)
def get_plot_piano_roll():
    from audiotrainer.visualization.piano_roll import plot_piano_roll

    return plot_piano_roll


def save_once(mode: str, path: Path, uploaded, backend: str, result: dict, feedback) -> None:
    data = uploaded.getvalue()
    fingerprint = hashlib.sha256(
        mode.encode() + data + json.dumps(result, sort_keys=True, default=str).encode()
    ).hexdigest()
    key = f"saved-{fingerprint}"
    if st.session_state.get(key):
        return
    retain = bool(st.session_state.get("retain-next-analysis", False))
    duration = float(result.get("quality", {}).get("duration", result.get("duration", 0.0)))
    if duration <= 0:
        import soundfile as sf

        duration = float(sf.info(path).duration)
    session = get_repository().save(
        mode=mode,
        duration=duration,
        source="browser recording" if "record" in str(getattr(uploaded, "name", "")).lower() else "upload",
        backend=backend,
        settings={},
        result=result,
        feedback=[item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in feedback],
        audio_source=path,
        retain_audio=retain,
    )
    st.session_state[key] = session.session_id
    st.caption(
        "Saved metrics to local practice history"
        + (" with retained audio." if retain else ". Temporary audio was not retained.")
    )


def generate_reference_tones(notes: list[str], *, sr: int = 22_050) -> bytes:
    return _generate_reference_tones(tuple(notes), sr)


@st.cache_data(show_spinner=False, max_entries=64)
def _generate_reference_tones(notes: tuple[str, ...], sr: int) -> bytes:
    import io
    import numpy as np
    import soundfile as sf

    sections = []
    for note in notes:
        frequency = midi_to_frequency(note_name_to_midi(note))
        time = np.arange(int(sr * 0.65), dtype=np.float64) / sr
        envelope = np.minimum(1.0, time / 0.03) * np.minimum(1.0, (0.65 - time) / 0.04)
        sections.extend([0.18 * np.sin(2 * np.pi * frequency * time) * envelope, np.zeros(int(sr * 0.12))])
    signal = np.concatenate(sections) if sections else np.zeros(1)
    output = io.BytesIO()
    sf.write(output, signal, sr, format="WAV")
    return output.getvalue()


def _concert_note(written: str, transposition: int) -> str:
    return _midi_name(note_name_to_midi(written) - transposition)


def _transpose_note(note: str | None, transposition: int) -> str | None:
    return _midi_name(note_name_to_midi(note) + transposition) if note else None


def _midi_name(value: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[value % 12]}{value // 12 - 1}"


def _session_score(result: dict):
    for path in [
        ("overall_accuracy",),
        ("accuracy",),
        ("score", "accuracy"),
        ("reference_comparison", "overall_score"),
        ("estimate", "confidence"),
    ]:
        value = result
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _session_label(session_id: str, sessions) -> str:
    session = next(item for item in sessions if item.session_id == session_id)
    return f"{session.created_at:%Y-%m-%d %H:%M} · {session.mode} · {session.backend}"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ["B", "KB", "MB", "GB"]:
        if amount < 1024 or unit == "GB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} GB"


if __name__ == "__main__":
    main()
