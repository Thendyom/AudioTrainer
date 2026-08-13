"""Local-first Streamlit product for AudioTrainer v0.2."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
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
    analyze_pitch_file,
    analyze_voice_profile_details,
    coach_speech_file,
    create_score_file,
    run_pitch_exercise,
)
from audiotrainer.audio.io import load_audio
from audiotrainer.audio.quality import analyze_audio_quality as analyze_quality_signal
from audiotrainer.history import SessionRepository
from audiotrainer.pitch.notes import midi_to_frequency, note_name_to_midi
from audiotrainer.transcription import (
    export_midi,
    export_notes_csv,
    export_score_musicxml,
    score_document_to_note_events,
)

st.set_page_config(page_title="AudioTrainer", page_icon="🎧", layout="wide")

PAGES = ["Dashboard", "Pitch", "Score", "Speech", "Voice", "Instruments", "Privacy"]
INSTRUMENT_TRANSPOSITIONS = {
    "Concert pitch": 0,
    "Tenor/soprano saxophone, trumpet, clarinet (Bb)": 2,
    "Alto/baritone saxophone (Eb)": -3,
    "French horn (F)": 7,
}


def main() -> None:
    st.title("AudioTrainer")
    st.caption("Fast, private audio practice with deterministic local analysis")
    page = st.sidebar.radio("Workspace", PAGES)
    st.sidebar.toggle(
        "Retain the next analyzed recording",
        value=bool(get_repository().get_setting("retain_audio_default", False)),
        key="retain-next-analysis",
        help="Off keeps metrics and reports but deletes the temporary recording after analysis.",
    )
    st.sidebar.caption("v0.2.0 · Audio is never sent to a cloud service")
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
        privacy_page()


def dashboard_page() -> None:
    st.header("Practice dashboard")
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
    selected = st.selectbox("Session details", [session.session_id for session in sessions], format_func=lambda value: _session_label(value, sessions))
    detail = repository.get(selected)
    if detail:
        st.json(detail.model_dump(mode="json"), expanded=False)
        if st.button("Delete selected session"):
            repository.delete(selected)
            st.rerun()
    export_cols = st.columns(3)
    export_cols[0].download_button("Download history JSON", repository.export_json(), "audiotrainer-history.json", "application/json")
    with NamedTemporaryFile(suffix=".csv") as temp:
        csv_path = repository.export_csv(temp.name)
        export_cols[1].download_button("Download history CSV", csv_path.read_bytes(), "audiotrainer-history.csv", "text/csv")
    confirm = export_cols[2].checkbox("Confirm clear all", key="clear-history-confirm")
    if export_cols[2].button("Clear all history", disabled=not confirm):
        repository.clear()
        st.rerun()


def pitch_page() -> None:
    st.header("Pitch trainer")
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
                    track, score, feedback = analyze_pitch_file(path, concert_target)
                    payload = {
                        "track": track,
                        "score": score,
                        "feedback": feedback,
                        "instrument": instrument,
                    }
                    st.session_state["pitch-free-result"] = payload
                    save_once("pitch", path, uploaded, "yin", score.model_dump(mode="json"), feedback)
                except ValueError as exc:
                    st.error(str(exc))
        payload = st.session_state.get("pitch-free-result")
        if payload:
            st.pyplot(get_plot_pitch_track()(payload["track"]), clear_figure=True)
            render_pitch_score(payload["score"], payload["instrument"])
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
                result = run_pitch_exercise(path, concert_targets)
                st.session_state["pitch-exercise-result"] = (result, targets)
                save_once("pitch", path, uploaded, result.metadata.actual_backend, result.model_dump(mode="json"), result.feedback)
            except ValueError as exc:
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
    st.header("Monophonic score creator")
    st.caption("Creates one melody line. Chords and polyphonic audio are intentionally not claimed.")
    settings = st.columns(3)
    bpm = settings[0].number_input("Tempo (BPM)", 30, 300, 120)
    signature = settings[1].selectbox("Time signature", ["4/4", "3/4", "6/8"])
    quantization_label = settings[2].selectbox("Smallest grid", ["Quarter", "Eighth", "Sixteenth", "Thirty-second"], index=2)
    quantization = {"Quarter": 1, "Eighth": 2, "Sixteenth": 4, "Thirty-second": 8}[quantization_label]
    uploaded = audio_input("score")
    if uploaded and st.button("Create score", type="primary"):
        with temp_audio(uploaded) as path, st.spinner("Detecting and quantizing notes…"):
            try:
                track, _, document, metadata = create_score_file(
                    path, bpm=int(bpm), time_signature=signature, quantization=quantization
                )
                audio, sr = load_audio(path)
                quality = analyze_quality_signal(audio, sr, pitch_track=track)
                st.session_state["score-analysis"] = {
                    "document": document,
                    "metadata": metadata,
                    "quality": quality,
                }
                save_once("score", path, uploaded, metadata.actual_backend, document.model_dump(mode="json"), [])
            except ValueError as exc:
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
    st.header("Speech coach")
    st.caption("Pitch, intensity, pauses, pace, and reference delivery comparison. This is not phoneme scoring.")
    goal = st.selectbox(
        "Speaking goal",
        ["balanced", "clear pronunciation", "presenter presence", "charismatic delivery"],
    )
    uploaded = audio_input("speech-user", label="Your recording")
    reference = audio_input("speech-reference", label="Optional reference")
    if uploaded and st.button("Analyze speech", type="primary"):
        with temp_audio(uploaded) as path, st.spinner("Analyzing delivery…"):
            if reference:
                with temp_audio(reference) as reference_path:
                    analyze_speech_result(path, uploaded, reference_path, goal)
            else:
                analyze_speech_result(path, uploaded, None, goal)
    speech_result = st.session_state.get("speech-result")
    if speech_result:
        render_speech_result(speech_result)


def voice_page() -> None:
    st.header("Voice profile")
    st.info("Record a slow comfortable scale or siren from low to high. Stop before straining. This is not a medical or definitive voice classification.")
    render_microphone_check("voice-quality")
    uploaded = audio_input("voice-profile")
    if uploaded and st.button("Create voice profile", type="primary"):
        with temp_audio(uploaded) as path, st.spinner("Building voice profile…"):
            try:
                track, vocal_range, estimate, feedback, quality, metadata = analyze_voice_profile_details(path)
                st.session_state["voice-result"] = (track, vocal_range, estimate, feedback, quality, metadata)
                result = {"range": vocal_range.model_dump(mode="json"), "estimate": estimate.model_dump(mode="json")}
                save_once("voice", path, uploaded, "yin", result, feedback)
            except ValueError as exc:
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
    st.header("Instrument lab")
    st.warning("Experimental: this is a broad rule-based estimate, not definitive instrument identification.")
    st.caption("The local classifier uses spectral and harmonic features and may return unknown.")
    uploaded = audio_input("instrument")
    if uploaded and st.button("Analyze instrument", type="primary"):
        with temp_audio(uploaded) as path, st.spinner("Classifying the clip…"):
            try:
                result = analyze_instrument_file(path)
                st.session_state["instrument-result"] = result
                save_once("instrument", path, uploaded, result.metadata.actual_backend, result.model_dump(mode="json"), [])
            except ValueError as exc:
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


def privacy_page() -> None:
    st.header("Privacy & data")
    st.success("All analysis on this website runs locally. No cloud APIs or AI models are used.")
    st.write("Recordings are temporary by default. Retaining audio is opt-in for each analysis; metrics and reports are saved locally in SQLite.")
    retain_default = get_repository().get_setting("retain_audio_default", False)
    updated = st.toggle("Default to retaining analyzed audio", value=bool(retain_default))
    if updated != retain_default:
        get_repository().set_setting("retain_audio_default", updated)
    st.subheader("What is stored")
    st.write("Completed metrics and coaching reports are saved in a local SQLite database so the dashboard survives restarts.")
    st.write("Recordings are deleted after analysis unless you explicitly enable retention before analyzing them.")
    st.subheader("What is not used")
    st.write("No accounts, telemetry, cloud APIs, model downloads, speech-to-text models, or remote storage.")


def render_live_pitch(instrument: str, transposition: int) -> None:
    st.subheader("Quick tuner")
    st.info("Record or upload a short tone below, then choose Analyze recording. This reliable workflow avoids extra streaming components.")


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
) -> None:
    """Run and persist speech analysis; rendering happens outside the action block."""

    try:
        result = coach_speech_file(
            path,
            reference_path=reference_path,
            goal=goal,
        )
        st.session_state["speech-result"] = result
        save_once("speech", path, uploaded, result.metadata.actual_backend, result.model_dump(mode="json"), result.feedback)
    except ValueError as exc:
        st.error(str(exc))


def render_speech_result(result) -> None:
    report = result.prosody
    metrics = st.columns(5)
    metrics[0].metric("Pitch range", "n/a" if report.pitch_range_semitones is None else f"{report.pitch_range_semitones:.1f} st")
    metrics[1].metric("Monotony", f"{report.monotony_score:.0%}")
    metrics[2].metric("Pauses", report.pause_count)
    metrics[3].metric("Pace proxy", "n/a" if report.estimated_speech_rate is None else f"{report.estimated_speech_rate:.1f}/s")
    metrics[4].metric("Mean pitch", "n/a" if report.mean_pitch_hz is None else f"{report.mean_pitch_hz:.0f} Hz")
    charts = st.tabs(["Pitch", "Intensity", "Pauses"])
    with charts[0]:
        if report.pitch_contour:
            st.line_chart([{"time": time, "pitch_hz": value} for time, value in report.pitch_contour], x="time", y="pitch_hz")
        else:
            st.info("No stable speech pitch contour was detected.")
    with charts[1]:
        if report.intensity_contour:
            st.line_chart([{"time": time, "intensity": value} for time, value in report.intensity_contour], x="time", y="intensity")
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
        downloads[3].download_button("Download MusicXML", xml_path.read_bytes(), "score.musicxml", "application/vnd.recordare.musicxml+xml")


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
    st.caption(f"Local analysis completed in {metadata.processing_time_ms:.0f} ms")
    if metadata.fallback_reason:
        st.info(metadata.fallback_reason)


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
    captured = columns[0].audio_input(f"Record {label.lower()}", key=f"{key}-record") if hasattr(columns[0], "audio_input") else None
    uploaded = columns[1].file_uploader(f"Upload {label.lower()}", type=["wav", "flac", "ogg", "aiff", "aif"], key=f"{key}-upload")
    return captured or uploaded


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


@st.cache_resource
def get_repository() -> SessionRepository:
    return SessionRepository()


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
    fingerprint = hashlib.sha256(mode.encode() + data + json.dumps(result, sort_keys=True, default=str).encode()).hexdigest()
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
    st.caption("Saved metrics to local practice history" + (" with retained audio." if retain else ". Temporary audio was not retained."))


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
    for path in [("overall_accuracy",), ("accuracy",), ("score", "accuracy"), ("reference_comparison", "overall_score"), ("estimate", "confidence")]:
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


if __name__ == "__main__":
    main()
