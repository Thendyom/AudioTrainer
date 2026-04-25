"""Streamlit application for AudioTrainer."""

from __future__ import annotations

from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile
import time

import streamlit as st

from audiotrainer.audio.device import record_audio
from audiotrainer.api.service import (
    analyze_pitch_file,
    analyze_speech_file,
    analyze_voice_profile_file,
    transcribe_file,
)
from audiotrainer.coaching.feedback import generate_pitch_feedback, generate_speech_feedback, generate_voice_feedback
from audiotrainer.coaching.scoring import score_pitch_accuracy
from audiotrainer.pitch import cents_error, detect_pitch, frequency_to_midi, hz_to_note
from audiotrainer.pitch.notes import midi_to_note_name, note_name_to_midi
from audiotrainer.speech import analyze_prosody, classify_voice_type, estimate_vocal_range
from audiotrainer.transcription.score_export import export_musicxml, note_events_to_score_text
from audiotrainer.transcription.note_events import export_notes_csv
from audiotrainer.transcription import pitch_track_to_notes
from audiotrainer.visualization import plot_piano_roll, plot_pitch_track, plot_score

st.set_page_config(page_title="AudioTrainer", layout="wide")
st.title("AudioTrainer")

INSTRUMENT_TRANSPOSITIONS = {
    "Concert pitch": 0,
    "Tenor saxophone (Bb)": 2,
    "Soprano saxophone (Bb)": 2,
    "Trumpet (Bb)": 2,
    "Clarinet (Bb)": 2,
    "Alto saxophone (Eb)": -3,
    "Baritone saxophone (Eb)": -3,
    "French horn (F)": 7,
}


def main() -> None:
    tabs = st.tabs(["Pitch Trainer", "Score Creator", "Speech Coach", "Voice Profile"])
    with tabs[0]:
        pitch_trainer()
    with tabs[1]:
        note_writer()
    with tabs[2]:
        speech_coach()
    with tabs[3]:
        voice_profile()


def pitch_trainer() -> None:
    instrument_mode = st.toggle("Instrument written-note mode", value=False)
    instrument_name = "Concert pitch"
    transposition = 0
    if instrument_mode:
        instrument_name = st.selectbox("Instrument", list(INSTRUMENT_TRANSPOSITIONS), index=1)
        transposition = INSTRUMENT_TRANSPOSITIONS[instrument_name]
    live_pitch_monitor(instrument_name=instrument_name, transposition=transposition)
    st.divider()
    uploaded = audio_input("pitch-audio")
    if uploaded:
        with temp_audio(uploaded) as path:
            track, auto_score, _ = analyze_pitch_file(path, None)
        auto_target = transpose_note_name(auto_score.target_note, transposition) if auto_score.target_note else ""
        target = st.text_input(
            "Target note",
            value=auto_target,
            key=f"pitch-target-{uploaded.name}-{auto_target}-{instrument_name}",
        )
        target_for_scoring = written_target_to_concert(target or auto_target or None, transposition)
        score = score_pitch_accuracy(track, target_for_scoring)
        feedback = generate_pitch_feedback(score)
        st.pyplot(plot_pitch_track(track))
        metric_cols = st.columns(6)
        metric_cols[0].metric("View", instrument_name)
        metric_cols[1].metric("Target", target or auto_target or "n/a")
        metric_cols[2].metric("Auto target", auto_target or "n/a")
        metric_cols[3].metric("Accuracy", f"{score.accuracy:.0%}")
        metric_cols[4].metric("Stability", f"{score.stability:.0%}")
        metric_cols[5].metric("Mean cents", "n/a" if score.mean_abs_cents is None else f"{score.mean_abs_cents:.1f}")
        feedback_table(feedback)


def live_pitch_monitor(*, instrument_name: str, transposition: int) -> None:
    st.subheader("Live Pitch Monitor")
    controls = st.columns(4)
    target_label = "Target note" if transposition == 0 else "Written target note"
    target = controls[0].text_input(target_label, value="A4", key=f"live-target-note-{instrument_name}")
    chunk_seconds = controls[1].slider("Window", min_value=0.15, max_value=1.0, value=0.35, step=0.05)
    monitor_seconds = controls[2].slider("Duration", min_value=3, max_value=60, value=15, step=1)
    sample_rate = controls[3].selectbox("Sample rate", [16_000, 22_050, 44_100], index=1)

    metric_slot = st.empty()
    chart_slot = st.empty()
    status_slot = st.empty()
    start_col, stop_col = st.columns([1, 1])
    if start_col.button("Start live monitor", type="primary"):
        st.session_state.live_pitch_running = True
        st.session_state.live_pitch_history = []
        st.session_state.live_pitch_started_at = time.monotonic()
    if stop_col.button("Stop"):
        st.session_state.live_pitch_running = False

    if not st.session_state.get("live_pitch_running", False):
        return

    try:
        audio = record_audio(chunk_seconds, sample_rate)
        track = detect_pitch(
            audio,
            sample_rate,
            frame_length=1024 if sample_rate <= 22_050 else 2048,
            hop_length=256,
            fmin=55.0,
            fmax=1_100.0,
            smooth=False,
        )
    except RuntimeError as exc:
        st.session_state.live_pitch_running = False
        status_slot.error(str(exc))
        return

    current = current_pitch(track, transposition=transposition)
    elapsed = time.monotonic() - st.session_state.live_pitch_started_at
    if current is None:
        with metric_slot.container():
            cols = st.columns(5)
            cols[0].metric("Frequency", "unvoiced")
            cols[1].metric("Closest note", "n/a")
            cols[2].metric("View", instrument_name)
            cols[3].metric("Cents off", "n/a")
            cols[4].metric("Confidence", "0%")
        status_slot.caption("No stable pitch detected in the current window.")
    else:
        frequency, concert_note, display_note, confidence = current
        selected_target = target.strip() or display_note
        concert_target = written_target_to_concert(selected_target, transposition)
        try:
            cents = cents_error(frequency, concert_target)
        except ValueError:
            cents = cents_error(frequency)
            status_slot.warning("Target note is invalid; showing cents to nearest detected note.")
        history = st.session_state.live_pitch_history
        history.append({"time": elapsed, "cents": cents})
        st.session_state.live_pitch_history = history[-120:]
        with metric_slot.container():
            cols = st.columns(6)
            cols[0].metric("Frequency", f"{frequency:.1f} Hz")
            cols[1].metric("Concert note", concert_note)
            cols[2].metric("Written note", display_note)
            cols[3].metric("Target", selected_target)
            cols[4].metric("Cents off", cents_status(cents), delta=cents_direction(cents))
            cols[5].metric("Confidence", f"{confidence:.0%}")
        chart_slot.line_chart(st.session_state.live_pitch_history, x="time", y="cents")
        status_slot.caption(f"{elapsed:.1f}s / {monitor_seconds:.1f}s")

    if elapsed >= monitor_seconds:
        st.session_state.live_pitch_running = False
        status_slot.caption("Live monitor stopped.")
        return
    time.sleep(0.05)
    st.rerun()


def current_pitch(track, *, transposition: int = 0):
    voiced = [frame for frame in track.frames if frame.frequency_hz is not None and frame.confidence >= 0.2]
    if not voiced:
        return None
    recent = voiced[-min(6, len(voiced)) :]
    frequency = float(median(frame.frequency_hz for frame in recent if frame.frequency_hz is not None))
    confidence = sum(frame.confidence for frame in recent) / len(recent)
    concert_note = hz_to_note(frequency).name
    display_note = transpose_frequency_to_note(frequency, transposition)
    return frequency, concert_note, display_note, confidence


def cents_status(cents: float) -> str:
    if abs(cents) < 3.0:
        return "on"
    return f"{abs(cents):.1f}"


def cents_direction(cents: float) -> str:
    if abs(cents) < 3.0:
        return "in tune"
    return "sharp" if cents > 0 else "flat"


def note_writer() -> None:
    live_score_creator()
    st.divider()
    uploaded = audio_input("notes-audio")
    if uploaded:
        with temp_audio(uploaded) as path:
            _, events = transcribe_file(path)
        render_score_creator(events)


def live_score_creator() -> None:
    st.subheader("Live Score Capture")
    controls = st.columns(3)
    duration = controls[0].slider("Capture length", min_value=2, max_value=30, value=8, step=1, key="score-live-duration")
    sample_rate = controls[1].selectbox("Sample rate", [16_000, 22_050, 44_100], index=1, key="score-live-sr")
    if controls[2].button("Record score sample", type="primary"):
        audio = record_audio(duration, sample_rate)
        track = detect_pitch(audio, sample_rate)
        st.session_state.score_live_events = pitch_track_to_notes(track)
    if st.session_state.get("score_live_events"):
        render_score_creator(st.session_state.score_live_events, prefix="live")


def render_score_creator(events, *, prefix: str = "upload") -> None:
    st.code(note_events_to_score_text(events), language="text")
    st.pyplot(plot_score(events))
    st.pyplot(plot_piano_roll(events))
    data = [event.model_dump() for event in events]
    st.dataframe(data, use_container_width=True, hide_index=True)
    with NamedTemporaryFile(delete=False, suffix=".csv") as temp:
        csv_path = export_notes_csv(events, temp.name)
    st.download_button(
        "Download CSV",
        data=Path(csv_path).read_bytes(),
        file_name="notes.csv",
        mime="text/csv",
        key=f"{prefix}-csv-download",
    )
    Path(csv_path).unlink(missing_ok=True)
    with NamedTemporaryFile(delete=False, suffix=".musicxml") as temp:
        musicxml_path = export_musicxml(events, temp.name)
    st.download_button(
        "Download MusicXML",
        data=Path(musicxml_path).read_bytes(),
        file_name="score.musicxml",
        mime="application/vnd.recordare.musicxml+xml",
        key=f"{prefix}-musicxml-download",
    )
    Path(musicxml_path).unlink(missing_ok=True)


def speech_coach() -> None:
    live_speech_coach()
    st.divider()
    uploaded = audio_input("speech-audio")
    goal = st.selectbox("Speaking goal", ["balanced", "clear pronunciation", "presenter presence", "charismatic delivery"])
    if uploaded:
        with temp_audio(uploaded) as path:
            report, feedback = analyze_speech_file(path, goal=goal)
        render_speech_report(report, feedback)


def live_speech_coach() -> None:
    st.subheader("Live Speech Capture")
    controls = st.columns(4)
    goal = controls[0].selectbox(
        "Live speaking goal",
        ["balanced", "clear pronunciation", "presenter presence", "charismatic delivery"],
        key="speech-live-goal",
    )
    duration = controls[1].slider("Capture length", min_value=3, max_value=60, value=12, step=1, key="speech-live-duration")
    sample_rate = controls[2].selectbox("Sample rate", [16_000, 22_050, 44_100], index=1, key="speech-live-sr")
    if controls[3].button("Record speech sample", type="primary"):
        audio = record_audio(duration, sample_rate)
        report = analyze_prosody(audio, sample_rate)
        feedback = generate_speech_feedback(report, goal=goal)
        st.session_state.speech_live_result = (report, feedback)
    if st.session_state.get("speech_live_result"):
        report, feedback = st.session_state.speech_live_result
        render_speech_report(report, feedback)


def render_speech_report(report, feedback) -> None:
    cols = st.columns(5)
    cols[0].metric("Pitch range", "n/a" if report.pitch_range_semitones is None else f"{report.pitch_range_semitones:.1f} st")
    cols[1].metric("Monotony", f"{report.monotony_score:.0%}")
    cols[2].metric("Pauses", str(report.pause_count))
    cols[3].metric("Speech-rate proxy", "n/a" if report.estimated_speech_rate is None else f"{report.estimated_speech_rate:.1f}/s")
    cols[4].metric("Mean pitch", "n/a" if report.mean_pitch_hz is None else f"{report.mean_pitch_hz:.0f} Hz")
    feedback_table(feedback)


def voice_profile() -> None:
    live_voice_profile()
    st.divider()
    uploaded = audio_input("voice-audio")
    if uploaded:
        with temp_audio(uploaded) as path:
            vocal_range, estimate, feedback = analyze_voice_profile_file(path)
        render_voice_profile(vocal_range, estimate, feedback)


def live_voice_profile() -> None:
    st.subheader("Live Voice Profile Capture")
    controls = st.columns(3)
    duration = controls[0].slider("Capture length", min_value=3, max_value=45, value=10, step=1, key="voice-live-duration")
    sample_rate = controls[1].selectbox("Sample rate", [16_000, 22_050, 44_100], index=1, key="voice-live-sr")
    if controls[2].button("Record voice sample", type="primary"):
        audio = record_audio(duration, sample_rate)
        track = detect_pitch(audio, sample_rate, fmin=55.0, fmax=1_100.0)
        vocal_range = estimate_vocal_range(track)
        prosody = analyze_prosody(audio, sample_rate)
        estimate = classify_voice_type(vocal_range, speaking_pitch=prosody.mean_pitch_hz)
        feedback = []
        feedback.extend(generate_voice_feedback(vocal_range))
        feedback.extend(generate_voice_feedback(estimate))
        st.session_state.voice_live_result = (vocal_range, estimate, feedback)
    if st.session_state.get("voice_live_result"):
        vocal_range, estimate, feedback = st.session_state.voice_live_result
        render_voice_profile(vocal_range, estimate, feedback)


def render_voice_profile(vocal_range, estimate, feedback) -> None:
    cols = st.columns(5)
    cols[0].metric("Lowest", vocal_range.lowest_note or "n/a")
    cols[1].metric("Highest", vocal_range.highest_note or "n/a")
    cols[2].metric("Stable span", f"{vocal_range.stable_range_semitones:.1f} st")
    cols[3].metric("Likely type", estimate.primary_type)
    cols[4].metric("Confidence", f"{estimate.confidence:.0%}")
    st.caption(estimate.explanation)
    feedback_table(feedback)


def transpose_frequency_to_note(frequency_hz: float, semitone_offset: int) -> str:
    midi = int(round(frequency_to_midi(frequency_hz))) + semitone_offset
    return midi_to_note_name(midi)


def transpose_note_name(note_name: str | None, semitone_offset: int) -> str | None:
    if note_name is None:
        return None
    return midi_to_note_name(note_name_to_midi(note_name) + semitone_offset)


def written_target_to_concert(note_name: str | None, semitone_offset: int) -> str | None:
    if not note_name:
        return None
    try:
        return midi_to_note_name(note_name_to_midi(note_name) - semitone_offset)
    except ValueError:
        return None


def audio_input(key: str):
    captured = None
    if hasattr(st, "audio_input"):
        captured = st.audio_input("Record audio", key=f"{key}-record")
    uploaded = st.file_uploader("Upload audio", type=["wav", "flac", "ogg", "aiff", "aif"], key=f"{key}-upload")
    return captured or uploaded


class temp_audio:
    def __init__(self, uploaded) -> None:
        self.uploaded = uploaded
        self.path: Path | None = None

    def __enter__(self) -> Path:
        suffix = Path(getattr(self.uploaded, "name", "audio.wav")).suffix or ".wav"
        data = self.uploaded.getvalue()
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(data)
            self.path = Path(temp.name)
        return self.path

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.path:
            self.path.unlink(missing_ok=True)


def feedback_table(feedback) -> None:
    rows = [item.model_dump() for item in feedback]
    st.dataframe(rows, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
