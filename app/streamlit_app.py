"""Streamlit application for AudioTrainer."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from audiotrainer.api.service import (
    analyze_pitch_file,
    analyze_speech_file,
    analyze_voice_profile_file,
    classify_instrument_file,
    transcribe_file,
)
from audiotrainer.coaching.feedback import generate_pitch_feedback
from audiotrainer.coaching.scoring import score_pitch_accuracy
from audiotrainer.transcription.score_export import export_musicxml, note_events_to_score_text
from audiotrainer.transcription.note_events import export_notes_csv
from audiotrainer.visualization import plot_piano_roll, plot_pitch_track, plot_score

st.set_page_config(page_title="AudioTrainer", layout="wide")
st.title("AudioTrainer")


def main() -> None:
    tabs = st.tabs(["Pitch Trainer", "Note Writer", "Speech Coach", "Voice Profile", "Instrument Detector"])
    with tabs[0]:
        pitch_trainer()
    with tabs[1]:
        note_writer()
    with tabs[2]:
        speech_coach()
    with tabs[3]:
        voice_profile()
    with tabs[4]:
        instrument_detector()


def pitch_trainer() -> None:
    uploaded = audio_input("pitch-audio")
    if uploaded:
        with temp_audio(uploaded) as path:
            track, auto_score, _ = analyze_pitch_file(path, None)
        auto_target = auto_score.target_note or ""
        target = st.text_input("Target note", value=auto_target, key=f"pitch-target-{uploaded.name}-{auto_target}")
        score = score_pitch_accuracy(track, target or auto_target or None)
        feedback = generate_pitch_feedback(score)
        st.pyplot(plot_pitch_track(track))
        metric_cols = st.columns(5)
        metric_cols[0].metric("Target", score.target_note or "n/a")
        metric_cols[1].metric("Auto target", auto_target or "n/a")
        metric_cols[2].metric("Accuracy", f"{score.accuracy:.0%}")
        metric_cols[3].metric("Stability", f"{score.stability:.0%}")
        metric_cols[4].metric("Mean cents", "n/a" if score.mean_abs_cents is None else f"{score.mean_abs_cents:.1f}")
        feedback_table(feedback)


def note_writer() -> None:
    uploaded = audio_input("notes-audio")
    if uploaded:
        with temp_audio(uploaded) as path:
            _, events = transcribe_file(path)
        st.code(note_events_to_score_text(events), language="text")
        st.pyplot(plot_score(events))
        st.pyplot(plot_piano_roll(events))
        data = [event.model_dump() for event in events]
        st.dataframe(data, use_container_width=True, hide_index=True)
        with NamedTemporaryFile(delete=False, suffix=".csv") as temp:
            csv_path = export_notes_csv(events, temp.name)
        st.download_button("Download CSV", data=Path(csv_path).read_bytes(), file_name="notes.csv", mime="text/csv")
        Path(csv_path).unlink(missing_ok=True)
        with NamedTemporaryFile(delete=False, suffix=".musicxml") as temp:
            musicxml_path = export_musicxml(events, temp.name)
        st.download_button(
            "Download MusicXML",
            data=Path(musicxml_path).read_bytes(),
            file_name="score.musicxml",
            mime="application/vnd.recordare.musicxml+xml",
        )
        Path(musicxml_path).unlink(missing_ok=True)


def speech_coach() -> None:
    uploaded = audio_input("speech-audio")
    goal = st.selectbox("Speaking goal", ["balanced", "clear pronunciation", "presenter presence", "charismatic delivery"])
    if uploaded:
        with temp_audio(uploaded) as path:
            report, feedback = analyze_speech_file(path, goal=goal)
        cols = st.columns(5)
        cols[0].metric("Pitch range", "n/a" if report.pitch_range_semitones is None else f"{report.pitch_range_semitones:.1f} st")
        cols[1].metric("Monotony", f"{report.monotony_score:.0%}")
        cols[2].metric("Pauses", str(report.pause_count))
        cols[3].metric("Speech-rate proxy", "n/a" if report.estimated_speech_rate is None else f"{report.estimated_speech_rate:.1f}/s")
        cols[4].metric("Mean pitch", "n/a" if report.mean_pitch_hz is None else f"{report.mean_pitch_hz:.0f} Hz")
        feedback_table(feedback)


def voice_profile() -> None:
    uploaded = audio_input("voice-audio")
    if uploaded:
        with temp_audio(uploaded) as path:
            vocal_range, estimate, feedback = analyze_voice_profile_file(path)
        cols = st.columns(5)
        cols[0].metric("Lowest", vocal_range.lowest_note or "n/a")
        cols[1].metric("Highest", vocal_range.highest_note or "n/a")
        cols[2].metric("Stable span", f"{vocal_range.stable_range_semitones:.1f} st")
        cols[3].metric("Likely type", estimate.primary_type)
        cols[4].metric("Confidence", f"{estimate.confidence:.0%}")
        st.caption(estimate.explanation)
        feedback_table(feedback)


def instrument_detector() -> None:
    uploaded = audio_input("instrument-audio")
    if uploaded:
        with temp_audio(uploaded) as path:
            estimate = classify_instrument_file(path)
        cols = st.columns(2)
        cols[0].metric("Instrument", estimate.label)
        cols[1].metric("Confidence", f"{estimate.confidence:.0%}")
        st.caption(estimate.explanation)


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
