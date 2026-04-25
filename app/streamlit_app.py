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
from audiotrainer.transcription.note_events import export_notes_csv
from audiotrainer.visualization import plot_piano_roll, plot_pitch_track

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
    target = st.text_input("Target note", value="A4")
    if uploaded:
        with temp_audio(uploaded) as path:
            track, score, feedback = analyze_pitch_file(path, target or None)
        st.pyplot(plot_pitch_track(track))
        metric_cols = st.columns(4)
        metric_cols[0].metric("Accuracy", f"{score.accuracy:.0%}")
        metric_cols[1].metric("Stability", f"{score.stability:.0%}")
        metric_cols[2].metric("Mean cents", "n/a" if score.mean_abs_cents is None else f"{score.mean_abs_cents:.1f}")
        metric_cols[3].metric("Voiced frames", str(score.voiced_frame_count))
        feedback_table(feedback)


def note_writer() -> None:
    uploaded = audio_input("notes-audio")
    if uploaded:
        with temp_audio(uploaded) as path:
            _, events = transcribe_file(path)
        st.pyplot(plot_piano_roll(events))
        data = [event.model_dump() for event in events]
        st.dataframe(data, use_container_width=True, hide_index=True)
        with NamedTemporaryFile(delete=False, suffix=".csv") as temp:
            csv_path = export_notes_csv(events, temp.name)
        st.download_button("Download CSV", data=Path(csv_path).read_bytes(), file_name="notes.csv", mime="text/csv")
        Path(csv_path).unlink(missing_ok=True)


def speech_coach() -> None:
    uploaded = audio_input("speech-audio")
    if uploaded:
        with temp_audio(uploaded) as path:
            report, feedback = analyze_speech_file(path)
        st.json(report.model_dump(mode="json"))
        feedback_table(feedback)


def voice_profile() -> None:
    uploaded = audio_input("voice-audio")
    if uploaded:
        with temp_audio(uploaded) as path:
            vocal_range, estimate, feedback = analyze_voice_profile_file(path)
        st.json({"range": vocal_range.model_dump(mode="json"), "estimate": estimate.model_dump(mode="json")})
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
