# AudioTrainer

AudioTrainer 0.2 is a local-first library and Streamlit product for music and speech practice. Every analysis is deterministic, runs on the local machine, and requires no account, cloud API, or model download.

The package is designed so the core library does the work and the CLI, Streamlit UI, and FastAPI service remain thin wrappers.

## Current Features

- Pitch detection with a compact YIN baseline.
- Recording-quality checks, sustained-note and note-sequence exercises, reference tones, and transposing-instrument notation.
- Frequency-to-note conversion and cents error reporting.
- Pitch accuracy and stability scoring.
- Automatic target-note inference with manual override.
- Editable beat-quantized monophonic scores with rests, measures, ties, CSV, JSON, MusicXML, and dependency-free MIDI export.
- Speech prosody analysis: pitch contour, intensity, pause patterns, speaking-rate proxy, monotony, and presenter-focused feedback.
- Reference speech comparison at the prosody and delivery level.
- Vocal range and rough voice type estimation with explicit uncertainty.
- Experimental rule-based instrument classification with explicit uncertainty.
- Local SQLite practice history, trends, export, deletion, and opt-in recording retention.
- Typer CLI, Streamlit app, FastAPI app, examples, and tests.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

For the UI and plotting extras:

```bash
pip install -e ".[app]"
```

## CLI

```bash
audiotrainer --help
audiotrainer pitch path/to/file.wav --target A4
audiotrainer pitch path/to/scale.wav --targets C4,D4,E4
audiotrainer pitch path/to/file.wav --target A4 --save
audiotrainer transcribe path/to/file.wav --musicxml-out score.musicxml --json-out score.json
audiotrainer speech path/to/speech.wav
audiotrainer speech path/to/user.wav --reference path/to/reference.wav
audiotrainer voice-profile path/to/scale.wav
audiotrainer instrument path/to/clip.wav
audiotrainer history list
audiotrainer app
```

## Library Usage

```python
from audiotrainer.audio.io import load_audio
from audiotrainer.pitch import detect_pitch
from audiotrainer.transcription import pitch_track_to_notes
from audiotrainer.speech import analyze_prosody

audio, sr = load_audio("take.wav")
track = detect_pitch(audio, sr)
events = pitch_track_to_notes(track)
speech = analyze_prosody(audio, sr)
```

Pitch scoring:

```python
from audiotrainer.coaching import generate_pitch_feedback, score_pitch_accuracy

score = score_pitch_accuracy(track, "A4")
feedback = generate_pitch_feedback(score)
```

Voice profile:

```python
from audiotrainer.speech import classify_voice_type, estimate_vocal_range

profile = estimate_vocal_range(track)
estimate = classify_voice_type(profile)
```

## App

Streamlit:

From the repository root:

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py
```

FastAPI:

```bash
.venv/bin/python -m uvicorn app.fastapi_app:app --reload
```

The Streamlit website exposes the deterministic offline product only: Dashboard, Pitch, Score, Speech, Voice, Instruments, and Privacy. Recorded workflows use the browser microphone or file uploads. Metrics and reports are stored locally; recordings remain temporary unless retention is enabled before analysis.

By default, history uses `platformdirs.user_data_dir("AudioTrainer")`. Set `AUDIOTRAINER_DATA_DIR` to isolate or relocate local data for testing or portable installations.

## Limitations

- Pitch detection assumes mostly monophonic foreground audio.
- Pronunciation analysis is prosody-level only; it does not claim phoneme-perfect scoring.
- Voice type estimates are rough and probabilistic. A short recording is not enough for confident classification.
- Instrument recognition is a rule-based baseline, not a trained classifier.
- Score creation is intentionally monophonic; chords are out of scope.
- Speech comparison does not transcribe words or claim phoneme-perfect scoring.
- FastAPI upload endpoints require the `app` extra because multipart upload support is optional.

## Next

- Longer-lived live-audio calibration profiles across different microphones.
- Better deterministic instrument discrimination and calibration.
- Richer language-neutral timing and delivery coaching.
- Richer exercise scheduling and goal-based practice plans.
