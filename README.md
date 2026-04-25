# AudioTrainer

AudioTrainer is a lightweight Python library and runnable app for audio coaching. It provides deterministic baselines for pitch detection, note transcription, speech prosody, voice profiling, instrument features, and feedback generation without shipping model weights or requiring large ML frameworks.

The package is designed so the core library does the work and the CLI, Streamlit UI, and FastAPI service remain thin wrappers.

## Current Features

- Pitch detection with a compact YIN baseline.
- Frequency-to-note conversion and cents error reporting.
- Pitch accuracy and stability scoring.
- Automatic target-note inference with manual override.
- Note-event transcription with CSV, MusicXML, and dependency-free MIDI export.
- Speech prosody analysis: pitch contour, intensity, pause patterns, speaking-rate proxy, monotony, and presenter-focused feedback.
- Reference speech comparison at the prosody level.
- Vocal range and rough voice type estimation with explicit uncertainty.
- Rule-based instrument classification from spectral features.
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
audiotrainer transcribe path/to/file.wav --csv-out notes.csv --midi-out notes.mid
audiotrainer speech path/to/speech.wav
audiotrainer speech path/to/user.wav --reference path/to/reference.wav
audiotrainer voice-profile path/to/scale.wav
audiotrainer instrument path/to/clip.wav
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

```bash
streamlit run app/streamlit_app.py
```

FastAPI:

```bash
uvicorn app.fastapi_app:app --reload
```

The app pages cover Pitch Trainer, Note Writer, Speech Coach, Voice Profile, and Instrument Detector.

## Limitations

- Pitch detection assumes mostly monophonic foreground audio.
- Pronunciation analysis is prosody-level only; it does not claim phoneme-perfect scoring.
- Voice type estimates are rough and probabilistic. A short recording is not enough for confident classification.
- Instrument recognition is a rule-based baseline, not a trained classifier.
- FastAPI upload endpoints require the `app` extra because multipart upload support is optional.

## Roadmap

- Optional advanced pitch backends behind adapter interfaces.
- Optional phoneme alignment and ASR adapters.
- Optional scikit-learn instrument classifier.
- More robust live audio workflows.
- Richer coaching-session persistence and progress tracking.
