# AudioTrainer

AudioTrainer 0.2 is a local-first library and Streamlit product for music and speech practice. The deterministic engines are always available. Optional local models can be enabled per feature, never download implicitly, and can be disabled globally without removing their weights.

The package is designed so the core library does the work and the CLI, Streamlit UI, and FastAPI service remain thin wrappers.

## Current Features

- Pitch detection with a compact YIN baseline.
- Recording-quality checks, sustained-note and note-sequence exercises, reference tones, and transposing-instrument notation.
- Frequency-to-note conversion and cents error reporting.
- Pitch accuracy and stability scoring.
- Automatic target-note inference with manual override.
- Editable beat-quantized monophonic scores with rests, measures, ties, CSV, JSON, MusicXML, and dependency-free MIDI export.
- Speech prosody analysis: pitch contour, intensity, pause patterns, speaking-rate proxy, monotony, and presenter-focused feedback.
- Optional Faster-Whisper transcription with word timestamps, real WPM, repetition detection, English-only filler rules, multilingual language detection, and ordered reference alignment.
- Reference speech comparison at the prosody and delivery level.
- Vocal range and rough voice type estimation with explicit uncertainty.
- Experimental rule-based instrument classification with optional local AudioSet AST and explicit unknown thresholds.
- Optional pYIN pitch tracking and localhost-only generative speech coaching.
- Local SQLite practice history, trends, export, deletion, and opt-in recording retention.
- Typer CLI, Streamlit app, FastAPI app, examples, and tests.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install ".[dev]"
```

For the UI and plotting extras:

```bash
pip install ".[app]"
```

Optional local ML dependencies are split so installations stay intentional:

```bash
pip install ".[ml-pitch]"
pip install ".[ml-speech]"
pip install ".[ml-instruments]"
# or all three
pip install ".[ml]"
```

Installing an extra does not download model weights. Use the **Models & Privacy** page or `audiotrainer models download speech|instruments` for an explicit download.

## CLI

```bash
audiotrainer --help
audiotrainer pitch path/to/file.wav --target A4
audiotrainer pitch path/to/scale.wav --targets C4,D4,E4
audiotrainer pitch path/to/file.wav --target A4 --save
audiotrainer transcribe path/to/file.wav --musicxml-out score.musicxml --json-out score.json
audiotrainer speech path/to/speech.wav
audiotrainer speech path/to/user.wav --reference path/to/reference.wav
audiotrainer speech path/to/user.wav --backend faster-whisper --ai --language en
audiotrainer voice-profile path/to/scale.wav
audiotrainer instrument path/to/clip.wav
audiotrainer history list
audiotrainer models status
audiotrainer models enable
audiotrainer models disable
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

The Streamlit website exposes Dashboard, Pitch, Score, Speech, Voice, Instruments, and Models & Privacy. Recorded workflows use the browser microphone or file uploads. Metrics and reports are stored locally; recordings remain temporary unless retention is enabled before analysis. The master local-AI switch defaults off. Heavy runtimes are lazy-loaded only after an AI-backed analysis is selected.

By default, history uses `platformdirs.user_data_dir("AudioTrainer")`. Set `AUDIOTRAINER_DATA_DIR` to isolate or relocate local data for testing or portable installations.

## Limitations

- Pitch detection assumes mostly monophonic foreground audio.
- Pronunciation analysis is prosody-level only; it does not claim phoneme-perfect scoring.
- Voice type estimates are rough and probabilistic. A short recording is not enough for confident classification.
- Instrument recognition remains experimental, including with AST.
- Score creation is intentionally monophonic; chords are out of scope.
- Word alignment is not phoneme scoring and makes no phoneme-perfect pronunciation claim.
- FastAPI upload endpoints require the `app` extra because multipart upload support is optional.

## Local model behavior

- `auto` tries an enabled, installed local backend and reports any deterministic fallback.
- An explicitly requested unavailable or disabled backend fails clearly instead of silently changing engines.
- Faster-Whisper and AST read only AudioTrainer-managed local model directories.
- Generative coaching accepts only `localhost`, `127.0.0.1`, or `::1` endpoints and sends metrics/transcript text—not audio.
- Filler-word rules run for English only; other languages omit filler scoring.
