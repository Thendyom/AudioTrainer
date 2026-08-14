# Library Usage

AudioTrainer can be imported as a normal Python package.

## Pitch

```python
from audiotrainer.audio.io import load_audio
from audiotrainer.pitch import detect_pitch, hz_to_note, cents_error

audio, sr = load_audio("voice.wav")
track = detect_pitch(audio, sr)
note = hz_to_note(440.0)
error = cents_error(445.0, "A4")
```

## Transcription

```python
from audiotrainer.api.service import create_score_file
from audiotrainer.transcription import export_midi, export_notes_csv, export_score_musicxml

track, events, score, metadata = create_score_file(
    "melody.wav",
    bpm=120,
    time_signature="4/4",
    quantization=4,
)
export_notes_csv(events, "notes.csv")
export_midi(events, "notes.mid")
export_score_musicxml(score, "notes.musicxml")
```

The score document preserves rests, validates monophonic edits, splits measures,
and marks notes tied across barlines.

## Speech

```python
from audiotrainer.speech import analyze_prosody, compare_reference_speech

report = analyze_prosody(audio, sr)
comparison = compare_reference_speech(user_audio, reference_audio, sr)
```

`compare_reference_speech` is intentionally prosody-level in the baseline release. It does not perform phoneme alignment.

For file-based coaching with quality checks and feedback:

```python
from audiotrainer.api.service import coach_speech_file

result = coach_speech_file(
    "take.wav",
    reference_path="reference.wav",
    goal="presenter presence",
)
```

To request local word-level transcription after explicitly installing and downloading the speech model:

```python
result = coach_speech_file(
    "take.wav",
    reference_path="reference.wav",
    backend="faster-whisper",
    ai_enabled=True,
    language=None,
)
print(result.transcript.text)
print(result.word_alignment)
```

Word alignment reports matched, omitted, substituted, and added words. It does not claim phoneme scoring.

## Voice Profile

```python
from audiotrainer.speech import classify_voice_type, estimate_vocal_range

profile = estimate_vocal_range(track)
estimate = classify_voice_type(profile)
```

Voice estimates include confidence and uncertainty wording.

## Instrument Features

```python
from audiotrainer.instruments import classify_instrument, extract_instrument_features

features = extract_instrument_features(audio, sr)
estimate = classify_instrument(features)
```

The default classifier is rule-based and intended as a small baseline.

Use `analyze_instrument_file(..., backend="ast", ai_enabled=True)` for the optional experimental AST adapter. Low confidence or a small top-two margin returns `unknown`.

## Optional AI settings

```python
from audiotrainer.ml import AISettings, get_ai_settings, save_ai_settings

settings = get_ai_settings()
save_ai_settings(settings.model_copy(update={"enabled": False}))
```

Library service calls still require an explicit `ai_enabled=True`; this keeps each analysis predictable and lets callers enforce a per-request off switch.

## Recording Quality and Exercises

```python
from audiotrainer.api.service import analyze_audio_quality, run_pitch_exercise

quality = analyze_audio_quality("scale.wav")
exercise = run_pitch_exercise("scale.wav", ["C4", "D4", "E4"])
```

## Local Practice History

```python
from audiotrainer.history import SessionRepository

history = SessionRepository()
recent = history.list(mode="pitch", limit=20)
```

Recordings are retained only when `retain_audio=True` is passed while saving a
session. Deleting the session also deletes its managed recording.
