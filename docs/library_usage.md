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
from audiotrainer.transcription import export_midi, export_notes_csv, pitch_track_to_notes

events = pitch_track_to_notes(track)
export_notes_csv(events, "notes.csv")
export_midi(events, "notes.mid")
```

## Speech

```python
from audiotrainer.speech import analyze_prosody, compare_reference_speech

report = analyze_prosody(audio, sr)
comparison = compare_reference_speech(user_audio, reference_audio, sr)
```

`compare_reference_speech` is intentionally prosody-level in the baseline release. It does not perform phoneme alignment.

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
