# Roadmap

AudioTrainer starts with deterministic audio analysis and leaves heavier ML backends optional.

## Near Term

- Improve YIN performance on noisy microphone input.
- Add live pitch buffering utilities shared by CLI and app.
- Add export helpers for JSON reports.
- Add more visualizations for speech pauses and intensity.
- Add regression fixtures with small generated audio clips.

## Optional Adapters

- `librosa.pyin` pitch backend as an optional adapter.
- scikit-learn instrument classifier adapter.
- phoneme alignment adapter for pronunciation reports.
- Whisper or wav2vec adapter for word-level speech feedback.

## Product

- Persistent coaching sessions.
- User-defined target-note exercises.
- Practice history and trend charts.
- Calibrated microphone setup checks.

## Quality

- More synthetic test coverage for noisy and unvoiced audio.
- Performance benchmarks for short clips.
- Public API stability checks.
