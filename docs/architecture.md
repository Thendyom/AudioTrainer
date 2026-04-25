# AudioTrainer Architecture

AudioTrainer is organized as a reusable Python library with thin application entrypoints.

## Layers

### Core Library

The `audiotrainer` package owns all analysis logic:

- `audio`: file loading, microphone recording, framing, and preprocessing.
- `pitch`: YIN/autocorrelation pitch detection, note conversion, and smoothing.
- `transcription`: pitch-track segmentation plus CSV and MIDI export.
- `speech`: prosody, pause detection, reference comparison, and voice profiling.
- `instruments`: spectral feature extraction and rule-based classification.
- `coaching`: scoring and feedback generation.
- `visualization`: optional matplotlib plotting helpers.
- `api`: pydantic result schemas and service functions.

Public results are pydantic models so CLI output, web responses, and library users share the same contracts.

### Application Layer

The Typer CLI, Streamlit UI, and FastAPI app call `audiotrainer.api.service`. They do not duplicate pitch detection, transcription, speech, or classifier logic.

### Dependency Strategy

The first implementation avoids TensorFlow, PyTorch, source separation, large ASR models, and pretrained weights. Core analysis is implemented with NumPy and small helper dependencies. App-only dependencies stay in the `app` optional extra.

### Optional Model Adapters

Future ML backends should be added behind narrow interfaces:

- pitch detector adapter returning `PitchTrack`
- pronunciation adapter returning `PronunciationReport`
- instrument classifier adapter returning `InstrumentEstimate`

Adapters should be optional extras and must not make the deterministic baseline slower or harder to install.
