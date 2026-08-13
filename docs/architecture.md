# AudioTrainer Architecture

AudioTrainer is organized as a reusable Python library with thin application entrypoints.

## Layers

### Core Library

The `audiotrainer` package owns all analysis logic:

- `audio`: file loading, browser-stream buffering, quality analysis, framing, and preprocessing.
- `pitch`: YIN/autocorrelation pitch detection, note conversion, and smoothing.
- `transcription`: pitch-track segmentation, editable score documents, and CSV/JSON/MIDI/MusicXML export.
- `speech`: prosody, pause detection, reference comparison, and voice profiling.
- `instruments`: spectral feature extraction and rule-based classification.
- `coaching`: scoring and feedback generation.
- `visualization`: optional matplotlib plotting helpers.
- `api`: pydantic result schemas and service functions.
- `backends`: stable discovery of the built-in deterministic capabilities.
- `history`: versioned local SQLite persistence with opt-in managed recordings.

Public results are pydantic models so CLI output, web responses, and library users share the same contracts.

### Application Layer

The Typer CLI, Streamlit UI, and FastAPI app call `audiotrainer.api.service`. They do not duplicate pitch detection, transcription, speech, or classifier logic.

### Dependency Strategy

The product avoids TensorFlow, PyTorch, source separation, cloud APIs, and pretrained weights. Analysis is implemented with NumPy, SciPy, and small helper dependencies. App-only dependencies stay in the `app` optional extra.

### Analysis Provenance

`AnalysisMetadata` records the built-in engine, processing time, and recording warnings. The legacy `auto` value remains accepted as an alias for source compatibility, but this release has no trained-model execution path.

### Local Data

Practice history lives in `platformdirs.user_data_dir("AudioTrainer")`. SQLite `PRAGMA user_version` controls migrations. Recordings are copied into managed storage only after opt-in and deleted with their session.
