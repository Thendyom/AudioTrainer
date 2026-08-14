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
- `ml`: optional pYIN, Faster-Whisper, AudioSet AST, localhost generative coaching, settings, and model lifecycle adapters.

Public results are pydantic models so CLI output, web responses, and library users share the same contracts.

### Application Layer

The Typer CLI, Streamlit UI, and FastAPI app call `audiotrainer.api.service`. They do not duplicate pitch detection, transcription, speech, or classifier logic.

### Dependency Strategy

The always-available path uses NumPy, SciPy, and small helper dependencies. Heavy runtimes are separated into `ml-pitch`, `ml-speech`, and `ml-instruments` extras. They are imported lazily inside adapter calls. Installing an extra never downloads weights; only the Models & Privacy page or the `models download` CLI command can populate AudioTrainer-managed model directories.

### Analysis Provenance

`AnalysisMetadata` records requested and actual engines, processing time, warnings, and fallback reasons. `auto` may try an enabled local model, but always discloses a deterministic fallback. Explicit disabled or unavailable model requests fail rather than silently switching engines.

### AI Boundaries

The global AI switch defaults off and each feature has its own switch. Faster-Whisper and AST accept only managed local weights. Generative coaching accepts only localhost endpoints and sends measured values and transcript text, never audio. No cloud inference provider is built in.

### Local Data

Practice history lives in `platformdirs.user_data_dir("AudioTrainer")`. SQLite `PRAGMA user_version` controls migrations. Recordings are copied into managed storage only after opt-in and deleted with their session.
