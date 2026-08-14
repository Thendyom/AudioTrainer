"""Settings, availability checks, and explicit model lifecycle management."""

from __future__ import annotations

from importlib.util import find_spec
import os
from pathlib import Path
import shutil
from typing import Literal

from pydantic import BaseModel, field_validator

from audiotrainer.api.schemas import ModelCapability
from audiotrainer.history import SessionRepository, default_data_dir

SPEECH_MODEL_ID = "Systran/faster-whisper-small"
INSTRUMENT_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"


class BackendUnavailableError(RuntimeError):
    """Raised when an explicitly requested optional backend cannot run."""


class BackendDisabledError(RuntimeError):
    """Raised when an explicitly requested AI backend is disabled."""


class AISettings(BaseModel):
    """Persistent local AI preferences. All optional features default off."""

    enabled: bool = False
    pitch_enabled: bool = True
    speech_enabled: bool = True
    instruments_enabled: bool = True
    generative_coaching_enabled: bool = False
    speech_model_id: str = SPEECH_MODEL_ID
    instrument_model_id: str = INSTRUMENT_MODEL_ID
    generative_endpoint: str = "http://127.0.0.1:11434"
    generative_model: str = ""

    @field_validator("generative_endpoint")
    @classmethod
    def validate_local_endpoint(cls, value: str) -> str:
        from urllib.parse import urlparse

        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("generative coaching endpoint must be localhost")
        return value.rstrip("/")

    def feature_enabled(self, feature: Literal["pitch", "speech", "instruments", "generative_coaching"]) -> bool:
        return self.enabled and bool(getattr(self, f"{feature}_enabled"))


def get_ai_settings(repository: SessionRepository | None = None) -> AISettings:
    """Read settings without importing or loading any ML runtime."""

    repo = repository or SessionRepository()
    saved = repo.get_setting("ai_settings", {})
    values = saved if isinstance(saved, dict) else {}
    env = os.environ.get("AUDIOTRAINER_AI_ENABLED")
    if env is not None:
        values = {**values, "enabled": env.strip().lower() in {"1", "true", "yes", "on"}}
    return AISettings.model_validate(values)


def save_ai_settings(settings: AISettings, repository: SessionRepository | None = None) -> None:
    """Persist local AI preferences."""

    (repository or SessionRepository()).set_setting("ai_settings", settings.model_dump(mode="json"))


def model_root(data_dir: str | Path | None = None) -> Path:
    return (Path(data_dir) if data_dir is not None else default_data_dir()) / "models"


def model_path(feature: Literal["speech", "instruments"], data_dir: str | Path | None = None) -> Path:
    return model_root(data_dir) / feature


def get_model_capabilities(
    settings: AISettings | None = None,
    *,
    data_dir: str | Path | None = None,
) -> list[ModelCapability]:
    """Report dependency/cache state using cheap import discovery only."""

    selected = settings or AISettings()
    speech_path = model_path("speech", data_dir)
    instrument_path = model_path("instruments", data_dir)
    pitch_dependency = find_spec("librosa") is not None
    speech_dependency = find_spec("faster_whisper") is not None and find_spec("huggingface_hub") is not None
    instrument_dependency = (
        find_spec("transformers") is not None
        and find_spec("torch") is not None
        and find_spec("huggingface_hub") is not None
    )
    capabilities = [
        ModelCapability(
            feature="pitch",
            backend="pyin",
            dependency_installed=pitch_dependency,
            weights_installed=True,
            available=pitch_dependency,
            enabled=selected.feature_enabled("pitch"),
            install_command='pip install ".[ml-pitch]"',
            status="Ready" if pitch_dependency else "Install the pitch extra",
        ),
        ModelCapability(
            feature="speech",
            backend="faster-whisper-small",
            dependency_installed=speech_dependency,
            weights_installed=_has_model_files(speech_path),
            available=speech_dependency and _has_model_files(speech_path),
            enabled=selected.feature_enabled("speech"),
            disk_usage_bytes=_directory_size(speech_path),
            install_command='pip install ".[ml-speech]"',
            model_id=selected.speech_model_id,
            status=_weighted_status(speech_dependency, speech_path),
        ),
        ModelCapability(
            feature="instruments",
            backend="ast",
            dependency_installed=instrument_dependency,
            weights_installed=_has_model_files(instrument_path),
            available=instrument_dependency and _has_model_files(instrument_path),
            enabled=selected.feature_enabled("instruments"),
            disk_usage_bytes=_directory_size(instrument_path),
            install_command='pip install ".[ml-instruments]"',
            model_id=selected.instrument_model_id,
            status=_weighted_status(instrument_dependency, instrument_path),
        ),
        ModelCapability(
            feature="generative_coaching",
            backend="local-ollama",
            dependency_installed=True,
            weights_installed=False,
            available=bool(selected.generative_model),
            enabled=selected.feature_enabled("generative_coaching"),
            install_command="Install Ollama locally and select one of your local models",
            status="Configured" if selected.generative_model else "Enter a local model name",
        ),
    ]
    return capabilities


def download_model(
    feature: Literal["speech", "instruments"],
    *,
    settings: AISettings | None = None,
    data_dir: str | Path | None = None,
) -> Path:
    """Explicitly download one model into AudioTrainer-managed storage."""

    if find_spec("huggingface_hub") is None:
        raise BackendUnavailableError("Install the relevant ML extra before downloading model weights")
    selected = settings or AISettings()
    repo_id = selected.speech_model_id if feature == "speech" else selected.instrument_model_id
    destination = model_path(feature, data_dir)
    destination.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=destination)
    return destination


def remove_model(feature: Literal["speech", "instruments"], *, data_dir: str | Path | None = None) -> bool:
    """Remove only the selected AudioTrainer-managed model directory."""

    destination = model_path(feature, data_dir)
    if not destination.is_dir():
        return False
    root = model_root(data_dir).resolve()
    if destination.resolve().parent != root:
        raise RuntimeError("Refusing to remove a model outside AudioTrainer storage")
    shutil.rmtree(destination)
    return True


def require_feature(enabled: bool, feature: str, backend: str) -> None:
    if not enabled:
        raise BackendDisabledError(f"{backend} is disabled; enable local AI and the {feature} feature first")


def _has_model_files(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def _directory_size(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _weighted_status(dependency_installed: bool, path: Path) -> str:
    if not dependency_installed:
        return "Install the matching ML extra"
    if not _has_model_files(path):
        return "Weights not downloaded"
    return "Ready"
