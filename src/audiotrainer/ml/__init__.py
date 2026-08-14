"""Optional, lazy-loaded local machine-learning backends."""

from audiotrainer.ml.manager import (
    AISettings,
    BackendDisabledError,
    BackendUnavailableError,
    download_model,
    get_ai_settings,
    get_model_capabilities,
    remove_model,
    save_ai_settings,
)

__all__ = [
    "AISettings",
    "BackendDisabledError",
    "BackendUnavailableError",
    "download_model",
    "get_ai_settings",
    "get_model_capabilities",
    "remove_model",
    "save_ai_settings",
]
