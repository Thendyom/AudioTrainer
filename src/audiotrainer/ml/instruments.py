"""Optional local AudioSet AST adapter and target-label mapping."""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from audiotrainer.api.schemas import InstrumentCandidate, InstrumentEstimate
from audiotrainer.audio.preprocessing import resample_audio
from audiotrainer.ml.manager import BackendUnavailableError, model_path, require_feature

TARGET_LABELS = ("voice", "piano", "guitar", "violin", "flute", "saxophone")
LABEL_KEYWORDS = {
    "voice": ("speech", "singing", "choir", "vocal", "human voice"),
    "piano": ("piano", "keyboard"),
    "guitar": ("guitar",),
    "violin": ("violin", "fiddle"),
    "flute": ("flute",),
    "saxophone": ("saxophone",),
}


def classify_ast(
    audio: NDArray[np.floating],
    sr: int,
    *,
    ai_enabled: bool,
    data_dir: str | Path | None = None,
    confidence_threshold: float = 0.22,
    margin_threshold: float = 0.05,
) -> tuple[InstrumentEstimate, list[InstrumentCandidate], float]:
    """Classify with local AST weights, returning unknown below calibrated thresholds."""

    require_feature(ai_enabled, "instruments", "AST")
    if find_spec("transformers") is None or find_spec("torch") is None:
        raise BackendUnavailableError('AST is unavailable; install with pip install ".[ml-instruments]"')
    weights = model_path("instruments", data_dir)
    if not weights.is_dir() or not any(item.is_file() for item in weights.rglob("*")):
        raise BackendUnavailableError("AST weights are not downloaded in Models & Privacy")
    processor, model = _load_ast(str(weights.resolve()))
    if sr != 16_000:
        audio = resample_audio(audio, sr, 16_000)
        sr = 16_000
    import torch

    inputs = processor(np.asarray(audio, dtype=np.float32), sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probabilities = torch.sigmoid(logits).cpu().numpy()
    id_to_label = getattr(model.config, "id2label", {})
    raw = sorted(
        [
            (str(id_to_label.get(index, id_to_label.get(str(index), index))), float(probability))
            for index, probability in enumerate(probabilities)
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    return map_ast_labels(raw, confidence_threshold=confidence_threshold, margin_threshold=margin_threshold)


def map_ast_labels(
    raw_candidates: list[tuple[str, float]],
    *,
    confidence_threshold: float = 0.22,
    margin_threshold: float = 0.05,
) -> tuple[InstrumentEstimate, list[InstrumentCandidate], float]:
    """Map AudioSet labels into AudioTrainer's intentionally small vocabulary."""

    mapped: dict[str, tuple[float, str]] = {label: (0.0, "") for label in TARGET_LABELS}
    for raw_label, probability in raw_candidates:
        normalized = raw_label.lower()
        for target, keywords in LABEL_KEYWORDS.items():
            if any(keyword in normalized for keyword in keywords) and probability > mapped[target][0]:
                mapped[target] = (float(max(0.0, min(1.0, probability))), raw_label)
    candidates = [
        InstrumentCandidate(label=label, confidence=score, raw_label=raw_label or None)
        for label, (score, raw_label) in sorted(mapped.items(), key=lambda item: item[1][0], reverse=True)
    ]
    top = candidates[0]
    runner_up = candidates[1].confidence if len(candidates) > 1 else 0.0
    margin = max(0.0, top.confidence - runner_up)
    if top.confidence < confidence_threshold or margin < margin_threshold:
        estimate = InstrumentEstimate(
            label="unknown",
            confidence=top.confidence,
            explanation="AST did not produce a sufficiently strong, separated mapped instrument label.",
        )
    else:
        estimate = InstrumentEstimate(
            label=top.label,
            confidence=top.confidence,
            explanation=f"Experimental local AST estimate mapped from AudioSet label: {top.raw_label}.",
        )
    return estimate, candidates, margin


@lru_cache(maxsize=1)
def _load_ast(weights_path: str):
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    processor = AutoFeatureExtractor.from_pretrained(weights_path, local_files_only=True)
    model = AutoModelForAudioClassification.from_pretrained(weights_path, local_files_only=True)
    model.eval()
    return processor, model
