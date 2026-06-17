from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model import SystemStateNet
from monitor import FEATURE_NAMES

MODEL_ARTIFACT_SCHEMA = "telemetry_norm_v1"
MODEL_METADATA_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelArtifactBundle:
    model: SystemStateNet
    classes: list[str]
    metadata: dict[str, Any]
    feature_mean: np.ndarray
    feature_std: np.ndarray


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise RuntimeError("model_metadata.json must contain a JSON object.")
    return metadata


def _normalize_stats(raw: Any, *, field: str, expected_len: int) -> np.ndarray:
    if not isinstance(raw, list):
        raise RuntimeError(f"model_metadata.{field} must be a JSON array.")
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != expected_len:
        raise RuntimeError(
            f"model_metadata.{field} must contain {expected_len} numeric values."
        )
    return arr


def load_model_artifacts(
    *,
    model_path: str | Path = "model.pth",
    classes_path: str | Path = "classes.json",
    metadata_path: str | Path = "model_metadata.json",
) -> ModelArtifactBundle:
    model_path = Path(model_path)
    classes_path = Path(classes_path)
    metadata_path = Path(metadata_path)

    missing = [str(path) for path in (model_path, classes_path, metadata_path) if not path.exists()]
    if missing:
        raise RuntimeError(
            "Missing model artifacts. Expected model.pth, classes.json, and "
            "model_metadata.json. Retrain with train_model.py."
        )

    classes_raw = _read_json(classes_path)
    if not isinstance(classes_raw, list) or not classes_raw:
        raise RuntimeError("classes.json must contain a non-empty JSON array.")
    classes = [str(name).strip() for name in classes_raw if str(name).strip()]
    if len(classes) != len(classes_raw):
        raise RuntimeError("classes.json contains empty class names.")

    metadata = _require_metadata(_read_json(metadata_path))
    schema = str(metadata.get("artifact_schema", "")).strip()
    version = int(metadata.get("schema_version", 0) or 0)
    if schema != MODEL_ARTIFACT_SCHEMA or version != MODEL_METADATA_SCHEMA_VERSION:
        raise RuntimeError(
            "Model metadata is incompatible with this runtime. "
            "Retrain with train_model.py to regenerate normalized artifacts."
        )

    feature_names = metadata.get("feature_names")
    if feature_names != FEATURE_NAMES:
        raise RuntimeError(
            "Model feature schema mismatch. Retrain with the current telemetry schema."
        )

    feature_mean = _normalize_stats(
        metadata.get("feature_mean"),
        field="feature_mean",
        expected_len=len(FEATURE_NAMES),
    )
    feature_std = _normalize_stats(
        metadata.get("feature_std"),
        field="feature_std",
        expected_len=len(FEATURE_NAMES),
    )
    if np.any(feature_std <= 0.0):
        raise RuntimeError("model_metadata.feature_std must contain positive values.")

    model = SystemStateNet(len(FEATURE_NAMES), len(classes))
    state = torch.load(model_path, map_location="cpu")
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            "Model weights are incompatible with the current telemetry schema. "
            "Retrain with train_model.py."
        ) from exc
    model.eval()

    return ModelArtifactBundle(
        model=model,
        classes=classes,
        metadata=metadata,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def normalize_features(values: np.ndarray | list[float], bundle: ModelArtifactBundle) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if x.shape[0] != bundle.feature_mean.shape[0]:
        raise RuntimeError(
            f"Feature vector has {x.shape[0]} values, expected {bundle.feature_mean.shape[0]}."
        )
    return (x - bundle.feature_mean) / bundle.feature_std
