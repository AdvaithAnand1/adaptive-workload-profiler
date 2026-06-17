"""Train a fresh SystemStateNet from all session CSVs under a data folder."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from model_artifacts import (
    MODEL_ARTIFACT_SCHEMA,
    MODEL_METADATA_SCHEMA_VERSION,
)
from model import SystemStateNet
from monitor import FEATURE_NAMES
from training_report import (
    compute_accuracy_by_label,
    compute_class_counts,
    compute_class_weights,
)

LATEST_MODEL_FILE = Path("model.pth")
LATEST_CLASSES_FILE = Path("classes.json")
LATEST_METADATA_FILE = Path("model_metadata.json")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train from every session CSV beneath a data directory."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def discover_session_csvs(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    files = sorted(path for path in root.rglob("*.csv") if path.is_file())
    if not files:
        raise RuntimeError(f"No session CSV files found under '{root}'.")
    return files


def load_session_dataframe(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, list[Path]]:
    files = discover_session_csvs(data_dir)
    required = ["timestamp", "session_id", "label", *FEATURE_NAMES]
    frames: list[pd.DataFrame] = []

    for path in files:
        frame = pd.read_csv(path)
        missing = [name for name in required if name not in frame.columns]
        if missing:
            raise RuntimeError(
                f"'{path}' does not match the current feature schema. "
                f"Missing columns: {missing}"
            )
        frames.append(frame[required])

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise RuntimeError("The session CSV files contain no samples.")
    return combined, files


def split_by_session(
    labels: np.ndarray,
    sessions: np.ndarray,
    *,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(labels)
    sessions_per_class = {
        int(label): len(np.unique(sessions[labels == label])) for label in classes
    }
    sparse = [label for label, count in sessions_per_class.items() if count < 2]
    if sparse:
        raise RuntimeError(
            "Need at least two separate recording sessions for every label "
            "before creating an honest session-isolated test set."
        )

    sample_indexes = np.arange(labels.shape[0])
    group_count = len(np.unique(sessions))
    minimum_test_fraction = len(classes) / max(1, group_count)
    effective_test_size = min(
        0.5,
        max(test_size, minimum_test_fraction + 0.01),
    )

    for candidate_seed in range(seed, seed + 100):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=effective_test_size,
            random_state=candidate_seed,
        )
        train_idx, test_idx = next(
            splitter.split(sample_indexes, labels, groups=sessions)
        )
        if (
            len(np.unique(labels[train_idx])) == len(classes)
            and len(np.unique(labels[test_idx])) == len(classes)
        ):
            return train_idx, test_idx

    raise RuntimeError(
        "Could not create train/test session groups containing every label. "
        "Record additional sessions for each label."
    )


def _effective_batch_size(sample_count: int, requested: int) -> int:
    size = min(sample_count, requested)
    if size < 2:
        raise RuntimeError("Training requires at least two samples.")
    if sample_count % size == 1 and size > 2:
        size -= 1
    return size


def _write_json(path: Path, payload: object):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_artifacts(
    model: SystemStateNet,
    classes: list[str],
    metadata: dict[str, object],
    *,
    model_dir: str | Path,
) -> Path:
    version = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    version_dir = Path(model_dir) / version
    version_dir.mkdir(parents=True, exist_ok=False)

    state = model.state_dict()
    torch.save(state, version_dir / "model.pth")
    _write_json(version_dir / "classes.json", classes)
    _write_json(version_dir / "model_metadata.json", metadata)

    torch.save(state, LATEST_MODEL_FILE)
    _write_json(LATEST_CLASSES_FILE, classes)
    _write_json(LATEST_METADATA_FILE, metadata)
    return version_dir


def train(
    *,
    data_dir: str | Path = "data",
    model_dir: str | Path = "models",
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    test_size: float = 0.2,
    seed: int = 42,
) -> float:
    if epochs <= 0 or batch_size <= 0 or lr <= 0:
        raise ValueError("epochs, batch-size, and lr must be positive")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test-size must be between 0 and 1")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    frame, files = load_session_dataframe(data_dir)
    feature_frame = frame[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    X = feature_frame.to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise RuntimeError("Dataset contains missing or non-numeric feature values.")

    label_text = frame["label"].astype(str).str.strip().to_numpy()
    if any(not label for label in label_text):
        raise RuntimeError("Dataset contains empty labels.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(label_text).astype(np.int64)
    sessions = frame["session_id"].astype(str).to_numpy()
    train_idx, test_idx = split_by_session(
        y,
        sessions,
        test_size=test_size,
        seed=seed,
    )

    train_features = X[train_idx]
    feature_mean = train_features.mean(axis=0).astype(np.float32)
    feature_std = train_features.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    X_norm = ((X - feature_mean) / feature_std).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(X_norm[train_idx]),
        torch.from_numpy(y[train_idx]),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_norm[test_idx]),
        torch.from_numpy(y[test_idx]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SystemStateNet(X.shape[1], len(encoder.classes_)).to(device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=_effective_batch_size(len(train_ds), batch_size),
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    counts = compute_class_counts(y[train_idx], len(encoder.classes_))
    weights = torch.tensor(
        compute_class_weights(counts),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.inference_mode():
        for features, targets in test_loader:
            features = features.to(device)
            targets = targets.to(device)
            predictions = model(features).argmax(dim=1)
            correct += int((predictions == targets).sum().item())
            total += int(targets.numel())
            y_true.extend(int(value) for value in targets.cpu().tolist())
            y_pred.extend(int(value) for value in predictions.cpu().tolist())

    accuracy = correct / max(1, total)
    classes = [str(name) for name in encoder.classes_]
    per_label_accuracy = compute_accuracy_by_label(y_true, y_pred, classes)
    metadata: dict[str, object] = {
        "schema_version": MODEL_METADATA_SCHEMA_VERSION,
        "artifact_schema": MODEL_ARTIFACT_SCHEMA,
        "created_at": datetime.now().isoformat(),
        "architecture": f"{len(FEATURE_NAMES)}-64-64-{len(classes)}",
        "feature_names": FEATURE_NAMES,
        "classes": classes,
        "source_files": [str(path) for path in files],
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "test_accuracy": accuracy,
        "test_accuracy_by_label": per_label_accuracy,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "dataset_version": "session_csv_v1",
        "seed": seed,
    }
    save_artifacts(model, classes, metadata, model_dir=model_dir)
    return accuracy


def main():
    args = _build_arg_parser().parse_args()
    accuracy = train(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(f"Accuracy: {accuracy * 100.0:.2f}%")


if __name__ == "__main__":
    main()
