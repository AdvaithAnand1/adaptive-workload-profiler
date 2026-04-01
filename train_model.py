# train_model.py
"""
Train SystemStateNet from training_data.csv.

Requirements:
    pip install torch scikit-learn pandas numpy
"""

import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from model import SystemStateNet
from monitor import FEATURE_NAMES

DATA_FILE = "training_data.csv"
MODEL_FILE = "model.pth"
CLASSES_FILE = "classes.json"

BATCH_SIZE = 256
EPOCHS = 25
LR = 1e-3


def _split_indices(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    num_classes = len(np.unique(y))

    if groups is not None and len(np.unique(groups)) >= 3:
        num_groups = len(np.unique(groups))
        min_group_test = num_classes / max(1, num_groups)
        group_test_size = min(0.5, max(0.2, min_group_test + 0.02))

        for seed in range(42, 102):
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=group_test_size,
                random_state=seed,
            )
            train_idx, val_idx = next(splitter.split(X, y, groups=groups))
            if (
                len(np.unique(y[train_idx])) == num_classes
                and len(np.unique(y[val_idx])) == num_classes
            ):
                print(f"Using session-aware group split (seed={seed}).")
                return train_idx, val_idx
        print(
            "[WARN] Could not find class-complete session split; "
            "falling back to row-level stratified split."
        )
    elif groups is not None:
        print(
            "[WARN] Not enough distinct sessions for group split; "
            "falling back to row-level stratified split."
        )

    idx = np.arange(X.shape[0])
    train_idx, val_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return train_idx, val_idx


def load_dataset():
    df = pd.read_csv(DATA_FILE)

    required = ["timestamp", "label", *FEATURE_NAMES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            "training_data.csv does not match the current feature schema.\n"
            f"Missing columns: {missing}\n"
            "Re-record data with record.py after the telemetry update."
        )

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y_text = df["label"].to_numpy()
    groups = None
    if "session_id" in df.columns:
        groups = df["session_id"].astype(str).to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(y_text).astype(np.int64)

    train_idx, val_idx = _split_indices(X, y, groups)
    X_train = X[train_idx]
    X_val = X[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    class_counts = np.bincount(y_train, minlength=len(le.classes_))
    print(f"Class counts (train): {dict(zip(le.classes_, class_counts.tolist()))}")

    return train_ds, val_ds, le, X.shape[1], class_counts


def train():
    train_ds, val_ds, le, input_dim, class_counts = load_dataset()
    num_classes = len(le.classes_)

    model = SystemStateNet(input_dim, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    weights = class_counts.sum() / np.maximum(class_counts, 1)
    weights = weights / max(1e-8, weights.mean())
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        # ---- eval ----
        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.numel()
        val_acc = correct / max(1, total)
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} val_acc={val_acc:.3f}")

    # Save model + label mapping
    torch.save(model.state_dict(), MODEL_FILE)
    with open(CLASSES_FILE, "w", encoding="utf-8") as f:
        json.dump(list(le.classes_), f, indent=2)
    print(f"Saved {MODEL_FILE} and {CLASSES_FILE}")


if __name__ == "__main__":
    train()
