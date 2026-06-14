from __future__ import annotations

from collections import Counter
from typing import Iterable


def compute_class_counts(labels: Iterable[int], num_classes: int) -> list[int]:
    counts = [0] * num_classes
    for label in labels:
        idx = int(label)
        if 0 <= idx < num_classes:
            counts[idx] += 1
    return counts


def compute_class_weights(counts: list[int], *, min_weight: float = 0.25, max_weight: float = 4.0) -> list[float]:
    nonzero = [count for count in counts if count > 0]
    if not nonzero:
        return [1.0 for _ in counts]
    total = sum(nonzero)
    n = len(nonzero)
    weights: list[float] = []
    for count in counts:
        if count <= 0:
            weights.append(max_weight)
            continue
        raw = total / (n * count)
        weights.append(max(min_weight, min(max_weight, raw)))
    return weights


def format_class_breakdown(class_names: list[str], counts: list[int], weights: list[float]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for idx, name in enumerate(class_names):
        rows.append(
            {
                "class_index": idx,
                "class_name": name,
                "count": counts[idx],
                "weight": round(weights[idx], 6),
            }
        )
    return rows


def compute_accuracy_by_label(y_true: Iterable[int], y_pred: Iterable[int], class_names: list[str]) -> dict[str, dict[str, float | int]]:
    correct = Counter()
    total = Counter()
    for truth, pred in zip(y_true, y_pred):
        truth_i = int(truth)
        pred_i = int(pred)
        if 0 <= truth_i < len(class_names):
            total[truth_i] += 1
            if truth_i == pred_i:
                correct[truth_i] += 1

    result: dict[str, dict[str, float | int]] = {}
    for idx, name in enumerate(class_names):
        denom = total[idx]
        result[name] = {
            "count": denom,
            "correct": correct[idx],
            "accuracy": (correct[idx] / denom) if denom else 0.0,
        }
    return result
