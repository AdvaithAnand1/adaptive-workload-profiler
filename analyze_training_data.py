"""Analyze training-data coverage before retraining.

Examples:
    python analyze_training_data.py --csv training_data.csv
    python analyze_training_data.py --csv training_data.csv --json-out run_logs/training_data_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

POWER_SAVER_COLUMN = "Power Saver [0/1]"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze training-data coverage for PerfAnalyze.")
    p.add_argument("--csv", default="training_data.csv", help="Training CSV path.")
    p.add_argument("--json-out", default="", help="Optional JSON output path.")
    return p


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_training_rows(rows: list[dict[str, str]], *, csv_path: str | Path = "") -> dict[str, Any]:
    if not rows:
        raise RuntimeError("Training CSV is empty.")

    required = {"timestamp", "session_id", "label"}
    missing = [name for name in required if name not in rows[0]]
    if missing:
        raise RuntimeError(f"Training CSV is missing required columns: {missing}")

    label_counts = Counter(str(r.get("label", "")).strip().lower() for r in rows)
    label_counts.pop("", None)
    if not label_counts:
        raise RuntimeError("Training CSV does not contain usable labels.")

    session_counts = Counter(str(r.get("session_id", "")).strip() for r in rows)
    session_counts.pop("", None)

    sessions_by_label: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        label = str(row.get("label", "")).strip().lower()
        session_id = str(row.get("session_id", "")).strip()
        if label and session_id:
            sessions_by_label[label].add(session_id)

    saver_balance_by_label: dict[str, dict[str, int]] = {}
    if POWER_SAVER_COLUMN in rows[0]:
        for label in label_counts:
            on_count = 0
            off_count = 0
            for row in rows:
                row_label = str(row.get("label", "")).strip().lower()
                if row_label != label:
                    continue
                raw = str(row.get(POWER_SAVER_COLUMN, "0")).strip()
                try:
                    value = float(raw)
                except ValueError:
                    value = 0.0
                if value >= 0.5:
                    on_count += 1
                else:
                    off_count += 1
            saver_balance_by_label[label] = {"on": on_count, "off": off_count}

    counts = list(label_counts.values())
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = min_count / max_count if max_count else 0.0

    recommendations: list[str] = []
    warnings: list[str] = []

    if len(label_counts) < 3:
        warnings.append(
            "Fewer than three labels were recorded. Add more workload classes before trusting retraining results."
        )

    sparse_labels = [label for label, count in label_counts.items() if count < max(120, int(max_count * 0.5))]
    if sparse_labels:
        recommendations.append(
            "Collect more samples for underrepresented labels: " + ", ".join(sorted(sparse_labels))
        )

    low_session_labels = [label for label, sessions in sessions_by_label.items() if len(sessions) < 2]
    if low_session_labels:
        recommendations.append(
            "Record additional separate sessions for labels with only one session: "
            + ", ".join(sorted(low_session_labels))
        )

    if imbalance_ratio < 0.55:
        warnings.append(
            f"Label counts are imbalanced (min/max ratio={imbalance_ratio:.2f}). Retraining may overfit dominant labels."
        )

    saver_imbalanced_labels: list[str] = []
    for label, counts_by_state in saver_balance_by_label.items():
        total = counts_by_state["on"] + counts_by_state["off"]
        if total == 0:
            continue
        dominant = max(counts_by_state["on"], counts_by_state["off"]) / total
        if dominant > 0.85:
            saver_imbalanced_labels.append(label)
    if saver_imbalanced_labels:
        recommendations.append(
            "Add power-saver on/off coverage for labels: " + ", ".join(sorted(saver_imbalanced_labels))
        )

    summary = {
        "csv": str(csv_path),
        "rows": len(rows),
        "label_counts": dict(label_counts),
        "session_counts": dict(session_counts),
        "sessions_per_label": {label: len(sessions) for label, sessions in sessions_by_label.items()},
        "power_saver_balance_by_label": saver_balance_by_label or None,
        "label_imbalance_ratio": imbalance_ratio,
        "recommendations": recommendations,
        "warnings": warnings,
    }
    return summary


def main():
    args = _build_arg_parser().parse_args()
    csv_path = Path(args.csv)
    rows = _read_rows(csv_path)
    summary = summarize_training_rows(rows, csv_path=csv_path)

    print(f"Training CSV: {csv_path}")
    print(f"- rows: {summary['rows']}")
    print(f"- label_counts: {summary['label_counts']}")
    print(f"- sessions_per_label: {summary['sessions_per_label']}")
    print(f"- label_imbalance_ratio: {float(summary['label_imbalance_ratio']):.3f}")
    if summary.get("power_saver_balance_by_label"):
        print(f"- power_saver_balance_by_label: {summary['power_saver_balance_by_label']}")
    if summary["warnings"]:
        print("- warnings:")
        for warning in summary["warnings"]:
            print(f"  - {warning}")
    if summary["recommendations"]:
        print("- recommendations:")
        for rec in summary["recommendations"]:
            print(f"  - {rec}")
    if not summary["warnings"] and not summary["recommendations"]:
        print("- coverage looks healthy for retraining")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"- wrote summary json: {out_path}")


if __name__ == "__main__":
    main()
