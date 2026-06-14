"""
Analyze a probe CSV log and output a functional/stability summary.

Example:
    python analyze_probe.py --log run_logs/study_probe.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze probe log quality and stability.")
    p.add_argument("--log", required=True, help="Path to probe CSV log.")
    p.add_argument(
        "--json-out",
        default="",
        help="Optional path to write machine-readable JSON summary.",
    )
    return p


def _pct(x: float) -> float:
    return round(100.0 * x, 2)


def _safe_rate(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "").strip())
    except Exception:
        return default


def _resolve_log_path(raw: str) -> Path:
    requested = Path(raw).expanduser()
    candidates: list[Path] = []

    if requested.is_absolute():
        candidates.append(requested)
    else:
        # Check current working directory first, then script directory.
        cwd = Path.cwd()
        script_dir = Path(__file__).resolve().parent
        candidates.append(cwd / requested)
        if script_dir != cwd:
            candidates.append(script_dir / requested)
        candidates.append(requested)

    for p in candidates:
        if p.exists():
            return p.resolve()

    checked = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Log file not found: {raw}. Checked: {checked}")


def _compute_score(
    low_conf_ratio: float,
    switches_per_10min: float,
    dominant_label_ratio: float,
    saver_heavy_gap: float | None,
) -> tuple[int, list[str]]:
    score = 100.0
    notes: list[str] = []

    if low_conf_ratio > 0.30:
        penalty = min(35.0, (low_conf_ratio - 0.30) * 120.0)
        score -= penalty
        notes.append(f"high uncertain prediction ratio ({_pct(low_conf_ratio)}%)")

    if switches_per_10min > 4.0:
        penalty = min(20.0, (switches_per_10min - 4.0) * 3.0)
        score -= penalty
        notes.append(f"profile churn is high ({switches_per_10min:.2f} switches/10min)")

    if dominant_label_ratio > 0.85:
        penalty = min(25.0, (dominant_label_ratio - 0.85) * 180.0)
        score -= penalty
        notes.append(f"model collapses to one label ({_pct(dominant_label_ratio)}% dominant)")

    if saver_heavy_gap is not None and saver_heavy_gap > 0.35:
        penalty = min(20.0, (saver_heavy_gap - 0.35) * 60.0)
        score -= penalty
        notes.append(
            "predictions appear strongly power-mode-sensitive "
            f"(heavy-rate gap {saver_heavy_gap:.2f})"
        )

    final = max(0, int(round(score)))
    return final, notes


def load_probe_rows(raw_log_path: str) -> tuple[Path, list[dict[str, str]]]:
    log_path = _resolve_log_path(raw_log_path)
    with log_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows_data = list(reader)
    return log_path, rows_data


def summarize_probe_rows(
    rows_data: list[dict[str, str]],
    *,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    if not rows_data:
        raise RuntimeError("Log file is empty.")

    required_cols = {"elapsed_s", "pred_label", "confidence", "confident", "switched"}
    missing = [c for c in required_cols if c not in rows_data[0]]
    if missing:
        raise RuntimeError(f"Log is missing required columns: {missing}")

    rows = len(rows_data)
    elapsed = [_f(r, "elapsed_s") for r in rows_data]
    duration_s = max(elapsed) - min(elapsed) if rows > 1 else 0.0
    minutes = duration_s / 60.0

    labels = [r.get("pred_label", "").strip() for r in rows_data]
    label_counts = dict(Counter(labels))
    dominant_label_ratio = max(label_counts.values()) / max(1, rows)

    confidences = [_f(r, "confidence") for r in rows_data]
    avg_conf = sum(confidences) / max(1, rows)
    margins = [_f(r, "margin") for r in rows_data] if "margin" in rows_data[0] else []
    avg_margin = sum(margins) / max(1, len(margins)) if margins else None

    raw_confidences = None
    avg_raw_conf = None
    raw_margins = None
    avg_raw_margin = None
    if "raw_confidence" in rows_data[0]:
        raw_confidences = [_f(r, "raw_confidence") for r in rows_data]
        avg_raw_conf = sum(raw_confidences) / max(1, rows)
    if "raw_margin" in rows_data[0]:
        raw_margins = [_f(r, "raw_margin") for r in rows_data]
        avg_raw_margin = sum(raw_margins) / max(1, rows)

    low_conf = sum(1 for r in rows_data if _f(r, "confident") < 0.5)
    low_conf_ratio = low_conf / max(1, rows)

    switches = sum(1 for r in rows_data if _f(r, "switched") >= 0.5)
    switches_per_10min = _safe_rate(float(switches), max(minutes, 1e-9)) * 10.0

    verified_switches = None
    verification_rate = None
    if "switch_verified" in rows_data[0]:
        verified_rows = [r for r in rows_data if str(r.get("switch_verified", "")).strip() != ""]
        if verified_rows:
            verified_switches = sum(1 for r in verified_rows if _f(r, "switch_verified") >= 0.5)
            verification_rate = verified_switches / len(verified_rows)

    gate_reason_counts = None
    if "gate_reason" in rows_data[0]:
        gate_reason_counts = dict(
            Counter(r.get("gate_reason", "").strip() for r in rows_data if r.get("gate_reason", "").strip())
        )

    saver_heavy_gap = None
    heavy_rate_saver_on = None
    heavy_rate_saver_off = None
    if "power_saver" in rows_data[0]:
        saver_on = [r for r in rows_data if _f(r, "power_saver") >= 0.5]
        saver_off = [r for r in rows_data if _f(r, "power_saver") < 0.5]
        if saver_on and saver_off:
            heavy_rate_saver_on = sum(
                1 for r in saver_on if r.get("pred_label", "").strip().lower() == "heavy"
            ) / len(saver_on)
            heavy_rate_saver_off = sum(
                1 for r in saver_off if r.get("pred_label", "").strip().lower() == "heavy"
            ) / len(saver_off)
            saver_heavy_gap = abs(heavy_rate_saver_on - heavy_rate_saver_off)

    score, notes = _compute_score(
        low_conf_ratio=low_conf_ratio,
        switches_per_10min=switches_per_10min,
        dominant_label_ratio=dominant_label_ratio,
        saver_heavy_gap=saver_heavy_gap,
    )

    if score >= 80:
        grade = "good"
    elif score >= 60:
        grade = "usable"
    elif score >= 40:
        grade = "unstable"
    else:
        grade = "poor"

    return {
        "log": str(log_path) if log_path is not None else "",
        "rows": rows,
        "duration_s": duration_s,
        "avg_confidence": avg_conf,
        "avg_decision_confidence": avg_conf,
        "avg_decision_margin": avg_margin,
        "avg_raw_confidence": avg_raw_conf,
        "avg_raw_margin": avg_raw_margin,
        "low_confidence_ratio": low_conf_ratio,
        "switches": switches,
        "switches_per_10min": switches_per_10min,
        "verified_switches": verified_switches,
        "switch_verification_rate": verification_rate,
        "label_counts": label_counts,
        "dominant_label_ratio": dominant_label_ratio,
        "gate_reason_counts": gate_reason_counts,
        "heavy_rate_power_saver_on": heavy_rate_saver_on,
        "heavy_rate_power_saver_off": heavy_rate_saver_off,
        "heavy_rate_power_saver_gap": saver_heavy_gap,
        "functional_score": score,
        "functional_grade": grade,
        "issues": notes,
    }


def main():
    args = _build_arg_parser().parse_args()
    log_path, rows_data = load_probe_rows(args.log)
    summary = summarize_probe_rows(rows_data, log_path=log_path)

    print(f"Probe Log: {log_path}")
    print(f"- rows: {summary['rows']}")
    duration_s = float(summary["duration_s"])
    minutes = duration_s / 60.0
    print(f"- duration: {duration_s:.1f}s ({minutes:.2f} min)")
    print(f"- avg_decision_confidence: {summary['avg_decision_confidence']:.3f}")
    if summary.get("avg_raw_confidence") is not None:
        print(f"- avg_raw_confidence: {summary['avg_raw_confidence']:.3f}")
    if summary.get("avg_decision_margin") is not None:
        print(f"- avg_decision_margin: {summary['avg_decision_margin']:.3f}")
    if summary.get("avg_raw_margin") is not None:
        print(f"- avg_raw_margin: {summary['avg_raw_margin']:.3f}")
    print(f"- low_confidence_pct: {_pct(float(summary['low_confidence_ratio'])):.2f}%")
    print(
        f"- switches: {summary['switches']} ({float(summary['switches_per_10min']):.2f} per 10 min)"
    )
    if summary.get("switch_verification_rate") is not None:
        print(
            f"- switch_verification_rate: {float(summary['switch_verification_rate']):.3f} "
            f"({summary.get('verified_switches')} verified)"
        )
    print(f"- label_counts: {summary['label_counts']}")
    print(f"- dominant_label_ratio: {float(summary['dominant_label_ratio']):.3f}")
    if summary.get("gate_reason_counts") is not None:
        print(f"- gate_reason_counts: {summary['gate_reason_counts']}")
    if summary.get("heavy_rate_power_saver_gap") is not None:
        print(
            "- heavy_rate_power_saver_on/off: "
            f"{float(summary['heavy_rate_power_saver_on']):.3f}/"
            f"{float(summary['heavy_rate_power_saver_off']):.3f} "
            f"(gap={float(summary['heavy_rate_power_saver_gap']):.3f})"
        )
    print(
        f"- functional_score: {summary['functional_score']}/100 ({summary['functional_grade']})"
    )

    issues = list(summary.get("issues", []))
    if issues:
        print("- issues:")
        for n in issues:
            print(f"  - {n}")
    else:
        print("- issues: none flagged by heuristics")

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"- wrote summary json: {json_path}")


if __name__ == "__main__":
    main()
