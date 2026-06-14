"""
Run live inference for a fixed duration and log behavior to CSV.

Typical usage:
    python run_probe.py --minutes 20 --out run_logs/study_probe.csv --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from config import CONFIG_FILE, load_config
from model import SystemStateNet
from monitor import FEATURE_NAMES, get_telemetry
from oracle_client import OracleClient, profile_for_label
from prediction_logic import ProbabilitySmoother, summarize_probabilities
from switch_policy import describe_gate, evaluate_switch_gate

MODEL_FILE = "model.pth"
CLASSES_FILE = "classes.json"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run inference probe for a fixed duration and write CSV log."
    )
    p.add_argument(
        "--config",
        default=CONFIG_FILE,
        help=(
            "Optional config JSON path. "
            f"Defaults to {CONFIG_FILE}; uses built-in defaults if missing."
        ),
    )
    p.add_argument(
        "--minutes",
        type=float,
        default=20.0,
        help="Run duration in minutes (default: 20)",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Telemetry/prediction interval in seconds (overrides config when set).",
    )
    p.add_argument(
        "--out",
        default="",
        help="Output CSV path. Default: run_logs/probe_YYYYmmdd_HHMMSS.csv",
    )
    p.add_argument(
        "--confidence-min",
        type=float,
        default=None,
        help="Minimum top-class confidence (overrides config when set).",
    )
    p.add_argument(
        "--confidence-margin",
        type=float,
        default=None,
        help="Minimum top1-top2 probability gap (overrides config when set).",
    )
    p.add_argument(
        "--stability-window",
        type=int,
        default=None,
        help="Consecutive reliable predictions needed before switching (overrides config).",
    )
    p.add_argument(
        "--min-switch-interval",
        type=float,
        default=None,
        help="Minimum seconds between profile switches (overrides config when set).",
    )
    p.add_argument(
        "--ema-alpha",
        type=float,
        default=None,
        help="EMA alpha for smoothing class probabilities (overrides config when set).",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Compute/log intended switches without external profile changes.",
    )
    mode.add_argument(
        "--apply-oracle",
        dest="dry_run",
        action="store_false",
        help="Actually call oracle profile switching.",
    )
    p.set_defaults(dry_run=True)
    return p


def _default_out_path() -> Path:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("run_logs") / f"probe_{now}.csv"


def _load_model_and_classes():
    model_path = Path(MODEL_FILE)
    classes_path = Path(CLASSES_FILE)
    if not model_path.exists() or not classes_path.exists():
        raise RuntimeError(
            "Missing model artifacts. Expected model.pth and classes.json."
        )

    with classes_path.open("r", encoding="utf-8") as f:
        classes = json.load(f)
    if not classes:
        raise RuntimeError("classes.json is empty; retrain the model.")

    model = SystemStateNet(len(FEATURE_NAMES), len(classes))
    state = torch.load(model_path, map_location="cpu")
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(
            "Model/feature schema mismatch. Re-record data and re-run train_model.py."
        ) from e
    model.eval()
    return model, classes


def _feature(x, index_map: dict[str, int], name: str) -> float:
    idx = index_map.get(name)
    if idx is None:
        return 0.0
    return float(x[idx])


def main():
    args = _build_arg_parser().parse_args()

    try:
        cfg_result = load_config(args.config)
    except RuntimeError as e:
        raise SystemExit(f"Config error: {e}") from e

    cfg = cfg_result.config

    if args.minutes <= 0:
        raise ValueError("--minutes must be > 0")

    interval = (
        args.interval
        if args.interval is not None
        else cfg.switching.poll_interval_sec
    )
    confidence_min = (
        args.confidence_min
        if args.confidence_min is not None
        else cfg.switching.confidence_min
    )
    confidence_margin = (
        args.confidence_margin
        if args.confidence_margin is not None
        else cfg.switching.confidence_margin
    )
    stability_window = (
        args.stability_window
        if args.stability_window is not None
        else cfg.switching.stability_window
    )
    min_switch_interval = (
        args.min_switch_interval
        if args.min_switch_interval is not None
        else cfg.switching.min_switch_interval_sec
    )
    ema_alpha = (
        args.ema_alpha
        if args.ema_alpha is not None
        else cfg.switching.probability_ema_alpha
    )

    if interval <= 0:
        raise ValueError("--interval (or config switching.poll_interval_sec) must be > 0")
    if stability_window <= 0:
        raise ValueError(
            "--stability-window (or config switching.stability_window) must be > 0"
        )
    if confidence_min < 0.0 or confidence_min > 1.0:
        raise ValueError(
            "--confidence-min (or config switching.confidence_min) must be in [0, 1]"
        )
    if confidence_margin < 0.0 or confidence_margin > 1.0:
        raise ValueError(
            "--confidence-margin (or config switching.confidence_margin) must be in [0, 1]"
        )
    if min_switch_interval < 0.0:
        raise ValueError(
            "--min-switch-interval (or config switching.min_switch_interval_sec) must be >= 0"
        )
    if ema_alpha < 0.0 or ema_alpha > 1.0:
        raise ValueError(
            "--ema-alpha (or config switching.probability_ema_alpha) must be in [0, 1]"
        )

    dry_run = bool(args.dry_run)
    out_path = Path(args.out) if args.out else _default_out_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, classes = _load_model_and_classes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    oracle = OracleClient()
    smoother = ProbabilitySmoother(ema_alpha)

    feature_index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    duration_sec = args.minutes * 60.0
    t0 = time.time()

    current_profile = None
    last_candidate = None
    stable_count = 0
    last_switch_ts = 0.0

    rows = 0
    switch_count = 0
    low_conf_count = 0
    label_counts: dict[str, int] = {}

    header = [
        "timestamp",
        "elapsed_s",
        "pred_label",
        "confidence",
        "runner_up_confidence",
        "margin",
        "raw_pred_label",
        "raw_confidence",
        "raw_runner_up_confidence",
        "raw_margin",
        "confident",
        "stable_count",
        "candidate_label",
        "target_profile",
        "current_profile",
        "gate_reason",
        "gate_direction",
        "gate_required_stability",
        "gate_wait_s",
        "switched",
        "switch_ok",
        "switch_verified",
        "switch_observed_profile",
        "switch_message",
        "oracle_backend",
        "config_source",
        "dry_run",
        "probability_ema_alpha",
        "on_battery",
        "power_saver",
        "cpu_usage_pct",
        "cpu_usage_ema_pct",
        "cpu_freq_mhz",
        "cpu_freq_ratio_pct",
        "ram_usage_pct",
        "disk_read_mb_s",
        "disk_write_mb_s",
        "net_down_kb_s",
        "net_up_kb_s",
        "process_count",
    ]

    print(
        f"Starting probe for {args.minutes:.1f} min at {interval:.2f}s interval. "
        f"dry_run={dry_run}, oracle={oracle.backend_name}, ema_alpha={ema_alpha:.2f}"
    )
    print(f"Config source: {cfg_result.source}")
    print(f"Logging to: {out_path}")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        while True:
            now = time.time()
            elapsed = now - t0
            if elapsed >= duration_sec:
                break

            x = get_telemetry()
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)

            with torch.inference_mode():
                logits = model(x_t)
                probs = F.softmax(logits, dim=1)

            raw_summary = summarize_probabilities(probs[0], classes)
            decision_probs = smoother.update(probs[0])
            decision_summary = summarize_probabilities(decision_probs, classes)

            pred_label = decision_summary.label
            confidence = decision_summary.confidence
            runner_up = decision_summary.runner_up_confidence
            margin = decision_summary.margin
            confident = (
                confidence >= confidence_min
                and margin >= confidence_margin
            )
            if not confident:
                low_conf_count += 1

            if confident:
                if pred_label == last_candidate:
                    stable_count += 1
                else:
                    last_candidate = pred_label
                    stable_count = 1
            else:
                last_candidate = None
                stable_count = 0

            target_profile = profile_for_label(
                pred_label,
                label_map=cfg.label_to_profile,
            )

            gate = evaluate_switch_gate(
                current_profile=current_profile,
                target_profile=target_profile,
                confident=confident,
                stable_count=stable_count,
                base_window=stability_window,
                downshift_extra_window=cfg.switching.downshift_extra_window,
                now_ts=now,
                last_switch_ts=last_switch_ts,
                min_switch_interval_sec=min_switch_interval,
                downshift_hold_sec=cfg.switching.downshift_hold_sec,
            )
            switched = False
            switch_ok = False
            switch_verified = ""
            switch_observed_profile = ""
            switch_message = ""
            if gate.allow:
                result = oracle.set_profile(target_profile, dry_run=dry_run)
                switched = True
                switch_ok = bool(result.ok)
                switch_verified = "" if result.verified is None else int(result.verified)
                switch_observed_profile = result.observed_profile or ""
                switch_message = result.message
                if result.ok:
                    current_profile = result.applied_profile or target_profile
                switch_count += 1
                last_switch_ts = now

            label_counts[pred_label] = label_counts.get(pred_label, 0) + 1

            writer.writerow(
                [
                    datetime.now().isoformat(),
                    round(elapsed, 3),
                    pred_label,
                    round(confidence, 6),
                    round(runner_up, 6),
                    round(margin, 6),
                    raw_summary.label,
                    round(raw_summary.confidence, 6),
                    round(raw_summary.runner_up_confidence, 6),
                    round(raw_summary.margin, 6),
                    int(confident),
                    stable_count,
                    last_candidate or "",
                    target_profile,
                    current_profile or "",
                    gate.reason,
                    gate.direction,
                    gate.required_stability,
                    round(gate.remaining_wait_s, 6),
                    int(switched),
                    int(switch_ok),
                    switch_verified,
                    switch_observed_profile,
                    switch_message,
                    oracle.backend_name,
                    cfg_result.source,
                    int(dry_run),
                    round(ema_alpha, 6),
                    _feature(x, feature_index, "On Battery [0/1]"),
                    _feature(x, feature_index, "Power Saver [0/1]"),
                    _feature(x, feature_index, "Total CPU Usage [%]"),
                    _feature(x, feature_index, "CPU Usage EMA [%]"),
                    _feature(x, feature_index, "CPU Frequency [MHz]"),
                    _feature(x, feature_index, "CPU Frequency Ratio [%]"),
                    _feature(x, feature_index, "RAM Usage [%]"),
                    _feature(x, feature_index, "Disk Read [MB/s]"),
                    _feature(x, feature_index, "Disk Write [MB/s]"),
                    _feature(x, feature_index, "Network Down [KB/s]"),
                    _feature(x, feature_index, "Network Up [KB/s]"),
                    _feature(x, feature_index, "Process Count"),
                ]
            )
            rows += 1

            if rows % 20 == 0:
                f.flush()
                print(
                    f"elapsed={elapsed:7.1f}s rows={rows:5d} "
                    f"raw={raw_summary.label:<8} decision={pred_label:<8} conf={confidence:.2f} "
                    f"stable={stable_count:2d}/{gate.required_stability} profile={current_profile} "
                    f"gate={describe_gate(gate)}"
                )

            time.sleep(interval)

        f.flush()

    run_sec = max(time.time() - t0, 1e-6)
    low_conf_pct = 100.0 * low_conf_count / max(1, rows)
    print("\nProbe complete.")
    print(f"- output: {out_path}")
    print(f"- rows: {rows}")
    print(f"- duration_s: {run_sec:.1f}")
    print(f"- switches: {switch_count}")
    print(f"- low_conf_pct: {low_conf_pct:.1f}%")
    print(f"- label_counts: {label_counts}")


if __name__ == "__main__":
    main()
