"""
Record labeled telemetry data to CSV for training.

Usage examples:
    python record.py idle
    python record.py idle --interval 0.25 --duration 180
    python record.py youtube --interval 0.20 --duration 240
    python record.py gaming --out training_data.csv
    python record.py heavy --session-id run2_heavy
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from monitor import FEATURE_NAMES, get_telemetry
from training.session_store import (
    default_session_manifest_path,
    write_training_session_manifest,
)

CSV_HEADER = ["timestamp", "session_id", "label", *FEATURE_NAMES]


def _validate_positive(name: str, value: float):
    if value <= 0:
        raise ValueError(f"{name} must be > 0; got {value}")


def _ensure_csv_schema(out_path: Path):
    if not out_path.exists() or out_path.stat().st_size == 0:
        return

    with out_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return

    if header != CSV_HEADER:
        raise RuntimeError(
            f"CSV schema mismatch in '{out_path}'.\n"
            "The file was created with a different feature set.\n"
            "Use a new output path or delete the old file and re-record."
        )


def _label_slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "_", label.strip().lower()).strip("_")
    if not slug:
        raise ValueError("label must contain at least one letter or number")
    return slug


def _default_output_path(data_dir: str | Path, label: str, session_id: str) -> Path:
    return Path(data_dir) / _label_slug(label) / f"{session_id}.csv"


def record(
    label: str,
    out_csv: str | None = None,
    interval: float = 0.2,
    duration: float | None = None,
    session_id: str | None = None,
    data_dir: str | Path = "data",
    session_manifest: Mapping[str, Any] | None = None,
    session_manifest_path: str | Path | None = None,
) -> Path:
    """
    Log telemetry + label at fixed interval.

    Each row: timestamp, label, <FEATURE_NAMES...>
    """
    _validate_positive("interval", interval)
    if duration is not None:
        _validate_positive("duration", duration)
    if session_id is None:
        session_id = f"s_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    session_id = _label_slug(session_id)
    started_at = datetime.now().isoformat()

    sample = get_telemetry()
    n_feat = len(sample)
    if n_feat != len(FEATURE_NAMES):
        raise RuntimeError(
            "Telemetry feature length does not match FEATURE_NAMES. "
            "Check monitor.py."
        )

    out_path = (
        Path(out_csv)
        if out_csv
        else _default_output_path(data_dir, label, session_id)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_csv_schema(out_path)
    needs_header = not out_path.exists() or out_path.stat().st_size == 0

    duration_msg = f"{duration:.1f}s" if duration is not None else "until Ctrl+C"
    print(
        f"Recording label='{label}' to '{out_path}' every {interval}s "
        f"for {duration_msg} (session_id={session_id})"
    )

    start = time.time()
    samples_written = 0
    ended_at = started_at

    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(CSV_HEADER)

        try:
            while True:
                x = get_telemetry()
                if x.shape[0] != n_feat:
                    raise RuntimeError(
                        "Feature length changed during recording; "
                        "check monitor.FEATURE_NAMES"
                    )
                writer.writerow(
                    [datetime.now().isoformat(), session_id, label, *x.tolist()]
                )
                samples_written += 1

                # Keep disk flush overhead low while still protecting progress.
                if samples_written % 20 == 0:
                    f.flush()

                if duration is not None and (time.time() - start) >= duration:
                    break

                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped recording (Ctrl+C).")
        finally:
            ended_at = datetime.now().isoformat()
            f.flush()

    if session_manifest is not None or session_manifest_path is not None:
        manifest_payload: dict[str, Any] = dict(session_manifest or {})
        manifest_payload.update(
            {
                "session_id": session_id,
                "label": label,
                "csv_path": str(out_path),
                "started_at": manifest_payload.get("started_at") or started_at,
                "ended_at": ended_at,
            }
        )
        manifest_path = (
            Path(session_manifest_path)
            if session_manifest_path is not None
            else default_session_manifest_path(out_path)
        )
        write_training_session_manifest(manifest_payload, manifest_path)

    elapsed = max(time.time() - start, 1e-6)
    print(
        f"Wrote {samples_written} samples for label='{label}' "
        f"in {elapsed:.1f}s to '{out_path}' (session_id={session_id})."
    )
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Record labeled telemetry samples.")
    p.add_argument("label", help="Workload label to assign (for example: idle, gaming)")
    p.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Sampling interval in seconds (default: 0.2)",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional fixed recording duration in seconds (default: run until Ctrl+C)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional explicit CSV path; otherwise writes one file per session.",
    )
    p.add_argument(
        "--data-dir",
        default="data",
        help="Session dataset root used when --out is omitted (default: data).",
    )
    p.add_argument(
        "--session-id",
        default=None,
        help="Optional session id for this recording window.",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    record(
        label=args.label,
        out_csv=args.out,
        interval=args.interval,
        duration=args.duration,
        session_id=args.session_id,
        data_dir=args.data_dir,
    )
