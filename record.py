"""
Record labeled telemetry data to CSV for training.

Usage examples:
    python record.py idle
    python record.py idle --interval 0.25 --duration 180
    python record.py gaming --interval 0.20 --duration 240 --out training_data.csv
    python record.py heavy --session-id run2_heavy
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from monitor import FEATURE_NAMES, get_telemetry

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


def record(
    label: str,
    out_csv: str = "training_data.csv",
    interval: float = 0.2,
    duration: float | None = None,
    session_id: str | None = None,
):
    """
    Log telemetry + label at fixed interval.

    Each row: timestamp, label, <FEATURE_NAMES...>
    """
    _validate_positive("interval", interval)
    if duration is not None:
        _validate_positive("duration", duration)
    if session_id is None:
        session_id = f"s_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

    sample = get_telemetry()
    n_feat = len(sample)
    if n_feat != len(FEATURE_NAMES):
        raise RuntimeError(
            "Telemetry feature length does not match FEATURE_NAMES. "
            "Check monitor.py."
        )

    out_path = Path(out_csv)
    _ensure_csv_schema(out_path)
    needs_header = not out_path.exists() or out_path.stat().st_size == 0

    duration_msg = f"{duration:.1f}s" if duration is not None else "until Ctrl+C"
    print(
        f"Recording label='{label}' to '{out_csv}' every {interval}s "
        f"for {duration_msg} (session_id={session_id})"
    )

    start = time.time()
    samples_written = 0

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
            f.flush()

    elapsed = max(time.time() - start, 1e-6)
    print(
        f"Wrote {samples_written} samples for label='{label}' "
        f"in {elapsed:.1f}s to '{out_csv}' (session_id={session_id})."
    )


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
        default="training_data.csv",
        help="Output CSV path (default: training_data.csv)",
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
    )
