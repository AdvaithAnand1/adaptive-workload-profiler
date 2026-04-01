"""
Collect a full labeled dataset in one command.

Examples:
    python collect_dataset.py
    python collect_dataset.py --sessions idle:120,light:180,heavy:240
    python collect_dataset.py --interval 0.25 --out training_data.csv --cycles 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from record import record

DEFAULT_SESSIONS = "idle:180,light:180,heavy:240"


def _parse_sessions(raw: str) -> list[tuple[str, float]]:
    sessions: list[tuple[str, float]] = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid session '{item}'. Expected format label:seconds"
            )
        label, seconds_str = item.split(":", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Invalid session '{item}': empty label")
        try:
            seconds = float(seconds_str)
        except ValueError as e:
            raise ValueError(
                f"Invalid duration in session '{item}': '{seconds_str}'"
            ) from e
        if seconds <= 0:
            raise ValueError(
                f"Invalid duration in session '{item}': must be > 0"
            )
        sessions.append((label, seconds))

    if not sessions:
        raise ValueError("At least one session is required")
    return sessions


def _countdown(seconds: int):
    if seconds <= 0:
        return
    for s in range(seconds, 0, -1):
        print(f"Starting in {s}s...", end="\r", flush=True)
        time.sleep(1.0)
    print(" " * 30, end="\r", flush=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect multi-label telemetry dataset.")
    p.add_argument(
        "--sessions",
        default=DEFAULT_SESSIONS,
        help=(
            "Comma-separated label:seconds plan "
            f"(default: {DEFAULT_SESSIONS})"
        ),
    )
    p.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Sampling interval in seconds (default: 0.2)",
    )
    p.add_argument(
        "--out",
        default="training_data.csv",
        help="Output CSV path (default: training_data.csv)",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Delete output CSV before collection starts.",
    )
    p.add_argument(
        "--countdown",
        type=int,
        default=5,
        help="Countdown before each session in seconds (default: 5)",
    )
    p.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Pause between sessions in seconds (default: 2.0)",
    )
    p.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Repeat the full session plan N times (default: 1).",
    )
    return p


def main():
    args = _build_arg_parser().parse_args()
    if args.cycles <= 0:
        raise ValueError("--cycles must be > 0")

    base_sessions = _parse_sessions(args.sessions)
    sessions = base_sessions * args.cycles
    out_path = Path(args.out)

    if args.reset and out_path.exists():
        out_path.unlink()
        print(f"Deleted existing dataset: '{args.out}'")

    print("Dataset collection plan:")
    if args.cycles > 1:
        print(f"  cycles: {args.cycles}")
    for i, (label, duration) in enumerate(sessions, start=1):
        print(f"  {i}. {label} for {duration:.1f}s")

    total = len(sessions)
    for i, (label, duration) in enumerate(sessions, start=1):
        print(f"\n[{i}/{total}] Prepare workload: '{label}'")
        _countdown(args.countdown)
        record(
            label=label,
            out_csv=args.out,
            interval=args.interval,
            duration=duration,
        )

        if i < total and args.cooldown > 0:
            print(f"Cooldown for {args.cooldown:.1f}s...")
            time.sleep(args.cooldown)

    print(f"\nFinished. Dataset saved to '{args.out}'.")


if __name__ == "__main__":
    main()
