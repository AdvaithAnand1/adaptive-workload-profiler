"""Generate switching recommendations from a probe log.

Examples:
    python tune_probe_config.py --log run_logs/study_probe.csv
    python tune_probe_config.py --log run_logs/study_probe.csv --write-config perfalyze_config.tuned.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from analyze_probe import load_probe_rows, summarize_probe_rows
from config import CONFIG_FILE, PerfAnalyzeConfig, load_config, write_config
from probe_tuning import tune_switching_config


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tune PerfAnalyze switching config from a probe log.")
    p.add_argument("--log", required=True, help="Path to a probe CSV log.")
    p.add_argument(
        "--config",
        default=CONFIG_FILE,
        help=f"Base config to tune (default: {CONFIG_FILE}).",
    )
    p.add_argument(
        "--write-config",
        default="",
        help="Optional output config path. If omitted, prints recommendations only.",
    )
    p.add_argument(
        "--json-out",
        default="",
        help="Optional JSON path for the tuning report.",
    )
    return p


def _apply_report(cfg: PerfAnalyzeConfig, switching_patch: dict[str, float | int]) -> PerfAnalyzeConfig:
    new_switching = replace(cfg.switching, **switching_patch)
    return PerfAnalyzeConfig(
        switching=new_switching,
        label_to_profile=dict(cfg.label_to_profile),
        profile_hotkeys=dict(cfg.profile_hotkeys),
    )


def main():
    args = _build_arg_parser().parse_args()

    log_path, rows = load_probe_rows(args.log)
    summary = summarize_probe_rows(rows, log_path=log_path)
    cfg_result = load_config(args.config)
    report = tune_switching_config(summary, cfg_result.config.switching)

    print(f"Probe: {log_path}")
    print(f"Base config: {cfg_result.source}")
    print(f"Functional score: {summary['functional_score']}/100 ({summary['functional_grade']})")
    print("Recommended switching values:")
    for key, value in report.recommended_switching.items():
        marker = "*" if key in report.changes else " "
        print(f"{marker} {key}: {value}")

    if report.changes:
        print("Changes:")
        for key, change in report.changes.items():
            print(f"- {key}: {change.old} -> {change.new} ({change.reason})")
    else:
        print("Changes: none")

    if report.notes:
        print("Notes:")
        for note in report.notes:
            print(f"- {note}")

    if report.warnings:
        print("Warnings:")
        for warning in report.warnings:
            print(f"- {warning}")

    if args.write_config:
        out_path = Path(args.write_config)
        tuned_cfg = _apply_report(cfg_result.config, report.recommended_switching)
        saved = write_config(tuned_cfg, out_path)
        print(f"Wrote tuned config: {saved}")

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                {
                    "probe_summary": summary,
                    "recommended_switching": report.recommended_switching,
                    "changes": {
                        key: {
                            "old": change.old,
                            "new": change.new,
                            "reason": change.reason,
                        }
                        for key, change in report.changes.items()
                    },
                    "notes": report.notes,
                    "warnings": report.warnings,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote tuning report: {json_path}")


if __name__ == "__main__":
    main()
