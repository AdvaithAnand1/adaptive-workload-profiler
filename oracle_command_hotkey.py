"""External hotkey oracle bridge for PerfAnalyze.

Implements the command-oracle protocol:
- set-profile <silent|balanced|performance>
- get-profile
- diagnostics

This is the easiest G-Helper-compatible path when profiles are exposed via hotkeys.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from config import CONFIG_FILE, VALID_PROFILES, load_config

DEFAULT_STATE = Path(".oracle_command_hotkey_state.json")
VALID_PROFILE_SET = set(VALID_PROFILES)


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"profile": "balanced"}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"profile": "balanced"}
    return raw if isinstance(raw, dict) else {"profile": "balanced"}


def _write_state(path: Path, state: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _normalize_profile(raw: str) -> str:
    return raw.strip().lower()


def _dry_run_enabled(args_dry_run: bool) -> bool:
    if args_dry_run:
        return True
    token = os.getenv("PERFANALYZE_HOTKEY_BACKEND_DRY_RUN", "").strip().lower()
    return token in {"1", "true", "yes", "on"}


def _hotkey_status_map(profile_hotkeys: dict[str, str]) -> dict[str, bool]:
    return {
        profile: bool(str(profile_hotkeys.get(profile, "")).strip())
        for profile in VALID_PROFILES
    }


def _send_hotkey(combo: str):
    import keyboard  # imported lazily so diagnostics/get-profile do not require it

    keyboard.send(combo)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PerfAnalyze hotkey command oracle")
    p.add_argument("command", choices=["set-profile", "get-profile", "diagnostics"])
    p.add_argument("profile", nargs="?", default="")
    p.add_argument("--config", default=CONFIG_FILE)
    p.add_argument("--state", default=str(DEFAULT_STATE))
    p.add_argument("--dry-run", action="store_true")
    return p


def main():
    args = _build_parser().parse_args()
    state_path = Path(args.state)
    state = _read_state(state_path)
    cfg_result = load_config(args.config)
    cfg = cfg_result.config
    dry_run = _dry_run_enabled(args.dry_run)
    profile_hotkeys = {
        profile: str(cfg.profile_hotkeys.get(profile, "")).strip()
        for profile in VALID_PROFILES
    }

    if args.command == "get-profile":
        profile = _normalize_profile(str(state.get("profile", "balanced"))) or "balanced"
        print(json.dumps({"profile": profile}))
        return

    if args.command == "diagnostics":
        selected = _normalize_profile(str(state.get("profile", "balanced"))) or "balanced"
        print(
            json.dumps(
                {
                    "backend_name": "oracle_command_hotkey",
                    "selected_profile": selected,
                    "details": {
                        "mode": "hotkey command bridge",
                        "config_source": cfg_result.source,
                        "config_path": str(cfg_result.path or args.config),
                        "state_path": str(state_path),
                        "dry_run": dry_run,
                        "hotkeys_ready": _hotkey_status_map(profile_hotkeys),
                    },
                }
            )
        )
        return

    profile = _normalize_profile(args.profile)
    if profile not in VALID_PROFILE_SET:
        print(
            json.dumps(
                {
                    "ok": False,
                    "applied_profile": state.get("profile", "balanced"),
                    "message": f"unsupported profile '{args.profile}'",
                }
            )
        )
        return

    hotkey = profile_hotkeys.get(profile, "")
    if not hotkey:
        print(
            json.dumps(
                {
                    "ok": False,
                    "applied_profile": state.get("profile", "balanced"),
                    "message": f"no hotkey configured for profile '{profile}'",
                }
            )
        )
        return

    if not dry_run:
        try:
            _send_hotkey(hotkey)
        except Exception as e:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "applied_profile": state.get("profile", "balanced"),
                        "message": f"hotkey send failed: {type(e).__name__}: {e}",
                    }
                )
            )
            return

    state.update(
        {
            "profile": profile,
            "last_hotkey": hotkey,
            "dry_run": dry_run,
            "config_source": cfg_result.source,
        }
    )
    _write_state(state_path, state)
    print(
        json.dumps(
            {
                "ok": True,
                "applied_profile": profile,
                "message": (
                    f"hotkey backend {'dry-run ' if dry_run else ''}applied {profile} via {hotkey}"
                ).strip(),
            }
        )
    )


if __name__ == "__main__":
    main()
