"""Simple external oracle command mock.

Examples:
    python oracle_command_mock.py diagnostics
    python oracle_command_mock.py set-profile performance
    python oracle_command_mock.py get-profile
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_STATE = Path(".oracle_command_mock_state.json")


def _read_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {"profile": "balanced"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"profile": "balanced"}
    if not isinstance(data, dict):
        return {"profile": "balanced"}
    profile = str(data.get("profile", "balanced")).strip().lower() or "balanced"
    return {"profile": profile}


def _write_state(path: Path, state: dict[str, str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mock oracle command backend for PerfAnalyze")
    p.add_argument("command", choices=["set-profile", "get-profile", "diagnostics"])
    p.add_argument("profile", nargs="?", default="")
    p.add_argument("--state", default=str(DEFAULT_STATE))
    return p


def main():
    args = _parser().parse_args()
    state_path = Path(args.state)
    state = _read_state(state_path)

    if args.command == "set-profile":
        profile = args.profile.strip().lower() or "balanced"
        state["profile"] = profile
        _write_state(state_path, state)
        print(
            json.dumps(
                {
                    "ok": True,
                    "applied_profile": profile,
                    "message": f"mock command applied {profile}",
                }
            )
        )
        return

    if args.command == "get-profile":
        print(json.dumps({"profile": state.get("profile", "balanced")}))
        return

    print(
        json.dumps(
            {
                "backend_name": "oracle_command_mock",
                "selected_profile": state.get("profile", "balanced"),
                "details": {
                    "mode": "mock external command",
                    "state_path": str(state_path),
                },
            }
        )
    )


if __name__ == "__main__":
    main()
