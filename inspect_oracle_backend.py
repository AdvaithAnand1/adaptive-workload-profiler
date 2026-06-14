"""Inspect PerfAnalyze backend selection and Windows power-plan mapping.

Examples:
    python inspect_oracle_backend.py
    set PERFANALYZE_ORACLE_BACKEND=powercfg && python inspect_oracle_backend.py
    set PERFANALYZE_ORACLE_BACKEND=command && set PERFANALYZE_ORACLE_COMMAND=python oracle_command_mock.py && python inspect_oracle_backend.py
"""

from __future__ import annotations

import json

from oracle_client import OracleClient


def main():
    client = OracleClient()
    diag = client.diagnostics()

    print(f"Backend: {diag.backend_name}")
    print(f"Selected profile: {diag.selected_profile}")

    if not diag.details:
        print("Details: none")
        return

    schemes = diag.details.get("schemes")
    if schemes:
        print("Power schemes:")
        for scheme in schemes:
            active = "*" if scheme.get("active") else " "
            mapped = scheme.get("mapped_profile") or "?"
            print(
                f"  {active} {scheme.get('name')} [{scheme.get('guid')}] -> {mapped}"
            )

    profile_tuning = diag.details.get("profile_tuning")
    if profile_tuning:
        print("Configured profile tuning:")
        for profile, tuning in profile_tuning.items():
            desc = tuning.get("description") or []
            pretty = ", ".join(desc) if desc else "none"
            print(f"  - {profile}: {pretty}")

    remaining = {
        key: value
        for key, value in diag.details.items()
        if key not in {"schemes", "profile_tuning"}
    }
    if remaining:
        print("Details JSON:")
        print(json.dumps(remaining, indent=2))


if __name__ == "__main__":
    main()
