"""Read-only inspection of portable Windows power settings.

Examples:
    python inspect_power_settings.py
    python inspect_power_settings.py --all
    python inspect_power_settings.py --json
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from powerplans.catalog import PORTABLE_SETTING_KEYS, PORTABLE_SETTINGS_BY_KEY
from powerplans.parser import DiscoveredPowerSetting
from powerplans.windows import PowerCfgDiscovery


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect supported Windows power settings without changing them."
    )
    parser.add_argument(
        "--scheme",
        default="SCHEME_CURRENT",
        help="Scheme alias or GUID to inspect (default: SCHEME_CURRENT).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show every discovered setting, including non-portable extensions.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    return parser


def _is_core_setting(setting: DiscoveredPowerSetting) -> bool:
    return setting.key.upper() in PORTABLE_SETTING_KEYS


def _portable_domain(setting: DiscoveredPowerSetting) -> str:
    spec = PORTABLE_SETTINGS_BY_KEY.get(setting.key.upper())
    return spec.domain if spec is not None else setting.domain


def _setting_dict(setting: DiscoveredPowerSetting) -> dict[str, Any]:
    return {
        "key": setting.key,
        "domain": _portable_domain(setting),
        "group_name": setting.group_name,
        "group_guid": setting.group_guid,
        "group_alias": setting.group_alias,
        "name": setting.setting_name,
        "guid": setting.setting_guid,
        "alias": setting.setting_alias,
        "minimum": setting.minimum,
        "maximum": setting.maximum,
        "increment": setting.increment,
        "units": setting.units,
        "options": [
            {"index": option.index, "name": option.name}
            for option in setting.options
        ],
        "ac_value": setting.ac_value,
        "dc_value": setting.dc_value,
    }


def _format_value(setting: DiscoveredPowerSetting, value: int | None) -> str:
    if value is None:
        return "-"
    option_names = {option.index: option.name for option in setting.options}
    if value in option_names:
        return f"{value} ({option_names[value]})"
    suffix = f" {setting.units}" if setting.units else ""
    return f"{value}{suffix}"


def main() -> int:
    args = _build_parser().parse_args()
    discovery = PowerCfgDiscovery()
    if not discovery.is_supported():
        raise RuntimeError("Windows powercfg is not available on this system.")

    schemes = discovery.list_schemes()
    snapshot = discovery.discover_settings(args.scheme)
    selected = [
        setting
        for setting in snapshot.settings
        if args.all or _is_core_setting(setting)
    ]
    if not args.all:
        domain_order = {"cpu": 0, "graphics": 1, "system": 2}
        selected.sort(
            key=lambda setting: (
                domain_order.get(_portable_domain(setting), 99),
                setting.setting_name.lower(),
            )
        )

    if args.json:
        print(
            json.dumps(
                {
                    "scheme": {
                        "guid": snapshot.scheme_guid,
                        "name": snapshot.scheme_name,
                    },
                    "installed_schemes": [
                        {
                            "guid": scheme.guid,
                            "name": scheme.name,
                            "active": scheme.active,
                        }
                        for scheme in schemes
                    ],
                    "setting_count": len(snapshot.settings),
                    "domain_counts": {
                        domain: len(snapshot.by_domain(domain))
                        for domain in ("cpu", "graphics", "other")
                    },
                    "portable_setting_count": len(
                        [setting for setting in snapshot.settings if _is_core_setting(setting)]
                    ),
                    "portable_domain_counts": {
                        domain: len(
                            [
                                setting
                                for setting in snapshot.settings
                                if _is_core_setting(setting)
                                and _portable_domain(setting) == domain
                            ]
                        )
                        for domain in ("cpu", "graphics", "system")
                    },
                    "settings": [_setting_dict(setting) for setting in selected],
                },
                indent=2,
            )
        )
        return 0

    print(f"Scheme: {snapshot.scheme_name} [{snapshot.scheme_guid}]")
    print(f"Discovered: {len(snapshot.settings)} settings in {len(snapshot.groups)} groups")
    print(
        "Domains: "
        + ", ".join(
            f"{domain}={len(snapshot.by_domain(domain))}"
            for domain in ("cpu", "graphics", "other")
        )
    )
    print(
        f"Showing: {len(selected)} "
        f"{'all discovered' if args.all else 'portable Windows'} settings"
    )

    current_domain = None
    for setting in selected:
        display_domain = _portable_domain(setting)
        if display_domain != current_domain:
            current_domain = display_domain
            print(f"\n{current_domain.upper()}")
        print(f"- {setting.setting_name}")
        print(f"  key={setting.key}")
        print(
            f"  AC={_format_value(setting, setting.ac_value)} | "
            f"DC={_format_value(setting, setting.dc_value)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
