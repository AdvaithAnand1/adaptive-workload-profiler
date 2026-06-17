"""Unified safe terminal interface for PerfAnalyze power plans.

This CLI intentionally exposes only catalog file operations, read-only Windows
discovery, and dry-run command previews. It cannot execute powercfg mutations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

from powerplans.catalog import PORTABLE_SETTINGS_BY_KEY
from powerplans.models import PlanDefinition, default_power_plan_catalog
from powerplans.parser import DiscoveredPowerSetting
from powerplans.service import ManagedPlanOperation, ManagedPowerPlanService
from powerplans.store import (
    POWER_PLAN_CATALOG_FILE,
    load_power_plan_catalog,
    write_power_plan_catalog,
)
from powerplans.windows import PowerCfgDiscovery


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect and preview PerfAnalyze-managed Windows power plans. "
            "No command in this interface applies Windows changes."
        )
    )
    parser.add_argument(
        "--catalog",
        default=POWER_PLAN_CATALOG_FILE,
        help=f"Power-plan catalog path (default: {POWER_PLAN_CATALOG_FILE}).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Show catalog and Windows scheme status.")

    catalog = subparsers.add_parser("catalog", help="Manage the local JSON catalog.")
    catalog_sub = catalog.add_subparsers(dest="catalog_command", required=True)
    init = catalog_sub.add_parser(
        "init",
        help="Write the default catalog JSON without changing Windows.",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing catalog file.",
    )

    plans = subparsers.add_parser("plans", help="Inspect saved plan definitions.")
    plans_sub = plans.add_subparsers(dest="plans_command", required=True)
    plans_sub.add_parser("list", help="List saved plans.")
    show = plans_sub.add_parser("show", help="Show one saved plan.")
    show.add_argument("plan_id")

    subparsers.add_parser(
        "mappings",
        help="List classification-to-plan mappings.",
    )

    windows = subparsers.add_parser(
        "windows",
        help="Read-only Windows powercfg discovery.",
    )
    windows_sub = windows.add_subparsers(dest="windows_command", required=True)
    windows_sub.add_parser("schemes", help="List installed Windows schemes.")
    settings = windows_sub.add_parser(
        "settings",
        help="Show supported portable settings.",
    )
    settings.add_argument(
        "--scheme",
        default="SCHEME_CURRENT",
        help="Scheme alias or GUID to inspect.",
    )
    settings.add_argument(
        "--all",
        action="store_true",
        help="Include non-portable discovered settings.",
    )

    preview = subparsers.add_parser(
        "preview",
        help="Generate powercfg commands without executing them.",
    )
    preview_sub = preview.add_subparsers(dest="preview_command", required=True)
    for operation in ("create", "update", "activate", "delete"):
        operation_parser = preview_sub.add_parser(
            operation,
            help=f"Preview {operation} for one saved plan.",
        )
        operation_parser.add_argument("plan_id")
        if operation == "create":
            operation_parser.add_argument(
                "--destination-guid",
                default=None,
                help="Optional deterministic GUID for the preview.",
            )

    return parser


def _load_plan(catalog_path: str, plan_id: str) -> PlanDefinition:
    catalog = load_power_plan_catalog(catalog_path)
    plan = catalog.plans.get(plan_id)
    if plan is None:
        valid = ", ".join(sorted(catalog.plans))
        raise RuntimeError(f"Unknown plan '{plan_id}'. Available: {valid}")
    return plan


def _format_setting_value(
    setting: DiscoveredPowerSetting,
    value: int | None,
) -> str:
    if value is None:
        return "-"
    options = {option.index: option.name for option in setting.options}
    if value in options:
        return f"{value} ({options[value]})"
    suffix = f" {setting.units}" if setting.units else ""
    return f"{value}{suffix}"


def _print_operation(operation: ManagedPlanOperation, out: TextIO) -> None:
    print("DRY RUN ONLY - no commands were executed", file=out)
    print(
        f"Operation: {operation.operation} | "
        f"plan={operation.plan_id} | guid={operation.target_guid}",
        file=out,
    )
    for index, command in enumerate(operation.render(), start=1):
        print(f"{index}. {command}", file=out)
    for warning in operation.warnings:
        print(f"WARNING: {warning}", file=out)


def _status(catalog_path: str, discovery: PowerCfgDiscovery, out: TextIO) -> None:
    catalog = load_power_plan_catalog(catalog_path)
    schemes = discovery.list_schemes()
    active = next((scheme for scheme in schemes if scheme.active), None)
    installed = {scheme.guid: scheme for scheme in schemes}
    attached = [
        plan
        for plan in catalog.plans.values()
        if plan.windows_guid and plan.windows_guid in installed
    ]
    stale = [
        plan
        for plan in catalog.plans.values()
        if plan.windows_guid and plan.windows_guid not in installed
    ]

    print(f"Catalog: {Path(catalog_path).expanduser()}", file=out)
    print(
        f"Saved plans: {len(catalog.plans)} | "
        f"mappings: {len(catalog.classification_to_plan)}",
        file=out,
    )
    if active is None:
        print("Active Windows scheme: unavailable", file=out)
    else:
        print(
            f"Active Windows scheme: {active.name} [{active.guid}]",
            file=out,
        )
    print(
        f"Installed schemes: {len(schemes)} | "
        f"attached managed plans: {len(attached)} | stale GUIDs: {len(stale)}",
        file=out,
    )


def _catalog_init(catalog_path: str, force: bool, out: TextIO) -> None:
    path = Path(catalog_path).expanduser()
    if path.exists() and not force:
        raise RuntimeError(
            f"Catalog '{path}' already exists. Use --force to replace it."
        )
    written = write_power_plan_catalog(default_power_plan_catalog(), path)
    print(f"Wrote default catalog: {written}", file=out)
    print("Windows power settings were not changed.", file=out)


def _plans_list(catalog_path: str, out: TextIO) -> None:
    catalog = load_power_plan_catalog(catalog_path)
    print("ID                 CPU  GPU  Power  Windows GUID  Name", file=out)
    for plan_id, plan in sorted(catalog.plans.items()):
        guid = plan.windows_guid or "-"
        print(
            f"{plan_id:<18} {plan.cpu_intent:>3}  "
            f"{plan.graphics_intent:>3}  {plan.total_power_level:>5}  "
            f"{guid:<12}  {plan.name}",
            file=out,
        )


def _plans_show(catalog_path: str, plan_id: str, out: TextIO) -> None:
    plan = _load_plan(catalog_path, plan_id)
    print(f"Plan: {plan.name} ({plan.plan_id})", file=out)
    print(
        f"Intent: CPU={plan.cpu_intent} graphics={plan.graphics_intent} "
        f"total={plan.total_power_level}",
        file=out,
    )
    print(f"Enabled: {plan.enabled}", file=out)
    print(f"Base scheme: {plan.base_scheme}", file=out)
    print(f"Windows GUID: {plan.windows_guid or '-'}", file=out)
    print(f"Description: {plan.description or '-'}", file=out)
    print("Settings:", file=out)
    for key, value in sorted(plan.settings.items()):
        spec = PORTABLE_SETTINGS_BY_KEY[key]
        print(
            f"  {spec.label}: AC={value.ac if value.ac is not None else '-'} "
            f"DC={value.dc if value.dc is not None else '-'}",
            file=out,
        )


def _mappings(catalog_path: str, out: TextIO) -> None:
    catalog = load_power_plan_catalog(catalog_path)
    for label, plan_id in sorted(catalog.classification_to_plan.items()):
        plan = catalog.plans[plan_id]
        print(f"{label:<20} -> {plan_id:<18} ({plan.name})", file=out)


def _windows_schemes(discovery: PowerCfgDiscovery, out: TextIO) -> None:
    for scheme in discovery.list_schemes():
        marker = "*" if scheme.active else " "
        print(f"{marker} {scheme.name} [{scheme.guid}]", file=out)


def _windows_settings(
    discovery: PowerCfgDiscovery,
    *,
    scheme: str,
    show_all: bool,
    out: TextIO,
) -> None:
    snapshot = discovery.discover_settings(scheme)
    selected = [
        setting
        for setting in snapshot.settings
        if show_all or setting.key.upper() in PORTABLE_SETTINGS_BY_KEY
    ]
    print(f"Scheme: {snapshot.scheme_name} [{snapshot.scheme_guid}]", file=out)
    print(
        f"Showing {len(selected)} of {len(snapshot.settings)} discovered settings",
        file=out,
    )
    for setting in selected:
        spec = PORTABLE_SETTINGS_BY_KEY.get(setting.key.upper())
        label = spec.label if spec is not None else setting.setting_name
        domain = spec.domain if spec is not None else setting.domain
        print(f"- [{domain}] {label}", file=out)
        print(f"  key={setting.key}", file=out)
        print(
            f"  AC={_format_setting_value(setting, setting.ac_value)} | "
            f"DC={_format_setting_value(setting, setting.dc_value)}",
            file=out,
        )


def _preview(
    catalog_path: str,
    discovery: PowerCfgDiscovery,
    *,
    operation_name: str,
    plan_id: str,
    destination_guid: str | None,
    out: TextIO,
) -> None:
    plan = _load_plan(catalog_path, plan_id)
    service = ManagedPowerPlanService(discovery=discovery)
    schemes = discovery.list_schemes()

    if operation_name == "create":
        snapshot = discovery.discover_settings()
        operation = service.build_create(
            plan,
            schemes=schemes,
            snapshot=snapshot,
            destination_guid=destination_guid,
        )
    elif operation_name == "update":
        if not plan.windows_guid:
            raise RuntimeError(
                f"Plan '{plan.plan_id}' has no Windows GUID; preview create first."
            )
        snapshot = discovery.discover_settings(plan.windows_guid)
        operation = service.build_update(
            plan,
            schemes=schemes,
            snapshot=snapshot,
        )
    elif operation_name == "activate":
        if not plan.windows_guid:
            raise RuntimeError(
                f"Plan '{plan.plan_id}' has no Windows GUID; create it first."
            )
        operation = service.build_activate_managed(plan, schemes=schemes)
    elif operation_name == "delete":
        operation = service.build_delete(plan, schemes=schemes)
    else:  # pragma: no cover - argparse constrains this
        raise RuntimeError(f"Unknown preview operation '{operation_name}'.")

    _print_operation(operation, out)


def run(
    argv: list[str] | None = None,
    *,
    discovery: PowerCfgDiscovery | None = None,
    out: TextIO | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    output = out or sys.stdout
    windows = discovery or PowerCfgDiscovery()

    if args.command == "catalog":
        _catalog_init(args.catalog, args.force, output)
    elif args.command == "plans":
        if args.plans_command == "list":
            _plans_list(args.catalog, output)
        else:
            _plans_show(args.catalog, args.plan_id, output)
    elif args.command == "mappings":
        _mappings(args.catalog, output)
    elif args.command == "status":
        _status(args.catalog, windows, output)
    elif args.command == "windows":
        if args.windows_command == "schemes":
            _windows_schemes(windows, output)
        else:
            _windows_settings(
                windows,
                scheme=args.scheme,
                show_all=args.all,
                out=output,
            )
    elif args.command == "preview":
        _preview(
            args.catalog,
            windows,
            operation_name=args.preview_command,
            plan_id=args.plan_id,
            destination_guid=getattr(args, "destination_guid", None),
            out=output,
        )
    return 0


def main() -> int:
    try:
        return run()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
