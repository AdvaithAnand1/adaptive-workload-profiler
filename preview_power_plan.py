"""Preview managed-plan powercfg commands without executing them."""

from __future__ import annotations

import argparse

from powerplans.service import ManagedPowerPlanService
from powerplans.store import POWER_PLAN_CATALOG_FILE, load_power_plan_catalog


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a dry-run command preview for one saved power plan. "
            "This command never applies changes."
        )
    )
    parser.add_argument("plan_id", help="Plan id from the power-plan catalog.")
    parser.add_argument(
        "--catalog",
        default=POWER_PLAN_CATALOG_FILE,
        help=f"Catalog path (default: {POWER_PLAN_CATALOG_FILE}).",
    )
    parser.add_argument(
        "--destination-guid",
        default=None,
        help="Optional deterministic GUID for create previews.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    catalog = load_power_plan_catalog(args.catalog)
    plan = catalog.plans.get(args.plan_id)
    if plan is None:
        valid = ", ".join(sorted(catalog.plans))
        raise RuntimeError(f"Unknown plan '{args.plan_id}'. Available: {valid}")

    service = ManagedPowerPlanService()
    schemes = service.discovery.list_schemes()
    snapshot = service.discovery.discover_settings()
    operation = service.build_create(
        plan,
        schemes=schemes,
        snapshot=snapshot,
        destination_guid=args.destination_guid,
    )

    print("DRY RUN ONLY - no commands were executed")
    print(f"Plan: {plan.name} ({plan.plan_id})")
    print(f"Generated Windows GUID: {operation.target_guid}")
    for index, command in enumerate(operation.render(), start=1):
        print(f"{index}. {command}")
    for warning in operation.warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
