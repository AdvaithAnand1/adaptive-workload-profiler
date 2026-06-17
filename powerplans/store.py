"""Persistence for the versioned power-plan catalog."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from powerplans.models import PowerPlanCatalog, default_power_plan_catalog

POWER_PLAN_CATALOG_FILE = "power_plan_catalog.json"


def load_power_plan_catalog(
    path: str | Path = POWER_PLAN_CATALOG_FILE,
    *,
    use_defaults_when_missing: bool = True,
) -> PowerPlanCatalog:
    catalog_path = Path(path).expanduser()
    if not catalog_path.exists():
        if use_defaults_when_missing:
            return default_power_plan_catalog()
        raise FileNotFoundError(catalog_path)
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        return PowerPlanCatalog.from_dict(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in '{catalog_path}' at line {exc.lineno}, "
            f"column {exc.colno}."
        ) from exc


def write_power_plan_catalog(
    catalog: PowerPlanCatalog,
    path: str | Path = POWER_PLAN_CATALOG_FILE,
) -> Path:
    catalog_path = Path(path).expanduser()
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(catalog.to_dict(), indent=2) + "\n"

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{catalog_path.name}.",
        suffix=".tmp",
        dir=str(catalog_path.parent),
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, catalog_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return catalog_path.resolve()
