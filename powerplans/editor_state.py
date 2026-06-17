"""Mutable editor state for the power-plan catalog."""

from __future__ import annotations

from pathlib import Path

from powerplans.models import (
    PlanDefinition,
    PowerPlanCatalog,
    default_power_plan_catalog,
)
from powerplans.store import (
    POWER_PLAN_CATALOG_FILE,
    load_power_plan_catalog,
    write_power_plan_catalog,
)


class PowerPlanEditorState:
    """Owns draft catalog edits without applying anything to Windows."""

    def __init__(
        self,
        catalog: PowerPlanCatalog,
        path: str | Path = POWER_PLAN_CATALOG_FILE,
    ):
        self.path = Path(path).expanduser()
        self.catalog = catalog
        self.dirty = False

    @classmethod
    def load(
        cls,
        path: str | Path = POWER_PLAN_CATALOG_FILE,
    ) -> "PowerPlanEditorState":
        return cls(load_power_plan_catalog(path), path)

    @property
    def plan_ids(self) -> tuple[str, ...]:
        return tuple(self.catalog.plans)

    def add_plan(self) -> PlanDefinition:
        index = 1
        while f"custom_{index}" in self.catalog.plans:
            index += 1
        plan_id = f"custom_{index}"
        plan = PlanDefinition(
            plan_id=plan_id,
            name=f"Custom Plan {index}",
            cpu_intent=1,
            graphics_intent=1,
            total_power_level=1,
            description="User-defined power plan draft.",
        )
        plans = dict(self.catalog.plans)
        plans[plan_id] = plan
        self.catalog = PowerPlanCatalog(
            plans=plans,
            classification_to_plan=dict(self.catalog.classification_to_plan),
        )
        self.dirty = True
        return plan

    def update_plan(self, plan: PlanDefinition):
        if plan.plan_id not in self.catalog.plans:
            raise KeyError(f"Unknown plan '{plan.plan_id}'.")
        if self.catalog.plans[plan.plan_id] == plan:
            return
        plans = dict(self.catalog.plans)
        plans[plan.plan_id] = plan
        self.catalog = PowerPlanCatalog(
            plans=plans,
            classification_to_plan=dict(self.catalog.classification_to_plan),
        )
        self.dirty = True

    def remove_plan(self, plan_id: str) -> tuple[str, ...]:
        if plan_id not in self.catalog.plans:
            raise KeyError(f"Unknown plan '{plan_id}'.")
        if len(self.catalog.plans) == 1:
            raise ValueError("At least one power plan must remain.")

        plans = dict(self.catalog.plans)
        del plans[plan_id]
        removed_mappings = tuple(
            sorted(
                label
                for label, mapped_plan_id in self.catalog.classification_to_plan.items()
                if mapped_plan_id == plan_id
            )
        )
        mapping = {
            label: mapped_plan_id
            for label, mapped_plan_id in self.catalog.classification_to_plan.items()
            if mapped_plan_id != plan_id
        }
        self.catalog = PowerPlanCatalog(
            plans=plans,
            classification_to_plan=mapping,
        )
        self.dirty = True
        return removed_mappings

    def reload(self):
        self.catalog = load_power_plan_catalog(self.path)
        self.dirty = False

    def restore_defaults(self):
        self.catalog = default_power_plan_catalog()
        self.dirty = True

    def save(self) -> Path:
        saved_path = write_power_plan_catalog(self.catalog, self.path)
        self.path = saved_path
        self.dirty = False
        return saved_path
