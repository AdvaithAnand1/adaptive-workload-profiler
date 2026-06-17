"""Portable data model for user-managed Windows power plans."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from powerplans.catalog import PORTABLE_SETTING_KEYS

CATALOG_SCHEMA_VERSION = 2
INTENT_MIN = 0
INTENT_MAX = 4
_PLAN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class PlanSettingValue:
    """Desired AC/DC indexes for one Windows power setting."""

    ac: int | None = None
    dc: int | None = None

    def __post_init__(self):
        if self.ac is None and self.dc is None:
            raise ValueError("A plan setting must define at least one of AC or DC.")
        for source, value in (("ac", self.ac), ("dc", self.dc)):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"Plan setting '{source}' must be a non-negative integer.")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PlanSettingValue":
        if not isinstance(raw, Mapping):
            raise ValueError("Plan setting values must be an object.")
        return cls(ac=raw.get("ac"), dc=raw.get("dc"))

    def to_dict(self) -> dict[str, int]:
        result: dict[str, int] = {}
        if self.ac is not None:
            result["ac"] = self.ac
        if self.dc is not None:
            result["dc"] = self.dc
        return result


@dataclass(frozen=True)
class PlanDefinition:
    """A named plan with independent CPU, graphics, and total-power intent."""

    plan_id: str
    name: str
    cpu_intent: int
    graphics_intent: int
    total_power_level: int
    enabled: bool = True
    base_scheme: str = "SCHEME_BALANCED"
    windows_guid: str | None = None
    description: str = ""
    settings: dict[str, PlanSettingValue] = field(default_factory=dict)

    def __post_init__(self):
        if not _PLAN_ID_RE.fullmatch(self.plan_id):
            raise ValueError(
                "Plan id must start with a lowercase letter or number and contain "
                "only lowercase letters, numbers, underscores, or hyphens."
            )
        if not self.name.strip():
            raise ValueError(f"Plan '{self.plan_id}' must have a non-empty name.")
        for field_name, value in (
            ("cpu_intent", self.cpu_intent),
            ("graphics_intent", self.graphics_intent),
            ("total_power_level", self.total_power_level),
        ):
            if type(value) is not int or not INTENT_MIN <= value <= INTENT_MAX:
                raise ValueError(
                    f"Plan '{self.plan_id}' {field_name} must be between "
                    f"{INTENT_MIN} and {INTENT_MAX}."
                )
        if not isinstance(self.enabled, bool):
            raise ValueError(f"Plan '{self.plan_id}' enabled must be true or false.")
        if not self.base_scheme.strip():
            raise ValueError(f"Plan '{self.plan_id}' must define a base scheme.")
        for key, value in self.settings.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"Plan '{self.plan_id}' has an invalid setting key.")
            if key not in PORTABLE_SETTING_KEYS:
                raise ValueError(
                    f"Plan '{self.plan_id}' setting '{key}' is not in the "
                    "portable Windows setting catalog."
                )
            if not isinstance(value, PlanSettingValue):
                raise ValueError(
                    f"Plan '{self.plan_id}' setting '{key}' must be PlanSettingValue."
                )

    @classmethod
    def from_dict(
        cls,
        plan_id: str,
        raw: Mapping[str, Any],
    ) -> "PlanDefinition":
        if not isinstance(raw, Mapping):
            raise ValueError(f"Plan '{plan_id}' must be an object.")
        raw_settings = raw.get("settings", {})
        if not isinstance(raw_settings, Mapping):
            raise ValueError(f"Plan '{plan_id}' settings must be an object.")
        settings = {
            str(key): PlanSettingValue.from_dict(value)
            for key, value in raw_settings.items()
        }
        return cls(
            plan_id=plan_id,
            name=str(raw.get("name", "")).strip(),
            cpu_intent=raw.get("cpu_intent"),
            graphics_intent=raw.get("graphics_intent"),
            total_power_level=raw.get("total_power_level"),
            enabled=raw.get("enabled", True),
            base_scheme=str(raw.get("base_scheme", "SCHEME_BALANCED")).strip(),
            windows_guid=(
                str(raw["windows_guid"]).strip()
                if raw.get("windows_guid")
                else None
            ),
            description=str(raw.get("description", "")).strip(),
            settings=settings,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cpu_intent": self.cpu_intent,
            "graphics_intent": self.graphics_intent,
            "total_power_level": self.total_power_level,
            "enabled": self.enabled,
            "base_scheme": self.base_scheme,
            "windows_guid": self.windows_guid,
            "description": self.description,
            "settings": {
                key: value.to_dict()
                for key, value in sorted(self.settings.items())
            },
        }


@dataclass(frozen=True)
class PowerPlanCatalog:
    """Unlimited named plans plus classification-to-plan assignments."""

    plans: dict[str, PlanDefinition]
    classification_to_plan: dict[str, str]
    schema_version: int = CATALOG_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != CATALOG_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported power-plan schema version {self.schema_version}; "
                f"expected {CATALOG_SCHEMA_VERSION}."
            )
        if not self.plans:
            raise ValueError("Power-plan catalog must contain at least one plan.")
        for key, plan in self.plans.items():
            if key != plan.plan_id:
                raise ValueError(
                    f"Plan key '{key}' does not match embedded id '{plan.plan_id}'."
                )
        for raw_label, plan_id in self.classification_to_plan.items():
            label = raw_label.strip().lower()
            if not label:
                raise ValueError("Classification names cannot be empty.")
            if raw_label != label:
                raise ValueError(
                    f"Classification '{raw_label}' must be normalized to '{label}'."
                )
            if plan_id not in self.plans:
                raise ValueError(
                    f"Classification '{label}' references unknown plan '{plan_id}'."
                )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PowerPlanCatalog":
        if not isinstance(raw, Mapping):
            raise ValueError("Power-plan catalog root must be an object.")
        raw_plans = raw.get("plans")
        if not isinstance(raw_plans, Mapping):
            raise ValueError("'plans' must be an object.")
        raw_mapping = raw.get("classification_to_plan", {})
        if not isinstance(raw_mapping, Mapping):
            raise ValueError("'classification_to_plan' must be an object.")
        plans = {
            str(plan_id): PlanDefinition.from_dict(str(plan_id), plan_raw)
            for plan_id, plan_raw in raw_plans.items()
        }
        mapping = {
            str(label).strip().lower(): str(plan_id).strip()
            for label, plan_id in raw_mapping.items()
        }
        return cls(
            schema_version=raw.get("schema_version", CATALOG_SCHEMA_VERSION),
            plans=plans,
            classification_to_plan=mapping,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plans": {
                plan_id: plan.to_dict()
                for plan_id, plan in sorted(self.plans.items())
            },
            "classification_to_plan": dict(
                sorted(self.classification_to_plan.items())
            ),
        }


def default_power_plan_catalog() -> PowerPlanCatalog:
    max_cpu = "SUB_PROCESSOR/PROCTHROTTLEMAX"
    boost_mode = "SUB_PROCESSOR/PERFBOOSTMODE"
    epp = "SUB_PROCESSOR/PERFEPP"
    gpu_policy = "SUB_GRAPHICS/GPUPREFERENCEPOLICY"
    pcie_aspm = "SUB_PCIEXPRESS/ASPM"

    plans = {
        "quiet": PlanDefinition(
            plan_id="quiet",
            name="PerfAnalyze Quiet",
            cpu_intent=0,
            graphics_intent=0,
            total_power_level=0,
            description="Low-noise, low-power operation for idle workloads.",
            settings={
                max_cpu: PlanSettingValue(ac=60, dc=45),
                boost_mode: PlanSettingValue(ac=0, dc=0),
                epp: PlanSettingValue(ac=75, dc=90),
                gpu_policy: PlanSettingValue(ac=1, dc=1),
                pcie_aspm: PlanSettingValue(ac=2, dc=2),
            },
        ),
        "everyday": PlanDefinition(
            plan_id="everyday",
            name="PerfAnalyze Everyday",
            cpu_intent=1,
            graphics_intent=1,
            total_power_level=1,
            description="Responsive general-purpose operation.",
            settings={
                max_cpu: PlanSettingValue(ac=100, dc=85),
                boost_mode: PlanSettingValue(ac=1, dc=0),
                epp: PlanSettingValue(ac=35, dc=70),
                gpu_policy: PlanSettingValue(ac=0, dc=1),
                pcie_aspm: PlanSettingValue(ac=1, dc=2),
            },
        ),
        "cpu_focus": PlanDefinition(
            plan_id="cpu_focus",
            name="PerfAnalyze CPU Focus",
            cpu_intent=4,
            graphics_intent=1,
            total_power_level=3,
            description="Prioritize CPU throughput while preserving package headroom.",
            settings={
                max_cpu: PlanSettingValue(ac=100, dc=90),
                boost_mode: PlanSettingValue(ac=2, dc=1),
                epp: PlanSettingValue(ac=10, dc=35),
                gpu_policy: PlanSettingValue(ac=1, dc=1),
                pcie_aspm: PlanSettingValue(ac=1, dc=2),
            },
        ),
        "graphics_focus": PlanDefinition(
            plan_id="graphics_focus",
            name="PerfAnalyze Graphics Focus",
            cpu_intent=2,
            graphics_intent=4,
            total_power_level=3,
            description="Prioritize graphics work using portable Windows controls.",
            settings={
                max_cpu: PlanSettingValue(ac=85, dc=75),
                boost_mode: PlanSettingValue(ac=1, dc=0),
                epp: PlanSettingValue(ac=35, dc=60),
                gpu_policy: PlanSettingValue(ac=0, dc=0),
                pcie_aspm: PlanSettingValue(ac=0, dc=1),
            },
        ),
        "maximum_mixed": PlanDefinition(
            plan_id="maximum_mixed",
            name="PerfAnalyze Maximum Mixed",
            cpu_intent=4,
            graphics_intent=4,
            total_power_level=4,
            description="High CPU and graphics performance for mixed workloads.",
            settings={
                max_cpu: PlanSettingValue(ac=100, dc=90),
                boost_mode: PlanSettingValue(ac=2, dc=1),
                epp: PlanSettingValue(ac=0, dc=25),
                gpu_policy: PlanSettingValue(ac=0, dc=0),
                pcie_aspm: PlanSettingValue(ac=0, dc=1),
            },
        ),
    }
    return PowerPlanCatalog(
        plans=plans,
        classification_to_plan={
            "idle": "quiet",
            "light": "everyday",
            "browsing": "everyday",
            "compiling": "cpu_focus",
            "cpu_rendering": "cpu_focus",
            "gaming": "graphics_focus",
            "gpu_rendering": "graphics_focus",
            "heavy": "maximum_mixed",
            "rendering": "maximum_mixed",
        },
    )


def migrate_legacy_label_mapping(
    legacy_mapping: Mapping[str, str],
) -> dict[str, str]:
    """Translate the existing silent/balanced/performance mapping to plan IDs."""

    profile_to_plan = {
        "silent": "quiet",
        "balanced": "everyday",
        "performance": "maximum_mixed",
    }
    result: dict[str, str] = {}
    for raw_label, raw_profile in legacy_mapping.items():
        label = str(raw_label).strip().lower()
        profile = str(raw_profile).strip().lower()
        if not label:
            continue
        result[label] = profile_to_plan.get(profile, "everyday")
    return result
