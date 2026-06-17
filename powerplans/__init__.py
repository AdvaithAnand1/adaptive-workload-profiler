"""Power-plan domain models and read-only Windows discovery."""

from powerplans.catalog import (
    PORTABLE_SETTING_KEYS,
    PORTABLE_SETTING_SPECS,
    PortableSettingSpec,
)
from powerplans.commands import (
    PowerCfgCommand,
    PowerCfgCommandResult,
    PowerCfgExecutor,
)
from powerplans.models import (
    CATALOG_SCHEMA_VERSION,
    PlanDefinition,
    PlanSettingValue,
    PowerPlanCatalog,
    default_power_plan_catalog,
)
from powerplans.editor_state import PowerPlanEditorState
from powerplans.parser import (
    DiscoveredPowerSetting,
    PowerSettingGroup,
    PowerSettingsSnapshot,
    WindowsPowerScheme,
    parse_power_schemes,
    parse_power_settings,
)
from powerplans.store import load_power_plan_catalog, write_power_plan_catalog
from powerplans.service import (
    ManagedPlanExecution,
    ManagedPlanOperation,
    ManagedPowerPlanService,
)

__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "DiscoveredPowerSetting",
    "ManagedPlanExecution",
    "ManagedPlanOperation",
    "ManagedPowerPlanService",
    "PORTABLE_SETTING_KEYS",
    "PORTABLE_SETTING_SPECS",
    "PlanDefinition",
    "PlanSettingValue",
    "PowerCfgCommand",
    "PowerCfgCommandResult",
    "PowerCfgExecutor",
    "PortableSettingSpec",
    "PowerPlanCatalog",
    "PowerPlanEditorState",
    "PowerSettingGroup",
    "PowerSettingsSnapshot",
    "WindowsPowerScheme",
    "default_power_plan_catalog",
    "load_power_plan_catalog",
    "parse_power_schemes",
    "parse_power_settings",
    "write_power_plan_catalog",
]
