"""Managed Windows power-plan operations with dry-run-first execution."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Protocol
from uuid import uuid4

from powerplans.commands import (
    PowerCfgCommand,
    PowerCfgCommandResult,
    PowerCfgExecutor,
)
from powerplans.models import PlanDefinition
from powerplans.parser import (
    DiscoveredPowerSetting,
    PowerSettingsSnapshot,
    WindowsPowerScheme,
)
from powerplans.windows import PowerCfgDiscovery

_GUID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")

BUILTIN_SCHEME_GUIDS = {
    "SCHEME_BALANCED": "381b4222-f694-41f0-9685-ff5bb260df2e",
    "SCHEME_MAX": "a1841308-3541-4fab-bc81-f71556f20b4a",
    "SCHEME_MIN": "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",
}
PROTECTED_SCHEME_GUIDS = frozenset(
    {
        *BUILTIN_SCHEME_GUIDS.values(),
        "e9a42b02-d5df-448d-aa00-03f14749eb61",
    }
)
MANAGED_SCHEME_PREFIX = "PerfAnalyze "


class CommandExecutor(Protocol):
    def execute(self, command: PowerCfgCommand) -> PowerCfgCommandResult:
        ...


@dataclass(frozen=True)
class ManagedPlanOperation:
    operation: str
    plan_id: str
    target_guid: str
    commands: tuple[PowerCfgCommand, ...]
    warnings: tuple[str, ...] = ()
    resulting_plan: PlanDefinition | None = None

    def render(self, executable: str = "powercfg") -> list[str]:
        return [command.render(executable) for command in self.commands]


@dataclass(frozen=True)
class ManagedPlanExecution:
    operation: ManagedPlanOperation
    results: tuple[PowerCfgCommandResult, ...]

    @property
    def executed(self) -> bool:
        return bool(self.results) and all(result.executed for result in self.results)


def _normalize_guid(raw: str, *, field: str) -> str:
    token = raw.strip().lower()
    if not _GUID_RE.fullmatch(token):
        raise ValueError(f"{field} must be a Windows scheme GUID.")
    return token


def resolve_scheme_guid(
    token: str,
    schemes: Sequence[WindowsPowerScheme],
) -> str:
    value = token.strip()
    upper = value.upper()
    if upper == "SCHEME_CURRENT":
        active = next((scheme for scheme in schemes if scheme.active), None)
        if active is None:
            raise ValueError("No active Windows power scheme was discovered.")
        return active.guid
    if upper in BUILTIN_SCHEME_GUIDS:
        return BUILTIN_SCHEME_GUIDS[upper]
    if _GUID_RE.fullmatch(value):
        return value.lower()

    exact = [
        scheme for scheme in schemes if scheme.name.strip().lower() == value.lower()
    ]
    if len(exact) == 1:
        return exact[0].guid
    if len(exact) > 1:
        raise ValueError(f"Power scheme name '{value}' is not unique.")
    raise ValueError(
        f"Could not resolve base scheme '{value}'. Use an exact name, GUID, "
        "SCHEME_CURRENT, or a built-in SCHEME_* alias."
    )


def _setting_index(snapshot: PowerSettingsSnapshot) -> dict[str, DiscoveredPowerSetting]:
    return {setting.key.upper(): setting for setting in snapshot.settings}


def managed_scheme_name(plan: PlanDefinition) -> str:
    if plan.name.casefold().startswith(MANAGED_SCHEME_PREFIX.casefold()):
        return plan.name
    return f"{MANAGED_SCHEME_PREFIX}{plan.name}"


def managed_scheme_description(plan: PlanDefinition) -> str:
    marker = f"Managed by PerfAnalyze (plan_id={plan.plan_id})."
    return f"{marker} {plan.description}".strip()


def _change_name_arguments(
    target_guid: str,
    plan: PlanDefinition,
) -> tuple[str, ...]:
    description = managed_scheme_description(plan)
    return (
        "/changename",
        target_guid,
        managed_scheme_name(plan),
        description,
    )


def _require_managed_scheme(
    plan: PlanDefinition,
    schemes: Sequence[WindowsPowerScheme],
) -> WindowsPowerScheme:
    if not plan.windows_guid:
        raise ValueError(
            f"Plan '{plan.plan_id}' has no Windows GUID; create it first."
        )
    target_guid = _normalize_guid(plan.windows_guid, field="windows_guid")
    installed = {scheme.guid: scheme for scheme in schemes}
    scheme = installed.get(target_guid)
    if scheme is None:
        raise ValueError(
            f"Managed plan '{plan.plan_id}' GUID {target_guid} is not installed."
        )
    if not scheme.name.casefold().startswith(MANAGED_SCHEME_PREFIX.casefold()):
        raise ValueError(
            f"Refusing to modify scheme '{scheme.name}': it is not marked as "
            "PerfAnalyze-managed."
        )
    return scheme


def _validate_setting_value(setting: DiscoveredPowerSetting, value: int) -> None:
    if setting.options:
        valid = {option.index for option in setting.options}
        if value not in valid:
            raise ValueError(
                f"Setting '{setting.key}' value {value} is not one of "
                f"{sorted(valid)} on this machine."
            )
        return
    if setting.minimum is not None and value < setting.minimum:
        raise ValueError(
            f"Setting '{setting.key}' value {value} is below "
            f"{setting.minimum}."
        )
    if setting.maximum is not None and value > setting.maximum:
        raise ValueError(
            f"Setting '{setting.key}' value {value} is above "
            f"{setting.maximum}."
        )
    if (
        setting.increment
        and setting.minimum is not None
        and (value - setting.minimum) % setting.increment != 0
    ):
        raise ValueError(
            f"Setting '{setting.key}' value {value} does not match increment "
            f"{setting.increment}."
        )


class ManagedPowerPlanService:
    """Generate and optionally execute managed power-plan operations."""

    def __init__(
        self,
        *,
        executor: CommandExecutor | None = None,
        discovery: PowerCfgDiscovery | None = None,
    ):
        self.executor = executor or PowerCfgExecutor()
        self.discovery = discovery or PowerCfgDiscovery()

    def capture_active_scheme(self) -> WindowsPowerScheme:
        schemes = self.discovery.list_schemes()
        active = next((scheme for scheme in schemes if scheme.active), None)
        if active is None:
            raise RuntimeError("Windows did not report an active power scheme.")
        return active

    def _setting_commands(
        self,
        plan: PlanDefinition,
        target_guid: str,
        snapshot: PowerSettingsSnapshot,
    ) -> tuple[list[PowerCfgCommand], list[str]]:
        discovered = _setting_index(snapshot)
        commands: list[PowerCfgCommand] = []
        warnings: list[str] = []

        for key, desired in sorted(plan.settings.items()):
            setting = discovered.get(key.upper())
            if setting is None:
                warnings.append(
                    f"Skipped unsupported setting '{key}' on this machine."
                )
                continue

            if desired.ac is not None:
                _validate_setting_value(setting, desired.ac)
                commands.append(
                    PowerCfgCommand(
                        arguments=(
                            "/setacvalueindex",
                            target_guid,
                            setting.group_guid,
                            setting.setting_guid,
                            str(desired.ac),
                        ),
                        description=f"Set AC value for {setting.setting_name}",
                    )
                )
            if desired.dc is not None:
                _validate_setting_value(setting, desired.dc)
                commands.append(
                    PowerCfgCommand(
                        arguments=(
                            "/setdcvalueindex",
                            target_guid,
                            setting.group_guid,
                            setting.setting_guid,
                            str(desired.dc),
                        ),
                        description=f"Set DC value for {setting.setting_name}",
                    )
                )
        return commands, warnings

    def build_create(
        self,
        plan: PlanDefinition,
        *,
        schemes: Sequence[WindowsPowerScheme],
        snapshot: PowerSettingsSnapshot,
        destination_guid: str | None = None,
    ) -> ManagedPlanOperation:
        base_guid = resolve_scheme_guid(plan.base_scheme, schemes)
        target_guid = (
            _normalize_guid(destination_guid, field="destination_guid")
            if destination_guid is not None
            else str(uuid4())
        )
        existing_guids = {scheme.guid for scheme in schemes}
        if target_guid in existing_guids:
            raise ValueError(
                f"Cannot create plan '{plan.plan_id}': GUID {target_guid} already exists."
            )

        commands = [
            PowerCfgCommand(
                arguments=("/duplicatescheme", base_guid, target_guid),
                description=f"Duplicate base scheme for {plan.name}",
            ),
            PowerCfgCommand(
                arguments=_change_name_arguments(target_guid, plan),
                description=f"Name managed plan {plan.name}",
            ),
        ]
        setting_commands, warnings = self._setting_commands(
            plan,
            target_guid,
            snapshot,
        )
        commands.extend(setting_commands)
        resulting_plan = replace(plan, windows_guid=target_guid)
        return ManagedPlanOperation(
            operation="create",
            plan_id=plan.plan_id,
            target_guid=target_guid,
            commands=tuple(commands),
            warnings=tuple(warnings),
            resulting_plan=resulting_plan,
        )

    def build_update(
        self,
        plan: PlanDefinition,
        *,
        schemes: Sequence[WindowsPowerScheme],
        snapshot: PowerSettingsSnapshot,
        allow_active: bool = False,
    ) -> ManagedPlanOperation:
        scheme = _require_managed_scheme(plan, schemes)
        if scheme.active and not allow_active:
            raise ValueError(
                f"Managed plan '{plan.plan_id}' is active. Switch away before "
                "updating it, or explicitly opt into an active-plan update."
            )
        target_guid = scheme.guid

        commands = [
            PowerCfgCommand(
                arguments=_change_name_arguments(target_guid, plan),
                description=f"Update managed plan name {plan.name}",
            )
        ]
        setting_commands, warnings = self._setting_commands(
            plan,
            target_guid,
            snapshot,
        )
        commands.extend(setting_commands)
        return ManagedPlanOperation(
            operation="update",
            plan_id=plan.plan_id,
            target_guid=target_guid,
            commands=tuple(commands),
            warnings=tuple(warnings),
            resulting_plan=plan,
        )

    def build_activate(
        self,
        *,
        plan_id: str,
        target_guid: str,
    ) -> ManagedPlanOperation:
        guid = _normalize_guid(target_guid, field="target_guid")
        return ManagedPlanOperation(
            operation="activate",
            plan_id=plan_id,
            target_guid=guid,
            commands=(
                PowerCfgCommand(
                    arguments=("/setactive", guid),
                    description=f"Activate managed plan {plan_id}",
                ),
            ),
        )

    def build_activate_managed(
        self,
        plan: PlanDefinition,
        *,
        schemes: Sequence[WindowsPowerScheme],
    ) -> ManagedPlanOperation:
        scheme = _require_managed_scheme(plan, schemes)
        return self.build_activate(
            plan_id=plan.plan_id,
            target_guid=scheme.guid,
        )

    def build_delete(
        self,
        plan: PlanDefinition,
        *,
        schemes: Sequence[WindowsPowerScheme],
    ) -> ManagedPlanOperation:
        scheme = _require_managed_scheme(plan, schemes)
        target_guid = scheme.guid
        if target_guid in PROTECTED_SCHEME_GUIDS:
            raise ValueError("Built-in Windows power schemes cannot be deleted.")

        if scheme.active:
            raise ValueError(
                f"Managed plan '{plan.plan_id}' is active and cannot be deleted."
            )
        return ManagedPlanOperation(
            operation="delete",
            plan_id=plan.plan_id,
            target_guid=target_guid,
            commands=(
                PowerCfgCommand(
                    arguments=("/delete", target_guid),
                    description=f"Delete managed plan {plan.name}",
                    destructive=True,
                ),
            ),
        )

    def execute(self, operation: ManagedPlanOperation) -> ManagedPlanExecution:
        results = tuple(
            self.executor.execute(command) for command in operation.commands
        )
        return ManagedPlanExecution(operation=operation, results=results)

    @contextmanager
    def temporary_activation(
        self,
        *,
        plan_id: str,
        target_guid: str,
    ) -> Iterator[WindowsPowerScheme]:
        """Activate temporarily and always restore the captured scheme GUID."""

        original = self.capture_active_scheme()
        target = _normalize_guid(target_guid, field="target_guid")
        try:
            if target != original.guid:
                self.execute(
                    self.build_activate(plan_id=plan_id, target_guid=target)
                )
            yield original
        finally:
            if target != original.guid:
                self.execute(
                    self.build_activate(
                        plan_id=f"restore:{original.name}",
                        target_guid=original.guid,
                    )
                )
