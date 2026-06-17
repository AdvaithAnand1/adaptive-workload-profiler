"""Parsers for read-only `powercfg` discovery output."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_GUID = r"[0-9a-fA-F-]{36}"


@dataclass(frozen=True)
class WindowsPowerScheme:
    guid: str
    name: str
    active: bool = False


@dataclass(frozen=True)
class PowerSettingOption:
    index: int
    name: str


@dataclass(frozen=True)
class DiscoveredPowerSetting:
    group_guid: str
    group_name: str
    group_alias: str | None
    setting_guid: str
    setting_name: str
    setting_alias: str | None
    minimum: int | None = None
    maximum: int | None = None
    increment: int | None = None
    units: str | None = None
    options: tuple[PowerSettingOption, ...] = ()
    ac_value: int | None = None
    dc_value: int | None = None

    @property
    def key(self) -> str:
        group = self.group_alias or self.group_guid
        setting = self.setting_alias or self.setting_guid
        return f"{group}/{setting}"

    @property
    def domain(self) -> str:
        alias = (self.group_alias or "").upper()
        name = self.group_name.lower()
        if alias == "SUB_PROCESSOR" or "processor" in name:
            return "cpu"
        if alias == "SUB_GRAPHICS" or "graphics" in name:
            return "graphics"
        if alias == "SUB_PCIEXPRESS":
            return "graphics"
        return "other"


@dataclass(frozen=True)
class PowerSettingGroup:
    guid: str
    name: str
    alias: str | None
    settings: tuple[DiscoveredPowerSetting, ...] = ()


@dataclass(frozen=True)
class PowerSettingsSnapshot:
    scheme_guid: str
    scheme_name: str
    groups: tuple[PowerSettingGroup, ...]

    @property
    def settings(self) -> tuple[DiscoveredPowerSetting, ...]:
        return tuple(setting for group in self.groups for setting in group.settings)

    def by_domain(self, domain: str) -> tuple[DiscoveredPowerSetting, ...]:
        norm = domain.strip().lower()
        return tuple(setting for setting in self.settings if setting.domain == norm)


@dataclass
class _SettingBuilder:
    guid: str
    name: str
    alias: str | None = None
    minimum: int | None = None
    maximum: int | None = None
    increment: int | None = None
    units: str | None = None
    options: list[PowerSettingOption] = field(default_factory=list)
    pending_option_index: int | None = None
    ac_value: int | None = None
    dc_value: int | None = None


@dataclass
class _GroupBuilder:
    guid: str
    name: str
    alias: str | None = None
    settings: list[DiscoveredPowerSetting] = field(default_factory=list)


def _parse_number(raw: str) -> int:
    token = raw.strip()
    return int(token, 16) if token.lower().startswith("0x") else int(token, 10)


def parse_power_schemes(output: str) -> list[WindowsPowerScheme]:
    pattern = re.compile(
        rf"Power Scheme GUID:\s*({_GUID})\s*\((.*?)\)\s*(\*)?",
        re.IGNORECASE,
    )
    schemes: list[WindowsPowerScheme] = []
    for line in output.splitlines():
        match = pattern.search(line)
        if match:
            schemes.append(
                WindowsPowerScheme(
                    guid=match.group(1).lower(),
                    name=match.group(2).strip(),
                    active=bool(match.group(3)),
                )
            )
    return schemes


def parse_power_settings(output: str) -> PowerSettingsSnapshot:
    scheme_guid = ""
    scheme_name = ""
    groups: list[PowerSettingGroup] = []
    group: _GroupBuilder | None = None
    setting: _SettingBuilder | None = None

    scheme_pattern = re.compile(
        rf"Power Scheme GUID:\s*({_GUID})\s*\((.*?)\)",
        re.IGNORECASE,
    )
    group_pattern = re.compile(
        rf"Subgroup GUID:\s*({_GUID})(?:\s*\((.*?)\))?",
        re.IGNORECASE,
    )
    setting_pattern = re.compile(
        rf"Power Setting GUID:\s*({_GUID})(?:\s*\((.*?)\))?",
        re.IGNORECASE,
    )

    def finish_setting():
        nonlocal setting
        if group is None or setting is None:
            return
        group.settings.append(
            DiscoveredPowerSetting(
                group_guid=group.guid,
                group_name=group.name,
                group_alias=group.alias,
                setting_guid=setting.guid,
                setting_name=setting.name,
                setting_alias=setting.alias,
                minimum=setting.minimum,
                maximum=setting.maximum,
                increment=setting.increment,
                units=setting.units,
                options=tuple(setting.options),
                ac_value=setting.ac_value,
                dc_value=setting.dc_value,
            )
        )
        setting = None

    def finish_group():
        nonlocal group
        finish_setting()
        if group is not None:
            groups.append(
                PowerSettingGroup(
                    guid=group.guid,
                    name=group.name,
                    alias=group.alias,
                    settings=tuple(group.settings),
                )
            )
        group = None

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = scheme_pattern.search(line)
        if match and not scheme_guid:
            scheme_guid = match.group(1).lower()
            scheme_name = match.group(2).strip()
            continue

        match = group_pattern.search(line)
        if match:
            finish_group()
            group = _GroupBuilder(
                guid=match.group(1).lower(),
                name=(match.group(2) or match.group(1)).strip(),
            )
            continue

        match = setting_pattern.search(line)
        if match:
            finish_setting()
            if group is None:
                raise ValueError("Encountered a power setting before its subgroup.")
            setting = _SettingBuilder(
                guid=match.group(1).lower(),
                name=(match.group(2) or match.group(1)).strip(),
            )
            continue

        if line.startswith("GUID Alias:"):
            alias = line.split(":", 1)[1].strip()
            if setting is not None:
                setting.alias = alias
            elif group is not None:
                group.alias = alias
            continue

        if setting is None:
            continue

        value_fields = {
            "Minimum Possible Setting:": "minimum",
            "Maximum Possible Setting:": "maximum",
            "Possible Settings increment:": "increment",
            "Current AC Power Setting Index:": "ac_value",
            "Current DC Power Setting Index:": "dc_value",
        }
        matched_value = False
        for prefix, field_name in value_fields.items():
            if line.startswith(prefix):
                setattr(setting, field_name, _parse_number(line.split(":", 1)[1]))
                matched_value = True
                break
        if matched_value:
            continue

        if line.startswith("Possible Settings units:"):
            setting.units = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Possible Setting Index:"):
            setting.pending_option_index = _parse_number(line.split(":", 1)[1])
            continue
        if line.startswith("Possible Setting Friendly Name:"):
            if setting.pending_option_index is not None:
                setting.options.append(
                    PowerSettingOption(
                        index=setting.pending_option_index,
                        name=line.split(":", 1)[1].strip(),
                    )
                )
                setting.pending_option_index = None

    finish_group()

    if not scheme_guid:
        raise ValueError("Could not find a power scheme in powercfg output.")
    return PowerSettingsSnapshot(
        scheme_guid=scheme_guid,
        scheme_name=scheme_name,
        groups=tuple(groups),
    )
