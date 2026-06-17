"""Portable Windows power settings supported by PerfAnalyze."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortableSettingSpec:
    key: str
    label: str
    domain: str
    description: str


# These are Windows-defined aliases, not vendor extension GUIDs. A setting is
# shown and applied only when powercfg discovery confirms that the current
# machine exposes it.
PORTABLE_SETTING_SPECS: tuple[PortableSettingSpec, ...] = (
    PortableSettingSpec(
        key="SUB_PROCESSOR/PROCTHROTTLEMAX",
        label="Maximum processor state",
        domain="cpu",
        description="Maximum requested processor performance percentage.",
    ),
    PortableSettingSpec(
        key="SUB_PROCESSOR/PERFBOOSTMODE",
        label="Processor boost mode",
        domain="cpu",
        description="Controls whether and how Windows requests processor boost.",
    ),
    PortableSettingSpec(
        key="SUB_PROCESSOR/PERFEPP",
        label="Processor energy-performance preference",
        domain="cpu",
        description="Balances processor energy efficiency against performance.",
    ),
    PortableSettingSpec(
        key="SUB_GRAPHICS/GPUPREFERENCEPOLICY",
        label="GPU preference policy",
        domain="graphics",
        description="Uses the standard Windows graphics power preference policy.",
    ),
    PortableSettingSpec(
        key="SUB_PCIEXPRESS/ASPM",
        label="PCI Express link-state power management",
        domain="graphics",
        description="Controls generic PCI Express link power savings.",
    ),
)

PORTABLE_SETTINGS_BY_KEY = {
    setting.key: setting for setting in PORTABLE_SETTING_SPECS
}
PORTABLE_SETTING_KEYS = frozenset(PORTABLE_SETTINGS_BY_KEY)


def is_portable_setting_key(key: str) -> bool:
    return key.strip().upper() in PORTABLE_SETTING_KEYS
