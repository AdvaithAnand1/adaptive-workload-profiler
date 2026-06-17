"""Read-only Windows power-plan discovery service."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

from powerplans.parser import (
    PowerSettingsSnapshot,
    WindowsPowerScheme,
    parse_power_schemes,
    parse_power_settings,
)


@dataclass(frozen=True)
class PowerCfgDiscovery:
    executable: str = "powercfg"
    timeout_sec: float = 8.0

    @staticmethod
    def is_supported() -> bool:
        return os.name == "nt" and shutil.which("powercfg") is not None

    def _run(self, *args: str) -> str:
        result = subprocess.run(
            [self.executable, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=self.timeout_sec,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown error").strip()
            raise RuntimeError(f"powercfg {' '.join(args)} failed: {detail}")
        return result.stdout

    def list_schemes(self) -> list[WindowsPowerScheme]:
        return parse_power_schemes(self._run("/list"))

    def discover_settings(
        self,
        scheme: str = "SCHEME_CURRENT",
        *,
        include_hidden: bool = True,
    ) -> PowerSettingsSnapshot:
        command = "/qh" if include_hidden else "/query"
        return parse_power_settings(self._run(command, scheme))
