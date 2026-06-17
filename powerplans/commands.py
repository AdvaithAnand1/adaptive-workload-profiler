"""Safe command objects and execution for Windows `powercfg`."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class PowerCfgCommand:
    """One explicit powercfg command with mutation metadata."""

    arguments: tuple[str, ...]
    description: str
    mutates: bool = True
    destructive: bool = False

    def __post_init__(self):
        if not self.arguments:
            raise ValueError("PowerCfgCommand requires at least one argument.")
        if not self.description.strip():
            raise ValueError("PowerCfgCommand requires a description.")

    def render(self, executable: str = "powercfg") -> str:
        return subprocess.list2cmdline([executable, *self.arguments])


@dataclass(frozen=True)
class PowerCfgCommandResult:
    command: PowerCfgCommand
    executed: bool
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""


@dataclass
class PowerCfgExecutor:
    """Execute powercfg commands only after two explicit safety opt-ins.

    Defaults:
      - dry_run=True: no subprocess is started.
      - allow_mutation=False: mutating commands are refused even if dry-run is
        disabled accidentally.
      - allow_destructive=False: deletion is separately locked.
    """

    executable: str = "powercfg"
    timeout_sec: float = 8.0
    dry_run: bool = True
    allow_mutation: bool = False
    allow_destructive: bool = False

    def execute(self, command: PowerCfgCommand) -> PowerCfgCommandResult:
        if self.dry_run:
            return PowerCfgCommandResult(command=command, executed=False)
        if command.mutates and not self.allow_mutation:
            raise PermissionError(
                "Mutating powercfg command refused. Set both dry_run=False and "
                "allow_mutation=True only from an explicit user-confirmed path."
            )
        if command.destructive and not self.allow_destructive:
            raise PermissionError(
                "Destructive powercfg command refused. Deletion additionally "
                "requires allow_destructive=True."
            )

        result = subprocess.run(
            [self.executable, *command.arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=self.timeout_sec,
        )
        command_result = PowerCfgCommandResult(
            command=command,
            executed=True,
            returncode=result.returncode,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown error").strip()
            raise RuntimeError(
                f"{command.render(self.executable)} failed: {detail}"
            )
        return command_result
