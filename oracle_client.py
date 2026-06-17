"""
Oracle adapters for applying Silent/Balanced/Performance decisions.

Backend order:
1. External `oracle` module if present (best path for G-Helper-specific integration)
2. Generic Windows power plan fallback via `powercfg`
3. In-memory demo stub
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from config import DEFAULT_LABEL_TO_PROFILE, VALID_PROFILES

Profile = Literal["silent", "balanced", "performance"]
_VALID_PROFILES = set(VALID_PROFILES)

WORKLOAD_TO_PROFILE: dict[str, Profile] = {
    label: cast(Profile, profile)
    for label, profile in DEFAULT_LABEL_TO_PROFILE.items()
}

POWERCFG_ALIAS_BY_PROFILE: dict[Profile, str] = {
    # Windows names these aliases by performance level, not power usage.
    "silent": "SCHEME_MAX",
    "balanced": "SCHEME_BALANCED",
    "performance": "SCHEME_MIN",
}

_POWERCFG_SUBGROUP_PROCESSOR = "54533251-82be-4824-96c1-47b60b740d00"
_POWERCFG_SETTING_MIN_CPU = "893dee8e-2bef-41e0-89c6-b55d0929964c"
_POWERCFG_SETTING_MAX_CPU = "bc5038f7-23e0-4960-96da-33abaf5935ec"
_POWERCFG_SETTING_BOOST_MODE = "be337238-0d82-4146-a960-4f3749d470c7"

_POWERCFG_BOOST_MODE_NAMES: dict[int, str] = {
    0: "disabled",
    1: "enabled",
    2: "aggressive",
    3: "efficient-enabled",
    4: "efficient-aggressive",
    5: "aggressive-at-guaranteed",
}

_DEFAULT_GUID_TO_PROFILE: dict[str, Profile] = {
    "a1841308-3541-4fab-bc81-f71556f20b4a": "silent",      # Power saver
    "381b4222-f694-41f0-9685-ff5bb260df2e": "balanced",   # Balanced
    "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "performance",  # High performance
    "e9a42b02-d5df-448d-aa00-03f14749eb61": "performance",  # Ultimate performance
}

_POWERCFG_KEYWORDS: dict[Profile, tuple[str, ...]] = {
    "silent": ("power saver", "energy saver", "best power efficiency", "efficiency"),
    "balanced": ("balanced",),
    "performance": (
        "ultimate performance",
        "high performance",
        "turbo",
        "performance",
    ),
}

_POWERCFG_ENV_BY_PROFILE: dict[Profile, str] = {
    "silent": "PERFANALYZE_POWERCFG_SILENT",
    "balanced": "PERFANALYZE_POWERCFG_BALANCED",
    "performance": "PERFANALYZE_POWERCFG_PERFORMANCE",
}

_ORACLE_COMMAND_ENV = "PERFANALYZE_ORACLE_COMMAND"
_ORACLE_COMMAND_TIMEOUT_ENV = "PERFANALYZE_ORACLE_COMMAND_TIMEOUT"
_ORACLE_VERIFY_ENV = "PERFANALYZE_VERIFY_ORACLE_APPLY"


@dataclass(frozen=True)
class PowerScheme:
    guid: str
    name: str
    active: bool = False


@dataclass
class OracleResult:
    ok: bool
    applied_profile: Profile | None = None
    message: str = ""
    observed_profile: Profile | None = None
    verified: bool | None = None


@dataclass(frozen=True)
class OracleDiagnostics:
    backend_name: str
    selected_profile: str
    details: dict[str, Any]


@dataclass(frozen=True)
class PowerCfgProfileTuning:
    min_cpu_percent: int | None = None
    max_cpu_percent: int | None = None
    boost_mode: int | None = None


def profile_for_label(
    label: str,
    label_map: Mapping[str, str] | None = None,
) -> Profile:
    source = label_map if label_map is not None else WORKLOAD_TO_PROFILE
    norm_label = label.strip().lower()
    raw_profile = str(source.get(norm_label, "balanced")).strip().lower()
    if raw_profile in _VALID_PROFILES:
        return cast(Profile, raw_profile)
    return "balanced"


def parse_powercfg_list(output: str) -> list[PowerScheme]:
    schemes: list[PowerScheme] = []
    pattern = re.compile(
        r"Power Scheme GUID:\s*([0-9a-fA-F\-]{36})\s*\((.*?)\)\s*(\*)?",
        re.IGNORECASE,
    )
    for raw_line in output.splitlines():
        line = raw_line.strip()
        match = pattern.search(line)
        if not match:
            continue
        schemes.append(
            PowerScheme(
                guid=match.group(1).strip().lower(),
                name=match.group(2).strip(),
                active=bool(match.group(3)),
            )
        )
    return schemes


def profile_for_power_scheme_name(name: str) -> Profile | None:
    norm = name.strip().lower()
    for profile, keywords in _POWERCFG_KEYWORDS.items():
        if any(keyword in norm for keyword in keywords):
            return profile
    return None


def profile_for_power_scheme(scheme: PowerScheme) -> Profile | None:
    if scheme.guid in _DEFAULT_GUID_TO_PROFILE:
        return _DEFAULT_GUID_TO_PROFILE[scheme.guid]
    return profile_for_power_scheme_name(scheme.name)


def _powercfg_preference_rank(profile: Profile, scheme: PowerScheme) -> int:
    norm = scheme.name.strip().lower()
    if profile == "performance":
        if "ultimate performance" in norm:
            return 0
        if "high performance" in norm:
            return 1
        if "performance" in norm:
            return 2
        return 20
    if profile == "silent":
        if "best power efficiency" in norm:
            return 0
        if "energy saver" in norm:
            return 1
        if "power saver" in norm:
            return 2
        return 20
    if "balanced" in norm:
        return 0
    return 20


def resolve_powercfg_target_token(raw: str, schemes: Sequence[PowerScheme]) -> str:
    token = raw.strip()
    if not token:
        return token

    guid_match = re.fullmatch(r"[0-9a-fA-F\-]{36}", token)
    if guid_match:
        return token.lower()

    upper = token.upper()
    if upper in {"SCHEME_MIN", "SCHEME_BALANCED", "SCHEME_MAX"}:
        return upper

    exact = [scheme for scheme in schemes if scheme.name.strip().lower() == token.lower()]
    if exact:
        return exact[0].guid

    partial = [scheme for scheme in schemes if token.lower() in scheme.name.strip().lower()]
    if len(partial) == 1:
        return partial[0].guid

    return token


def powercfg_scheme_report(schemes: Sequence[PowerScheme]) -> list[dict[str, Any]]:
    return [
        {
            "guid": scheme.guid,
            "name": scheme.name,
            "active": scheme.active,
            "mapped_profile": profile_for_power_scheme(scheme),
        }
        for scheme in schemes
    ]


def _parse_percentage_value(raw: str, *, env_name: str) -> int:
    try:
        value = int(raw.strip())
    except ValueError as e:
        raise RuntimeError(f"{env_name} must be an integer between 0 and 100.") from e
    if value < 0 or value > 100:
        raise RuntimeError(f"{env_name} must be between 0 and 100.")
    return value


def _parse_boost_mode_value(raw: str, *, env_name: str) -> int:
    token = raw.strip().lower().replace("_", "-").replace(" ", "-")
    if token.isdigit():
        value = int(token)
        if 0 <= value <= 5:
            return value
        raise RuntimeError(f"{env_name} numeric value must be between 0 and 5.")

    name_to_value = {name: value for value, name in _POWERCFG_BOOST_MODE_NAMES.items()}
    aliases = {
        "off": 0,
        "on": 1,
        "efficient": 3,
        "efficient-aggressive": 4,
        "efficient-enabled": 3,
    }
    if token in aliases:
        return aliases[token]
    if token in name_to_value:
        return name_to_value[token]
    raise RuntimeError(
        f"{env_name} must be one of {sorted(name_to_value)} or an integer 0-5."
    )


def _describe_powercfg_tuning(tuning: PowerCfgProfileTuning) -> list[str]:
    parts: list[str] = []
    if tuning.min_cpu_percent is not None:
        parts.append(f"min_cpu={tuning.min_cpu_percent}%")
    if tuning.max_cpu_percent is not None:
        parts.append(f"max_cpu={tuning.max_cpu_percent}%")
    if tuning.boost_mode is not None:
        mode_name = _POWERCFG_BOOST_MODE_NAMES.get(tuning.boost_mode, str(tuning.boost_mode))
        parts.append(f"boost={mode_name}")
    return parts


def load_powercfg_profile_tuning(
    environ: Mapping[str, str] | None = None,
) -> dict[Profile, PowerCfgProfileTuning]:
    env = environ if environ is not None else os.environ
    result: dict[Profile, PowerCfgProfileTuning] = {}
    for profile in VALID_PROFILES:
        prefix = f"PERFANALYZE_POWERCFG_{profile.upper()}_"
        min_raw = str(env.get(prefix + "MIN_CPU", "")).strip()
        max_raw = str(env.get(prefix + "MAX_CPU", "")).strip()
        boost_raw = str(env.get(prefix + "BOOST_MODE", "")).strip()
        if not min_raw and not max_raw and not boost_raw:
            continue
        result[cast(Profile, profile)] = PowerCfgProfileTuning(
            min_cpu_percent=_parse_percentage_value(min_raw, env_name=prefix + "MIN_CPU") if min_raw else None,
            max_cpu_percent=_parse_percentage_value(max_raw, env_name=prefix + "MAX_CPU") if max_raw else None,
            boost_mode=_parse_boost_mode_value(boost_raw, env_name=prefix + "BOOST_MODE") if boost_raw else None,
        )
    return result


def powercfg_tuning_report(
    profile_tuning: Mapping[Profile, PowerCfgProfileTuning],
) -> dict[str, dict[str, Any]]:
    return {
        profile: {
            "min_cpu_percent": tuning.min_cpu_percent,
            "max_cpu_percent": tuning.max_cpu_percent,
            "boost_mode": tuning.boost_mode,
            "description": _describe_powercfg_tuning(tuning),
        }
        for profile, tuning in profile_tuning.items()
    }


def build_powercfg_tuning_commands(
    scheme_guid: str,
    tuning: PowerCfgProfileTuning,
) -> list[tuple[str, ...]]:
    commands: list[tuple[str, ...]] = []

    def add_pair(setting_guid: str, value: int):
        commands.append(
            (
                "/setacvalueindex",
                scheme_guid,
                _POWERCFG_SUBGROUP_PROCESSOR,
                setting_guid,
                str(value),
            )
        )
        commands.append(
            (
                "/setdcvalueindex",
                scheme_guid,
                _POWERCFG_SUBGROUP_PROCESSOR,
                setting_guid,
                str(value),
            )
        )

    if tuning.min_cpu_percent is not None:
        add_pair(_POWERCFG_SETTING_MIN_CPU, tuning.min_cpu_percent)
    if tuning.max_cpu_percent is not None:
        add_pair(_POWERCFG_SETTING_MAX_CPU, tuning.max_cpu_percent)
    if tuning.boost_mode is not None:
        add_pair(_POWERCFG_SETTING_BOOST_MODE, tuning.boost_mode)
    return commands


def powercfg_target_for_profile(
    profile: Profile,
    schemes: Sequence[PowerScheme],
    *,
    overrides: Mapping[str, str] | None = None,
) -> str:
    override = ""
    if overrides is not None:
        override = str(overrides.get(profile, "")).strip()
    if override:
        return resolve_powercfg_target_token(override, schemes)

    matching = [scheme for scheme in schemes if profile_for_power_scheme(scheme) == profile]
    if matching:
        matching.sort(key=lambda scheme: (_powercfg_preference_rank(profile, scheme), scheme.name.lower()))
        return matching[0].guid

    return POWERCFG_ALIAS_BY_PROFILE[profile]


class DemoOracle:
    """In-memory fallback used when no external system backend is available."""

    def __init__(self):
        self._profile: Profile = "balanced"

    def set_profile(self, profile: Profile):
        self._profile = profile
        return {
            "ok": True,
            "applied_profile": profile,
            "message": "demo stub applied profile",
        }

    def get_profile(self):
        return self._profile

    def diagnostics(self) -> OracleDiagnostics:
        return OracleDiagnostics(
            backend_name="demo_stub",
            selected_profile=self._profile,
            details={"mode": "in-memory stub"},
        )


class CommandOracle:
    """External command bridge for mock/G-Helper-compatible integrations."""

    def __init__(self, command: str | Sequence[str], *, timeout_sec: float = 5.0):
        if isinstance(command, str):
            parts = shlex.split(command, posix=os.name != "nt")
        else:
            parts = [str(part) for part in command]
        if not parts:
            raise RuntimeError("Oracle command cannot be empty.")
        self._base_command = parts
        self._timeout_sec = float(timeout_sec)
        self._last_profile: Profile = "balanced"

    def _invoke(self, *args: str):
        result = subprocess.run(
            [*self._base_command, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=self._timeout_sec,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown error").strip()
            raise RuntimeError(f"oracle command failed: {detail}")

        text = (result.stdout or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def set_profile(self, profile: Profile):
        raw = self._invoke("set-profile", profile)
        if isinstance(raw, Mapping):
            applied = str(raw.get("applied_profile", profile)).strip().lower()
            if applied in _VALID_PROFILES:
                self._last_profile = cast(Profile, applied)
        else:
            self._last_profile = profile
        return raw

    def get_profile(self):
        raw = self._invoke("get-profile")
        if isinstance(raw, Mapping):
            for key in ("profile", "applied_profile", "selected_profile"):
                value = str(raw.get(key, "")).strip().lower()
                if value in _VALID_PROFILES:
                    self._last_profile = cast(Profile, value)
                    return self._last_profile
            return self._last_profile
        if isinstance(raw, str):
            norm = raw.strip().lower()
            if norm in _VALID_PROFILES:
                self._last_profile = cast(Profile, norm)
                return self._last_profile
        return self._last_profile

    def diagnostics(self) -> OracleDiagnostics:
        raw = self._invoke("diagnostics")
        if isinstance(raw, OracleDiagnostics):
            return raw
        if isinstance(raw, Mapping):
            return OracleDiagnostics(
                backend_name=str(raw.get("backend_name", "oracle_command")),
                selected_profile=str(raw.get("selected_profile", self._last_profile)),
                details=dict(raw.get("details", {})) if isinstance(raw.get("details", {}), Mapping) else {},
            )
        return OracleDiagnostics(
            backend_name="oracle_command",
            selected_profile=self._last_profile,
            details={"mode": "external command bridge", "command": self._base_command},
        )


class PowerCfgOracle:
    """Generic Windows fallback using built-in power plans via `powercfg`."""

    def __init__(self):
        self._last_profile: Profile = "balanced"
        self._schemes: list[PowerScheme] = []
        self._last_active_scheme_name = "unknown"
        self._last_target_token = ""
        self._last_applied_tuning: list[str] = []
        self._overrides = {
            profile: os.getenv(env_var, "").strip()
            for profile, env_var in _POWERCFG_ENV_BY_PROFILE.items()
            if os.getenv(env_var, "").strip()
        }
        self._profile_tuning = load_powercfg_profile_tuning()
        self._refresh_schemes()
        active = self._active_scheme()
        if active is not None:
            mapped = profile_for_power_scheme(active)
            if mapped is not None:
                self._last_profile = mapped
            self._last_active_scheme_name = active.name

    def _run(self, *args: str) -> str:
        result = subprocess.run(
            ["powercfg", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=4.0,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown error").strip()
            raise RuntimeError(f"powercfg {' '.join(args)} failed: {detail}")
        return result.stdout

    def _refresh_schemes(self):
        self._schemes = parse_powercfg_list(self._run("/list"))

    def _active_scheme(self) -> PowerScheme | None:
        for scheme in self._schemes:
            if scheme.active:
                return scheme
        return None

    def _apply_profile_tuning(self, scheme_guid: str, profile: Profile) -> list[str]:
        tuning = self._profile_tuning.get(profile)
        if tuning is None:
            self._last_applied_tuning = []
            return []

        commands = build_powercfg_tuning_commands(scheme_guid, tuning)
        for command in commands:
            self._run(*command)
        if commands:
            self._run("/setactive", scheme_guid)
        self._last_applied_tuning = _describe_powercfg_tuning(tuning)
        return list(self._last_applied_tuning)

    def set_profile(self, profile: Profile):
        token = powercfg_target_for_profile(
            profile,
            self._schemes,
            overrides=self._overrides,
        )
        self._last_target_token = token
        self._run("/setactive", token)
        self._refresh_schemes()
        active = self._active_scheme()
        tuning_parts: list[str] = []
        if active is not None:
            self._last_active_scheme_name = active.name
            tuning_parts = self._apply_profile_tuning(active.guid, profile)
            if tuning_parts:
                self._refresh_schemes()
                active = self._active_scheme()
                if active is not None:
                    self._last_active_scheme_name = active.name
            mapped = profile_for_power_scheme(active) if active is not None else None
            self._last_profile = mapped or profile
        else:
            self._last_profile = profile
            self._last_applied_tuning = []
        message = f"powercfg applied {self._last_active_scheme_name}"
        if tuning_parts:
            message += " | " + ", ".join(tuning_parts)
        return {
            "ok": True,
            "applied_profile": self._last_profile,
            "message": message,
        }

    def get_profile(self):
        try:
            self._refresh_schemes()
            active = self._active_scheme()
            if active is not None:
                self._last_active_scheme_name = active.name
                mapped = profile_for_power_scheme(active)
                if mapped is not None:
                    self._last_profile = mapped
                    return mapped
                return None
        except Exception:
            pass
        return self._last_profile

    def diagnostics(self) -> OracleDiagnostics:
        active = self._active_scheme()
        active_profile = profile_for_power_scheme(active) if active is not None else None
        return OracleDiagnostics(
            backend_name="windows_powercfg",
            selected_profile=active_profile or self._last_profile,
            details={
                "active_scheme_name": self._last_active_scheme_name,
                "active_scheme_guid": active.guid if active is not None else "",
                "last_target_token": self._last_target_token,
                "last_applied_tuning": list(self._last_applied_tuning),
                "overrides": dict(self._overrides),
                "profile_tuning": powercfg_tuning_report(self._profile_tuning),
                "schemes": powercfg_scheme_report(self._schemes),
            },
        )


class OracleClient:
    """
    Normalized oracle interface for the app.

    Backend preference:
    - PERFANALYZE_ORACLE_BACKEND=oracle_module|command|powercfg|demo|auto
    - auto: command bridge (if configured) -> external module -> Windows powercfg -> demo stub
    """

    def __init__(self):
        backend, backend_name = self._load_backend()
        self._backend = backend
        self.backend_name = backend_name
        self._last_profile: Profile = "balanced"

    def _load_backend(self):
        requested = os.getenv("PERFANALYZE_ORACLE_BACKEND", "auto").strip().lower() or "auto"

        if requested in {"demo", "demo_stub"}:
            return DemoOracle(), "demo_stub"

        if requested in {"command", "oracle_command", "command_bridge"}:
            backend = self._try_load_command_oracle()
            if backend is not None:
                return backend, "oracle_command"
            return DemoOracle(), "demo_stub"

        if requested in {"oracle", "oracle_module", "external"}:
            module = self._try_load_external_oracle()
            if module is not None:
                return module, "oracle_module"
            return DemoOracle(), "demo_stub"

        if requested in {"powercfg", "windows_powercfg"}:
            backend = self._try_load_powercfg_oracle()
            if backend is not None:
                return backend, "windows_powercfg"
            return DemoOracle(), "demo_stub"

        backend = self._try_load_command_oracle()
        if backend is not None:
            return backend, "oracle_command"

        module = self._try_load_external_oracle()
        if module is not None:
            return module, "oracle_module"

        backend = self._try_load_powercfg_oracle()
        if backend is not None:
            return backend, "windows_powercfg"

        return DemoOracle(), "demo_stub"

    @staticmethod
    def _try_load_command_oracle():
        command = os.getenv(_ORACLE_COMMAND_ENV, "").strip()
        if not command:
            return None
        timeout_raw = os.getenv(_ORACLE_COMMAND_TIMEOUT_ENV, "5").strip()
        try:
            timeout_sec = float(timeout_raw)
        except ValueError:
            timeout_sec = 5.0
        try:
            return CommandOracle(command, timeout_sec=timeout_sec)
        except Exception:
            return None

    @staticmethod
    def _try_load_external_oracle():
        try:
            import oracle as oracle_module  # type: ignore
        except Exception:
            return None
        return oracle_module

    @staticmethod
    def _try_load_powercfg_oracle():
        if os.name != "nt":
            return None
        if shutil.which("powercfg") is None:
            return None
        try:
            return PowerCfgOracle()
        except Exception:
            return None

    def set_profile(self, profile: Profile, dry_run: bool = False) -> OracleResult:
        if dry_run:
            self._last_profile = profile
            return OracleResult(
                ok=True,
                applied_profile=profile,
                message="dry run",
                observed_profile=profile,
                verified=None,
            )

        try:
            if hasattr(self._backend, "set_profile"):
                raw = self._backend.set_profile(profile)
            elif callable(self._backend):
                raw = self._backend(profile)
            else:
                raise RuntimeError("Oracle backend does not expose set_profile(profile)")
            result = self._normalize_result(raw, requested=profile)
            if result.ok:
                result = self._verify_result(result, requested=profile)
            if result.ok and result.applied_profile is not None:
                self._last_profile = result.applied_profile
            elif result.observed_profile is not None:
                self._last_profile = result.observed_profile
            return result
        except Exception as e:
            return OracleResult(ok=False, message=f"{type(e).__name__}: {e}")

    def _read_backend_profile(self) -> Profile | None:
        try:
            if hasattr(self._backend, "get_profile"):
                raw = self._backend.get_profile()
                return self._coerce_profile(raw, fallback=None)
        except Exception:
            return None
        return None

    def _should_verify_apply(self) -> bool:
        token = os.getenv(_ORACLE_VERIFY_ENV, "1").strip().lower()
        return token not in {"0", "false", "no", "off"}

    def _verify_result(self, result: OracleResult, *, requested: Profile) -> OracleResult:
        if not self._should_verify_apply():
            return result

        observed = self._read_backend_profile()
        if observed is None:
            message = result.message or "verification unavailable"
            return OracleResult(
                ok=result.ok,
                applied_profile=result.applied_profile,
                message=message,
                observed_profile=None,
                verified=None,
            )

        verified = observed == requested
        if verified:
            message = result.message or f"verified applied profile {observed}"
            return OracleResult(
                ok=True,
                applied_profile=observed,
                message=message,
                observed_profile=observed,
                verified=True,
            )

        detail = result.message + " | " if result.message else ""
        detail += f"verification mismatch: observed {observed}, expected {requested}"
        return OracleResult(
            ok=False,
            applied_profile=result.applied_profile,
            message=detail,
            observed_profile=observed,
            verified=False,
        )

    def get_profile(self) -> Profile | None:
        observed = self._read_backend_profile()
        if observed is not None:
            self._last_profile = observed
            return observed
        return self._last_profile

    def diagnostics(self) -> OracleDiagnostics:
        try:
            if hasattr(self._backend, "diagnostics"):
                raw = self._backend.diagnostics()
                if isinstance(raw, OracleDiagnostics):
                    return raw
                if isinstance(raw, Mapping):
                    backend_name = str(raw.get("backend_name", self.backend_name))
                    selected_profile = str(raw.get("selected_profile", self._last_profile))
                    details = dict(raw.get("details", {})) if isinstance(raw.get("details", {}), Mapping) else {}
                    return OracleDiagnostics(
                        backend_name=backend_name,
                        selected_profile=selected_profile,
                        details=details,
                    )
        except Exception as e:
            return OracleDiagnostics(
                backend_name=self.backend_name,
                selected_profile=str(self._last_profile),
                details={"error": f"{type(e).__name__}: {e}"},
            )

        details: dict[str, Any] = {}
        if self.backend_name == "oracle_module":
            details["mode"] = "external oracle module"
        elif self.backend_name == "demo_stub":
            details["mode"] = "in-memory stub"
        return OracleDiagnostics(
            backend_name=self.backend_name,
            selected_profile=str(self._last_profile),
            details=details,
        )

    def _normalize_result(self, raw, requested: Profile) -> OracleResult:
        if isinstance(raw, dict):
            ok = bool(raw.get("ok", True))
            applied = self._coerce_profile(
                raw.get("applied_profile"), fallback=requested if ok else None
            )
            message = str(raw.get("message", ""))
            return OracleResult(ok=ok, applied_profile=applied, message=message)

        if isinstance(raw, bool):
            return OracleResult(
                ok=raw,
                applied_profile=requested if raw else None,
            )

        if isinstance(raw, str):
            applied = self._coerce_profile(raw, fallback=requested)
            return OracleResult(ok=True, applied_profile=applied)

        if raw is None:
            return OracleResult(ok=True, applied_profile=requested)

        return OracleResult(
            ok=True,
            applied_profile=requested,
            message=f"unrecognized return type: {type(raw).__name__}",
        )

    @staticmethod
    def _coerce_profile(value, fallback: Profile | None) -> Profile | None:
        if isinstance(value, str):
            norm = value.strip().lower()
            if norm in _VALID_PROFILES:
                return cast(Profile, norm)
        return fallback
