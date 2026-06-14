"""
Shared runtime configuration for PerfAnalyze.

Load order:
1) Built-in defaults
2) Optional JSON file override (perfalyze_config.json by default)

Invalid user config values raise RuntimeError with concrete field names.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_FILE = "perfalyze_config.json"
VALID_PROFILES = ("silent", "balanced", "performance")

DEFAULT_LABEL_TO_PROFILE: dict[str, str] = {
    "idle": "silent",
    "light": "balanced",
    "browsing": "balanced",
    "gaming": "performance",
    "rendering": "performance",
    "heavy": "performance",
}

DEFAULT_PROFILE_HOTKEYS: dict[str, str] = {
    "silent": "ctrl+shift+alt+f16",
    "balanced": "ctrl+shift+alt+f17",
    "performance": "ctrl+shift+alt+f18",
}

_DEFAULT_CONFIG: dict[str, Any] = {
    "switching": {
        "poll_interval_sec": 0.5,
        "stability_window": 5,
        "confidence_min": 0.55,
        "confidence_margin": 0.10,
        "min_switch_interval_sec": 8.0,
        "probability_ema_alpha": 0.45,
        "downshift_extra_window": 2,
        "downshift_hold_sec": 10.0,
    },
    "label_to_profile": DEFAULT_LABEL_TO_PROFILE,
    "profile_hotkeys": DEFAULT_PROFILE_HOTKEYS,
}


@dataclass(frozen=True)
class SwitchingConfig:
    poll_interval_sec: float
    stability_window: int
    confidence_min: float
    confidence_margin: float
    min_switch_interval_sec: float
    probability_ema_alpha: float
    downshift_extra_window: int
    downshift_hold_sec: float


@dataclass(frozen=True)
class PerfAnalyzeConfig:
    switching: SwitchingConfig
    label_to_profile: dict[str, str]
    profile_hotkeys: dict[str, str]


@dataclass(frozen=True)
class ConfigLoadResult:
    config: PerfAnalyzeConfig
    source: str
    path: Path | None


def default_config() -> PerfAnalyzeConfig:
    return _validate_and_build(copy.deepcopy(_DEFAULT_CONFIG))


def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> None:
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, Mapping)
        ):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def _must_be_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"'{field}' must be an object/map.")
    return value


def _must_be_float(
    value: Any,
    field: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"'{field}' must be a number.")
    x = float(value)
    if min_value is not None and x < min_value:
        raise RuntimeError(f"'{field}' must be >= {min_value}.")
    if max_value is not None and x > max_value:
        raise RuntimeError(f"'{field}' must be <= {max_value}.")
    return x


def _must_be_int(value: Any, field: str, *, min_value: int | None = None) -> int:
    if not isinstance(value, int):
        raise RuntimeError(f"'{field}' must be an integer.")
    if min_value is not None and value < min_value:
        raise RuntimeError(f"'{field}' must be >= {min_value}.")
    return value


def _normalize_profile(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"'{field}' must be a non-empty string.")
    norm = value.strip().lower()
    if norm not in VALID_PROFILES:
        valid = ", ".join(VALID_PROFILES)
        raise RuntimeError(f"'{field}' must be one of: {valid}.")
    return norm


def _parse_label_to_profile(value: Any) -> dict[str, str]:
    raw = _must_be_mapping(value, "label_to_profile")
    out: dict[str, str] = {}
    for raw_label, raw_profile in raw.items():
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise RuntimeError("All keys in 'label_to_profile' must be non-empty strings.")
        label = raw_label.strip().lower()
        profile = _normalize_profile(
            raw_profile,
            f"label_to_profile.{raw_label}",
        )
        out[label] = profile

    if not out:
        raise RuntimeError("'label_to_profile' cannot be empty.")
    return out


def _parse_profile_hotkeys(value: Any) -> dict[str, str]:
    raw = _must_be_mapping(value, "profile_hotkeys")
    out: dict[str, str] = {}
    for raw_profile, raw_hotkey in raw.items():
        profile = _normalize_profile(raw_profile, f"profile_hotkeys key '{raw_profile}'")
        if not isinstance(raw_hotkey, str):
            raise RuntimeError(
                f"'profile_hotkeys.{raw_profile}' must be a string."
            )
        out[profile] = raw_hotkey.strip()
    return out


def _validate_and_build(merged: Mapping[str, Any]) -> PerfAnalyzeConfig:
    switching = _must_be_mapping(merged.get("switching"), "switching")
    switching_cfg = SwitchingConfig(
        poll_interval_sec=_must_be_float(
            switching.get("poll_interval_sec"),
            "switching.poll_interval_sec",
            min_value=0.01,
        ),
        stability_window=_must_be_int(
            switching.get("stability_window"),
            "switching.stability_window",
            min_value=1,
        ),
        confidence_min=_must_be_float(
            switching.get("confidence_min"),
            "switching.confidence_min",
            min_value=0.0,
            max_value=1.0,
        ),
        confidence_margin=_must_be_float(
            switching.get("confidence_margin"),
            "switching.confidence_margin",
            min_value=0.0,
            max_value=1.0,
        ),
        min_switch_interval_sec=_must_be_float(
            switching.get("min_switch_interval_sec"),
            "switching.min_switch_interval_sec",
            min_value=0.0,
        ),
        probability_ema_alpha=_must_be_float(
            switching.get("probability_ema_alpha"),
            "switching.probability_ema_alpha",
            min_value=0.0,
            max_value=1.0,
        ),
        downshift_extra_window=_must_be_int(
            switching.get("downshift_extra_window"),
            "switching.downshift_extra_window",
            min_value=0,
        ),
        downshift_hold_sec=_must_be_float(
            switching.get("downshift_hold_sec"),
            "switching.downshift_hold_sec",
            min_value=0.0,
        ),
    )

    label_to_profile = _parse_label_to_profile(merged.get("label_to_profile"))
    profile_hotkeys = _parse_profile_hotkeys(merged.get("profile_hotkeys"))

    return PerfAnalyzeConfig(
        switching=switching_cfg,
        label_to_profile=label_to_profile,
        profile_hotkeys=profile_hotkeys,
    )


def load_config(path: str | Path = CONFIG_FILE) -> ConfigLoadResult:
    config_path = Path(path).expanduser()
    merged = copy.deepcopy(_DEFAULT_CONFIG)
    source = "defaults"
    resolved_path: Path | None = None

    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Invalid JSON in '{config_path}': line {e.lineno}, col {e.colno}."
            ) from e
        except OSError as e:
            raise RuntimeError(f"Could not read '{config_path}': {e}") from e

        if not isinstance(raw, Mapping):
            raise RuntimeError(f"Config root in '{config_path}' must be an object/map.")
        _deep_update(merged, raw)
        resolved_path = config_path.resolve()
        source = str(resolved_path)

    cfg = _validate_and_build(merged)
    return ConfigLoadResult(config=cfg, source=source, path=resolved_path)


def build_config(raw: Mapping[str, Any]) -> PerfAnalyzeConfig:
    if not isinstance(raw, Mapping):
        raise RuntimeError("Config root must be an object/map.")
    merged = copy.deepcopy(_DEFAULT_CONFIG)
    _deep_update(merged, raw)
    return _validate_and_build(merged)


def config_to_dict(cfg: PerfAnalyzeConfig) -> dict[str, Any]:
    return {
        "switching": {
            "poll_interval_sec": cfg.switching.poll_interval_sec,
            "stability_window": cfg.switching.stability_window,
            "confidence_min": cfg.switching.confidence_min,
            "confidence_margin": cfg.switching.confidence_margin,
            "min_switch_interval_sec": cfg.switching.min_switch_interval_sec,
            "probability_ema_alpha": cfg.switching.probability_ema_alpha,
            "downshift_extra_window": cfg.switching.downshift_extra_window,
            "downshift_hold_sec": cfg.switching.downshift_hold_sec,
        },
        "label_to_profile": dict(cfg.label_to_profile),
        "profile_hotkeys": dict(cfg.profile_hotkeys),
    }


def write_config(
    cfg: PerfAnalyzeConfig,
    path: str | Path = CONFIG_FILE,
) -> Path:
    out_path = Path(path).expanduser()
    data = config_to_dict(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return out_path.resolve()
