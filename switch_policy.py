from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from config import VALID_PROFILES

Profile = Literal["silent", "balanced", "performance"]
_PROFILE_RANK = {name: idx for idx, name in enumerate(VALID_PROFILES)}


@dataclass(frozen=True)
class SwitchGateResult:
    allow: bool
    reason: str
    direction: str
    required_stability: int
    remaining_wait_s: float


def classify_direction(current_profile: str | None, target_profile: str) -> str:
    if not current_profile:
        return "initial"
    current_rank = _PROFILE_RANK.get(str(current_profile).strip().lower(), 1)
    target_rank = _PROFILE_RANK.get(str(target_profile).strip().lower(), 1)
    if target_rank > current_rank:
        return "upshift"
    if target_rank < current_rank:
        return "downshift"
    return "same"


def required_stability_window(
    *,
    base_window: int,
    current_profile: str | None,
    target_profile: str,
    downshift_extra_window: int,
) -> int:
    direction = classify_direction(current_profile, target_profile)
    if direction == "downshift":
        return max(1, int(base_window) + max(0, int(downshift_extra_window)))
    return max(1, int(base_window))


def evaluate_switch_gate(
    *,
    current_profile: str | None,
    target_profile: str,
    confident: bool,
    stable_count: int,
    base_window: int,
    downshift_extra_window: int,
    now_ts: float,
    last_switch_ts: float,
    min_switch_interval_sec: float,
    downshift_hold_sec: float,
) -> SwitchGateResult:
    direction = classify_direction(current_profile, target_profile)
    required = required_stability_window(
        base_window=base_window,
        current_profile=current_profile,
        target_profile=target_profile,
        downshift_extra_window=downshift_extra_window,
    )

    if direction == "same":
        return SwitchGateResult(
            allow=False,
            reason="already-applied",
            direction=direction,
            required_stability=required,
            remaining_wait_s=0.0,
        )

    if not confident:
        return SwitchGateResult(
            allow=False,
            reason="low-confidence",
            direction=direction,
            required_stability=required,
            remaining_wait_s=0.0,
        )

    if stable_count < required:
        return SwitchGateResult(
            allow=False,
            reason="building-stability",
            direction=direction,
            required_stability=required,
            remaining_wait_s=0.0,
        )

    elapsed = max(0.0, now_ts - float(last_switch_ts))
    base_remaining = max(0.0, float(min_switch_interval_sec) - elapsed)
    downshift_remaining = 0.0
    if direction == "downshift":
        downshift_remaining = max(0.0, float(downshift_hold_sec) - elapsed)

    if downshift_remaining > 0.0:
        return SwitchGateResult(
            allow=False,
            reason="downshift-hold",
            direction=direction,
            required_stability=required,
            remaining_wait_s=downshift_remaining,
        )

    if base_remaining > 0.0:
        return SwitchGateResult(
            allow=False,
            reason="cooldown",
            direction=direction,
            required_stability=required,
            remaining_wait_s=base_remaining,
        )

    return SwitchGateResult(
        allow=True,
        reason="ready",
        direction=direction,
        required_stability=required,
        remaining_wait_s=0.0,
    )


def describe_gate(result: SwitchGateResult) -> str:
    if result.reason == "ready":
        return f"ready to {result.direction}"
    if result.reason == "already-applied":
        return "target already applied"
    if result.reason == "low-confidence":
        return "holding for stronger confidence"
    if result.reason == "building-stability":
        return f"building stability ({result.required_stability} samples needed)"
    if result.reason == "cooldown":
        return f"cooldown ({result.remaining_wait_s:.1f}s left)"
    if result.reason == "downshift-hold":
        return f"holding higher profile ({result.remaining_wait_s:.1f}s left)"
    return result.reason
