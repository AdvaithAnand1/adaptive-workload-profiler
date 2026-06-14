from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from config import SwitchingConfig


@dataclass(frozen=True)
class TuningChange:
    old: float | int
    new: float | int
    reason: str


@dataclass(frozen=True)
class ProbeTuningReport:
    recommended_switching: dict[str, float | int]
    changes: dict[str, TuningChange]
    notes: list[str]
    warnings: list[str]


def _clamp_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def tune_switching_config(
    summary: Mapping[str, Any],
    current: SwitchingConfig,
) -> ProbeTuningReport:
    proposal: dict[str, float | int] = {
        "poll_interval_sec": current.poll_interval_sec,
        "stability_window": current.stability_window,
        "confidence_min": current.confidence_min,
        "confidence_margin": current.confidence_margin,
        "min_switch_interval_sec": current.min_switch_interval_sec,
        "probability_ema_alpha": current.probability_ema_alpha,
        "downshift_extra_window": current.downshift_extra_window,
        "downshift_hold_sec": current.downshift_hold_sec,
    }
    reasons: dict[str, str] = {}
    notes: list[str] = []
    warnings: list[str] = []

    rows = int(summary.get("rows", 0) or 0)
    low_conf_ratio = _as_float(summary.get("low_confidence_ratio"))
    switches_per_10min = _as_float(summary.get("switches_per_10min"))
    avg_conf = _as_float(summary.get("avg_decision_confidence"), current.confidence_min)
    avg_margin = _as_float(summary.get("avg_decision_margin"), current.confidence_margin)
    avg_raw_conf = _as_float(summary.get("avg_raw_confidence"), avg_conf)
    dominant_label_ratio = _as_float(summary.get("dominant_label_ratio"))
    saver_gap = _as_float(summary.get("heavy_rate_power_saver_gap"), -1.0)
    gate_counts_raw = summary.get("gate_reason_counts")
    gate_counts = gate_counts_raw if isinstance(gate_counts_raw, Mapping) else {}

    def gate_ratio(reason: str) -> float:
        if rows <= 0:
            return 0.0
        try:
            return float(gate_counts.get(reason, 0)) / rows
        except Exception:
            return 0.0

    def set_value(key: str, new_value: float | int, reason: str):
        old_value = proposal[key]
        if old_value != new_value:
            proposal[key] = new_value
            reasons[key] = reason

    if dominant_label_ratio > 0.85:
        warnings.append(
            "Predictions collapse onto one label too often; collect more balanced workload sessions before trusting aggressive threshold changes."
        )
    if saver_gap > 0.35:
        warnings.append(
            "Predictions are still strongly power-plan-sensitive; record each label with power saver on and off before relaxing thresholds further."
        )

    if switches_per_10min > 4.0:
        severity = min(2, int((switches_per_10min - 4.0) // 2.0) + 1)
        set_value(
            "stability_window",
            _clamp_int(current.stability_window + severity, 2, 12),
            "high switch churn -> require more stable agreement before acting",
        )
        set_value(
            "min_switch_interval_sec",
            round(_clamp_float(current.min_switch_interval_sec + 2.0 * severity, 2.0, 30.0), 2),
            "high switch churn -> add more time between profile changes",
        )
        set_value(
            "confidence_margin",
            round(_clamp_float(current.confidence_margin + 0.02 * severity, 0.02, 0.30), 3),
            "high switch churn -> require a clearer class lead",
        )
        set_value(
            "probability_ema_alpha",
            round(_clamp_float(current.probability_ema_alpha - 0.05 * severity, 0.20, 0.95), 3),
            "high switch churn -> smooth probabilities more strongly",
        )
        set_value(
            "downshift_extra_window",
            _clamp_int(current.downshift_extra_window + 1, 0, 6),
            "high switch churn -> make downshifts slightly more conservative",
        )
        set_value(
            "downshift_hold_sec",
            round(_clamp_float(current.downshift_hold_sec + 2.0 * severity, 0.0, 30.0), 2),
            "high switch churn -> hold higher profiles a bit longer before dropping",
        )
        notes.append("Probe suggests the system is over-reacting; tune toward steadier switching.")

    if low_conf_ratio > 0.35 and not warnings:
        if avg_conf >= max(0.40, current.confidence_min - 0.08):
            set_value(
                "confidence_min",
                round(_clamp_float(current.confidence_min - 0.05, 0.35, 0.85), 3),
                "many predictions are blocked as low-confidence -> slightly relax confidence threshold",
            )
        if avg_margin >= max(0.03, current.confidence_margin - 0.03):
            set_value(
                "confidence_margin",
                round(_clamp_float(proposal["confidence_margin"] - 0.02, 0.02, 0.30), 3),
                "many predictions are blocked by thin class separation -> slightly relax margin threshold",
            )
        if avg_raw_conf - avg_conf > 0.06:
            set_value(
                "probability_ema_alpha",
                round(_clamp_float(float(proposal["probability_ema_alpha"]) + 0.05, 0.20, 0.95), 3),
                "raw confidence is much sharper than smoothed confidence -> make smoothing more responsive",
            )
        if gate_ratio("building-stability") > 0.15 and switches_per_10min <= 4.0:
            set_value(
                "stability_window",
                _clamp_int(int(proposal["stability_window"]) - 1, 2, 12),
                "system spends too long building stability -> shorten the agreement window slightly",
            )
        notes.append("Probe shows many held predictions; modestly relaxing thresholds may improve responsiveness.")

    if switches_per_10min < 0.8 and low_conf_ratio < 0.20:
        if gate_ratio("cooldown") > 0.10:
            set_value(
                "min_switch_interval_sec",
                round(_clamp_float(float(proposal["min_switch_interval_sec"]) - 1.5, 2.0, 30.0), 2),
                "few switches and many cooldown blocks -> reduce enforced cooldown slightly",
            )
        if gate_ratio("downshift-hold") > 0.10:
            set_value(
                "downshift_hold_sec",
                round(_clamp_float(float(proposal["downshift_hold_sec"]) - 2.0, 0.0, 30.0), 2),
                "few switches and many downshift holds -> shorten the high-profile hold slightly",
            )
            set_value(
                "downshift_extra_window",
                _clamp_int(int(proposal["downshift_extra_window"]) - 1, 0, 6),
                "few switches and many downshift holds -> reduce extra downshift samples slightly",
            )
        if gate_ratio("building-stability") > 0.12:
            set_value(
                "stability_window",
                _clamp_int(int(proposal["stability_window"]) - 1, 2, 12),
                "few switches and many stability holds -> shorten the stability window slightly",
            )
        if reasons:
            notes.append("Probe suggests the policy may be overly sticky; loosen only the gates doing the most blocking.")

    changes: dict[str, TuningChange] = {}
    current_map = {
        "poll_interval_sec": current.poll_interval_sec,
        "stability_window": current.stability_window,
        "confidence_min": current.confidence_min,
        "confidence_margin": current.confidence_margin,
        "min_switch_interval_sec": current.min_switch_interval_sec,
        "probability_ema_alpha": current.probability_ema_alpha,
        "downshift_extra_window": current.downshift_extra_window,
        "downshift_hold_sec": current.downshift_hold_sec,
    }
    for key, new_value in proposal.items():
        old_value = current_map[key]
        if old_value != new_value:
            changes[key] = TuningChange(
                old=old_value,
                new=new_value,
                reason=reasons.get(key, "probe-driven adjustment"),
            )

    if not changes:
        notes.append("Current switching parameters already look reasonable for this probe.")

    return ProbeTuningReport(
        recommended_switching=proposal,
        changes=changes,
        notes=notes,
        warnings=warnings,
    )
