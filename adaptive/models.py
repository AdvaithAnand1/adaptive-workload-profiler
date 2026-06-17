from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TelemetrySnapshot:
    captured_at: str
    predicted_label: str
    plan_id: str
    confidence: float
    on_battery: bool
    power_source: str
    battery_percent: float | None = None
    battery_drain_pct_per_hour: float | None = None
    cpu_usage_pct: float | None = None
    gpu_usage_pct: float | None = None
    throughput_mb_s: float | None = None
    manual_override: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "captured_at": self.captured_at,
            "predicted_label": self.predicted_label,
            "plan_id": self.plan_id,
            "confidence": self.confidence,
            "on_battery": self.on_battery,
            "power_source": self.power_source,
            "battery_percent": self.battery_percent,
            "battery_drain_pct_per_hour": self.battery_drain_pct_per_hour,
            "cpu_usage_pct": self.cpu_usage_pct,
            "gpu_usage_pct": self.gpu_usage_pct,
            "throughput_mb_s": self.throughput_mb_s,
            "manual_override": self.manual_override,
        }


@dataclass(frozen=True)
class ObservationWindow:
    window_id: str
    predicted_label: str
    plan_id: str
    power_source: str
    started_at: str
    ended_at: str
    sample_count: int
    settled_sample_count: int
    average_confidence: float
    average_battery_drain_pct_per_hour: float | None = None
    average_cpu_usage_pct: float | None = None
    average_gpu_usage_pct: float | None = None
    average_throughput_mb_s: float | None = None
    manual_override_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "predicted_label": self.predicted_label,
            "plan_id": self.plan_id,
            "power_source": self.power_source,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "sample_count": self.sample_count,
            "settled_sample_count": self.settled_sample_count,
            "average_confidence": self.average_confidence,
            "average_battery_drain_pct_per_hour": self.average_battery_drain_pct_per_hour,
            "average_cpu_usage_pct": self.average_cpu_usage_pct,
            "average_gpu_usage_pct": self.average_gpu_usage_pct,
            "average_throughput_mb_s": self.average_throughput_mb_s,
            "manual_override_count": self.manual_override_count,
        }


@dataclass(frozen=True)
class PolicyOutcome:
    window: ObservationWindow
    score: float
    helped: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window.to_dict(),
            "score": self.score,
            "helped": self.helped,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class OutcomeComparison:
    predicted_label: str
    baseline_plan_id: str
    candidate_plan_id: str
    baseline_score: float
    candidate_score: float
    comparable_windows: int
    better_plan_id: str | None = None
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_label": self.predicted_label,
            "baseline_plan_id": self.baseline_plan_id,
            "candidate_plan_id": self.candidate_plan_id,
            "baseline_score": self.baseline_score,
            "candidate_score": self.candidate_score,
            "comparable_windows": self.comparable_windows,
            "better_plan_id": self.better_plan_id,
            "created_at": self.created_at,
        }

