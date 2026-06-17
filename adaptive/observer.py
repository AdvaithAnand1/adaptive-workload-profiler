from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from uuid import uuid4

from .models import ObservationWindow, PolicyOutcome, TelemetrySnapshot


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class _WindowBuffer:
    started_at: str
    settled_samples: int = 0
    samples: list[TelemetrySnapshot] = None

    def __post_init__(self):
        if self.samples is None:
            self.samples = []


class PolicyObservationTracker:
    def __init__(self, *, settle_seconds: float = 15.0, window_seconds: float = 60.0):
        self.settle_seconds = float(settle_seconds)
        self.window_seconds = float(window_seconds)
        self._window: _WindowBuffer | None = None
        self._last_phase = "idle"

    def phase(self) -> str:
        return self._last_phase

    def observe(self, snapshot: TelemetrySnapshot) -> ObservationWindow | None:
        captured = datetime.fromisoformat(snapshot.captured_at)
        if self._window is None:
            self._window = _WindowBuffer(started_at=snapshot.captured_at)
            self._last_phase = "settling"

        started = datetime.fromisoformat(self._window.started_at)
        elapsed = (captured - started).total_seconds()
        if elapsed < self.settle_seconds:
            self._window.settled_samples += 1
            self._last_phase = "settling"
            return None

        self._window.samples.append(snapshot)
        self._last_phase = "measuring"
        if elapsed < (self.settle_seconds + self.window_seconds):
            return None

        window = self._finalize_window(self._window, ended_at=snapshot.captured_at)
        self._window = None
        self._last_phase = "complete"
        return window

    def _finalize_window(self, buffer: _WindowBuffer, *, ended_at: str) -> ObservationWindow:
        samples = buffer.samples
        first = samples[0]
        return ObservationWindow(
            window_id=uuid4().hex,
            predicted_label=first.predicted_label,
            plan_id=first.plan_id,
            power_source=first.power_source,
            started_at=buffer.started_at,
            ended_at=ended_at,
            sample_count=len(samples),
            settled_sample_count=buffer.settled_samples,
            average_confidence=mean(sample.confidence for sample in samples),
            average_battery_drain_pct_per_hour=_mean_or_none(
                sample.battery_drain_pct_per_hour for sample in samples
            ),
            average_cpu_usage_pct=_mean_or_none(sample.cpu_usage_pct for sample in samples),
            average_gpu_usage_pct=_mean_or_none(sample.gpu_usage_pct for sample in samples),
            average_throughput_mb_s=_mean_or_none(
                sample.throughput_mb_s for sample in samples
            ),
            manual_override_count=sum(1 for sample in samples if sample.manual_override),
        )

    def summarize(self, window: ObservationWindow) -> PolicyOutcome:
        score = 0.0
        if window.average_cpu_usage_pct is not None:
            score += window.average_cpu_usage_pct
        if window.average_gpu_usage_pct is not None:
            score += window.average_gpu_usage_pct
        if window.average_battery_drain_pct_per_hour is not None:
            score -= abs(window.average_battery_drain_pct_per_hour)
        helped = score >= 0.0
        return PolicyOutcome(window=window, score=score, helped=helped, reason="read-only observation")


def _mean_or_none(values):
    vals = [value for value in values if value is not None]
    if not vals:
        return None
    return float(mean(vals))
