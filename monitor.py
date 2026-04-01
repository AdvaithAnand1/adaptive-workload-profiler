"""
OS-native telemetry collector for workload classification.

No third-party desktop telemetry software is required.
All metrics are read-only snapshots from the local OS via psutil.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass

import numpy as np
import psutil


FEATURE_NAMES = [
    "Total CPU Usage [%]",
    "CPU Usage EMA [%]",
    "CPU Frequency [MHz]",
    "CPU Frequency Ratio [%]",
    "CPU Core Usage StdDev [%]",
    "CPU Core Usage StdDev EMA [%]",
    "CPU User Time [%]",
    "CPU System Time [%]",
    "Context Switches [/s]",
    "Interrupts [/s]",
    "RAM Usage [%]",
    "RAM Available [MB]",
    "Disk Read [MB/s]",
    "Disk Write [MB/s]",
    "Network Down [KB/s]",
    "Network Up [KB/s]",
    "Process Count",
    "On Battery [0/1]",
    "Power Saver [0/1]",
    "CPU Physical Cores",
    "CPU Logical Cores",
]


@dataclass
class _CounterState:
    ts: float
    ctx_switches: int
    interrupts: int
    disk_read_bytes: int
    disk_write_bytes: int
    net_recv_bytes: int
    net_sent_bytes: int


def _safe_rate(curr: float, prev: float, dt: float) -> float:
    if dt <= 0.0:
        return 0.0
    return max(0.0, (curr - prev) / dt)


def _safe_cpu_freq_metrics() -> tuple[float, float]:
    freq = psutil.cpu_freq()
    if freq is None:
        return 0.0, 0.0

    current = float(freq.current or 0.0)
    max_freq = float(freq.max or 0.0)
    if max_freq <= 0.0:
        max_freq = current

    ratio = 0.0 if max_freq <= 0.0 else (100.0 * current / max_freq)
    return current, float(min(max(ratio, 0.0), 100.0))


def _core_counts() -> tuple[float, float]:
    return (
        float(psutil.cpu_count(logical=False) or 0),
        float(psutil.cpu_count(logical=True) or 0),
    )


def _on_battery_flag() -> float:
    battery = psutil.sensors_battery()
    if battery is None:
        return 0.0
    return 0.0 if bool(battery.power_plugged) else 1.0


class OSTelemetrySampler:
    """
    Samples system telemetry without external desktop utilities.

    Design notes:
      - Uses non-blocking psutil calls.
      - Derives rates from cumulative counters between samples.
      - Includes lightweight temporal features (EMA) for stability.
      - Returns fixed-order vectors aligned with FEATURE_NAMES.
    """

    EMA_ALPHA = 0.35
    POWER_PLAN_REFRESH_SEC = 5.0

    def __init__(self):
        self._prime_percent_calculators()
        self.prev = self._read_counters()

        self.cpu_ema: float | None = None
        self.core_std_ema: float | None = None

        self._power_saver = 0.0
        self._power_plan_name = "unknown"
        self._last_power_poll = 0.0

    def _prime_percent_calculators(self):
        # First non-blocking percent call can be stale; prime once up front.
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
        psutil.cpu_times_percent(interval=None)

    def _read_counters(self) -> _CounterState:
        now = time.time()

        cpu_stats = psutil.cpu_stats()
        disk = psutil.disk_io_counters()
        net = psutil.net_io_counters()

        return _CounterState(
            ts=now,
            ctx_switches=int(getattr(cpu_stats, "ctx_switches", 0)),
            interrupts=int(getattr(cpu_stats, "interrupts", 0)),
            disk_read_bytes=int(getattr(disk, "read_bytes", 0)) if disk else 0,
            disk_write_bytes=int(getattr(disk, "write_bytes", 0)) if disk else 0,
            net_recv_bytes=int(getattr(net, "bytes_recv", 0)) if net else 0,
            net_sent_bytes=int(getattr(net, "bytes_sent", 0)) if net else 0,
        )

    def _update_ema(self, current: float, prev: float | None) -> float:
        if prev is None:
            return current
        return self.EMA_ALPHA * current + (1.0 - self.EMA_ALPHA) * prev

    def _refresh_power_plan_if_needed(self):
        now = time.time()
        if (now - self._last_power_poll) < self.POWER_PLAN_REFRESH_SEC:
            return
        self._last_power_poll = now

        try:
            output = subprocess.check_output(
                ["powercfg", "/getactivescheme"],
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=2.0,
            )
            m = re.search(r"\((.*?)\)", output)
            name = m.group(1).strip() if m else output.strip()
            norm = name.lower()
            saver = 1.0 if ("power saver" in norm or "energy saver" in norm) else 0.0

            self._power_plan_name = name
            self._power_saver = saver
        except Exception:
            # Keep last known state; do not fail telemetry collection.
            return

    def sample(self) -> np.ndarray:
        self._refresh_power_plan_if_needed()

        per_core = np.asarray(
            psutil.cpu_percent(interval=None, percpu=True), dtype=np.float32
        )
        total_cpu_pct = float(per_core.mean()) if per_core.size else 0.0
        core_std_pct = float(per_core.std()) if per_core.size else 0.0

        self.cpu_ema = self._update_ema(total_cpu_pct, self.cpu_ema)
        self.core_std_ema = self._update_ema(core_std_pct, self.core_std_ema)

        cpu_times = psutil.cpu_times_percent(interval=None)
        cpu_user_pct = float(getattr(cpu_times, "user", 0.0))
        cpu_system_pct = float(getattr(cpu_times, "system", 0.0))

        freq_mhz, freq_ratio = _safe_cpu_freq_metrics()

        vm = psutil.virtual_memory()
        ram_usage_pct = float(vm.percent)
        ram_available_mb = float(vm.available / (1024.0 * 1024.0))

        current = self._read_counters()
        dt = max(current.ts - self.prev.ts, 1e-6)

        ctx_per_s = _safe_rate(current.ctx_switches, self.prev.ctx_switches, dt)
        interrupts_per_s = _safe_rate(current.interrupts, self.prev.interrupts, dt)

        disk_read_mb_s = _safe_rate(
            current.disk_read_bytes, self.prev.disk_read_bytes, dt
        ) / (1024.0 * 1024.0)
        disk_write_mb_s = _safe_rate(
            current.disk_write_bytes, self.prev.disk_write_bytes, dt
        ) / (1024.0 * 1024.0)

        net_down_kb_s = _safe_rate(
            current.net_recv_bytes, self.prev.net_recv_bytes, dt
        ) / 1024.0
        net_up_kb_s = _safe_rate(
            current.net_sent_bytes, self.prev.net_sent_bytes, dt
        ) / 1024.0

        process_count = float(len(psutil.pids()))
        on_battery = _on_battery_flag()
        phys, logical = _core_counts()

        self.prev = current

        base = [
            total_cpu_pct,
            float(self.cpu_ema or total_cpu_pct),
            freq_mhz,
            freq_ratio,
            core_std_pct,
            float(self.core_std_ema or core_std_pct),
            cpu_user_pct,
            cpu_system_pct,
            ctx_per_s,
            interrupts_per_s,
            ram_usage_pct,
            ram_available_mb,
            disk_read_mb_s,
            disk_write_mb_s,
            net_down_kb_s,
            net_up_kb_s,
            process_count,
            on_battery,
            self._power_saver,
            phys,
            logical,
        ]

        return np.asarray(base, dtype=np.float32)


_mock_rng = np.random.default_rng(42)
_sampler: OSTelemetrySampler | None = None


def _use_mock_mode() -> bool:
    return os.getenv("PERFANALYZE_MOCK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _mock_features() -> np.ndarray:
    t = time.time()
    cpu_usage = 35.0 + 25.0 * np.sin(t / 6.0)
    cpu_ema = 35.0 + 22.0 * np.sin((t - 1.8) / 6.0)
    core_std = 5.0 + 18.0 * abs(np.sin(t / 4.0))
    core_std_ema = 5.5 + 15.0 * abs(np.sin((t - 2.2) / 4.0))
    user_pct = max(0.0, cpu_usage * 0.75)
    sys_pct = max(0.0, cpu_usage * 0.20)
    ctx = 14000.0 + 4500.0 * abs(np.sin(t / 3.0))
    intr = 3800.0 + 1800.0 * abs(np.sin(t / 5.0))
    ram_pct = 50.0 + 8.0 * abs(np.sin(t / 8.0))
    ram_avail_mb = 12000.0 - 85.0 * ram_pct
    disk_r = 0.8 + 28.0 * abs(np.sin(t / 7.0))
    disk_w = 0.6 + 18.0 * abs(np.sin(t / 5.5))
    net_d = 10.0 + 420.0 * abs(np.sin(t / 9.0))
    net_u = 4.0 + 120.0 * abs(np.sin(t / 9.0 + 0.8))
    proc_count = 210.0 + 35.0 * abs(np.sin(t / 25.0))
    freq_mhz = 1700.0 + 28.0 * cpu_usage
    freq_ratio = min(100.0, max(0.0, freq_mhz / 45.0))
    on_battery = 0.0
    power_saver = 0.0
    phys, logical = _core_counts()

    base = [
        float(cpu_usage),
        float(cpu_ema),
        float(freq_mhz),
        float(freq_ratio),
        float(core_std),
        float(core_std_ema),
        float(user_pct),
        float(sys_pct),
        float(ctx),
        float(intr),
        float(ram_pct),
        float(ram_avail_mb),
        float(disk_r),
        float(disk_w),
        float(net_d),
        float(net_u),
        float(proc_count),
        float(on_battery),
        float(power_saver),
        float(phys),
        float(logical),
    ]

    noise = _mock_rng.normal(0.0, 0.35, size=len(base)).astype(np.float32)
    # Keep binary fields deterministic in mock mode.
    noise[17] = 0.0
    noise[18] = 0.0
    return np.asarray(base, dtype=np.float32) + noise


def get_telemetry() -> np.ndarray:
    """
    One-call helper: returns fixed-order feature vector for current system state.
    """
    if _use_mock_mode():
        return _mock_features()

    global _sampler
    if _sampler is None:
        _sampler = OSTelemetrySampler()
    return _sampler.sample()


if __name__ == "__main__":
    print("Collecting OS-native telemetry samples...")
    for _ in range(5):
        print("Features:", get_telemetry())
        time.sleep(0.5)
