"""
OS-native telemetry collector for workload classification.

No third-party desktop telemetry software is required.
All metrics are read-only snapshots from the local OS via psutil.
"""

from __future__ import annotations

import csv
import io
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

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
    "GPU Usage [%]",
    "GPU 3D Usage [%]",
    "GPU Compute Usage [%]",
    "GPU Video Usage [%]",
    "GPU Copy Usage [%]",
    "GPU Dedicated Memory [MB]",
    "GPU Shared Memory [MB]",
    "GPU Available [0/1]",
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


@dataclass(frozen=True)
class NativeRuntimeContext:
    captured_at: str
    battery_percent: float | None
    battery_plugged: bool | None
    battery_drain_pct_per_hour: float | None
    active_power_plan: str
    power_saver: bool


_battery_context_prev: tuple[float, float] | None = None


def get_runtime_context() -> NativeRuntimeContext:
    global _battery_context_prev

    sampler = _ensure_sampler()
    sampler._refresh_power_plan_if_needed()  # read-only probe of active scheme

    now = time.time()
    captured_at = datetime.now(timezone.utc).isoformat()
    battery = psutil.sensors_battery()
    battery_percent: float | None = None
    battery_plugged: bool | None = None
    drain_pct_per_hour: float | None = None

    if battery is not None:
        battery_percent = float(battery.percent)
        battery_plugged = bool(battery.power_plugged)
        if _battery_context_prev is not None and not battery_plugged:
            prev_percent, prev_ts = _battery_context_prev
            dt = max(now - prev_ts, 1e-6)
            drain_pct_per_hour = ((prev_percent - battery_percent) / dt) * 3600.0
        _battery_context_prev = (battery_percent, now)

    return NativeRuntimeContext(
        captured_at=captured_at,
        battery_percent=battery_percent,
        battery_plugged=battery_plugged,
        battery_drain_pct_per_hour=drain_pct_per_hour,
        active_power_plan=sampler._power_plan_name,
        power_saver=bool(sampler._power_saver),
    )


@dataclass(frozen=True)
class _GPUMetrics:
    usage_pct: float = 0.0
    usage_3d_pct: float = 0.0
    usage_compute_pct: float = 0.0
    usage_video_pct: float = 0.0
    usage_copy_pct: float = 0.0
    dedicated_memory_mb: float = 0.0
    shared_memory_mb: float = 0.0
    available: float = 0.0


def _parse_typeperf_gpu_output(output: str) -> _GPUMetrics:
    rows = list(csv.reader(io.StringIO(output)))
    header_index = next(
        (
            i
            for i, row in enumerate(rows)
            if row and row[0].strip().startswith("(PDH-CSV")
        ),
        None,
    )
    if header_index is None or header_index + 1 >= len(rows):
        return _GPUMetrics()

    header = rows[header_index]
    values = rows[header_index + 1]
    if len(values) != len(header):
        return _GPUMetrics()

    usage = {"3d": 0.0, "compute": 0.0, "video": 0.0, "copy": 0.0}
    dedicated_bytes = 0.0
    shared_bytes = 0.0
    found_counter = False

    for name, raw_value in zip(header[1:], values[1:]):
        try:
            value = max(0.0, float(raw_value))
        except ValueError:
            continue

        norm = name.lower()
        if "utilization percentage" in norm:
            found_counter = True
            if "engtype_video" in norm:
                usage["video"] += value
            elif "engtype_compute" in norm:
                usage["compute"] += value
            elif "engtype_copy" in norm:
                usage["copy"] += value
            elif "engtype_3d" in norm:
                usage["3d"] += value
        elif "dedicated usage" in norm:
            found_counter = True
            dedicated_bytes += value
        elif "shared usage" in norm:
            found_counter = True
            shared_bytes += value

    clamped = {name: min(value, 100.0) for name, value in usage.items()}
    return _GPUMetrics(
        usage_pct=max(clamped.values()),
        usage_3d_pct=clamped["3d"],
        usage_compute_pct=clamped["compute"],
        usage_video_pct=clamped["video"],
        usage_copy_pct=clamped["copy"],
        dedicated_memory_mb=dedicated_bytes / (1024.0 * 1024.0),
        shared_memory_mb=shared_bytes / (1024.0 * 1024.0),
        available=1.0 if found_counter else 0.0,
    )


class _WindowsGPUSampler:
    COUNTERS = (
        r"\GPU Engine(*)\Utilization Percentage",
        r"\GPU Adapter Memory(*)\Dedicated Usage",
        r"\GPU Adapter Memory(*)\Shared Usage",
    )

    def __init__(self):
        self._metrics = _GPUMetrics()
        self._lock = threading.Lock()
        if os.name == "nt":
            threading.Thread(target=self._run, daemon=True).start()

    def snapshot(self) -> _GPUMetrics:
        with self._lock:
            return self._metrics

    def _run(self):
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        while True:
            try:
                output = subprocess.check_output(
                    ["typeperf", *self.COUNTERS, "-sc", "1"],
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    timeout=5.0,
                    creationflags=creationflags,
                )
                metrics = _parse_typeperf_gpu_output(output)
                with self._lock:
                    self._metrics = metrics
            except Exception:
                time.sleep(5.0)


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
        self._gpu_sampler = _WindowsGPUSampler()

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
        gpu = self._gpu_sampler.snapshot()

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
            gpu.usage_pct,
            gpu.usage_3d_pct,
            gpu.usage_compute_pct,
            gpu.usage_video_pct,
            gpu.usage_copy_pct,
            gpu.dedicated_memory_mb,
            gpu.shared_memory_mb,
            gpu.available,
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


def _active_mock_label(now_s: float) -> str:
    raw = os.getenv("PERFANALYZE_MOCK_LABEL", "").strip().lower()
    if raw in {"idle", "light", "heavy"}:
        return raw

    # Probe mode fallback: cycle through labels when no explicit label is set.
    cycle_len = 45.0
    phase = now_s % (cycle_len * 3.0)
    if phase < cycle_len:
        return "idle"
    if phase < 2.0 * cycle_len:
        return "light"
    return "heavy"


def _mock_features() -> np.ndarray:
    t = time.time()
    label = _active_mock_label(t)

    if label == "idle":
        cpu_mid, cpu_amp = 12.0, 6.0
        disk_mid, disk_amp = 0.4, 1.5
        net_mid, net_amp = 25.0, 70.0
        ram_mid, ram_amp = 38.0, 4.0
        proc_mid, proc_amp = 165.0, 12.0
    elif label == "light":
        cpu_mid, cpu_amp = 35.0, 12.0
        disk_mid, disk_amp = 4.5, 8.0
        net_mid, net_amp = 180.0, 230.0
        ram_mid, ram_amp = 52.0, 6.0
        proc_mid, proc_amp = 220.0, 20.0
    else:  # heavy
        cpu_mid, cpu_amp = 78.0, 12.0
        disk_mid, disk_amp = 35.0, 20.0
        net_mid, net_amp = 420.0, 260.0
        ram_mid, ram_amp = 71.0, 7.0
        proc_mid, proc_amp = 280.0, 28.0

    gpu_profiles = {
        "idle": (3.0, 2.0, 0.0, 0.5, 0.5, 180.0, 420.0),
        "light": (32.0, 8.0, 2.0, 32.0, 2.0, 420.0, 850.0),
        "heavy": (82.0, 72.0, 38.0, 18.0, 8.0, 1800.0, 2400.0),
    }
    gpu_usage, gpu_3d, gpu_compute, gpu_video, gpu_copy, gpu_dedicated, gpu_shared = (
        gpu_profiles[label]
    )

    cpu_usage = cpu_mid + cpu_amp * np.sin(t / 4.8)
    cpu_ema = cpu_mid + (cpu_amp * 0.85) * np.sin((t - 1.4) / 4.8)
    core_std = 2.0 + (cpu_amp * 0.35) + 6.0 * abs(np.sin(t / 4.0))
    core_std_ema = core_std * 0.92 + 1.0 * abs(np.sin((t - 2.2) / 4.0))
    user_pct = max(0.0, cpu_usage * 0.74)
    sys_pct = max(0.0, cpu_usage * 0.22)
    ctx = 4200.0 + (cpu_usage * 230.0) + 900.0 * abs(np.sin(t / 3.0))
    intr = 1600.0 + (cpu_usage * 55.0) + 500.0 * abs(np.sin(t / 5.0))
    ram_pct = ram_mid + ram_amp * abs(np.sin(t / 8.0))
    ram_avail_mb = 13000.0 - 92.0 * ram_pct
    disk_r = disk_mid + disk_amp * abs(np.sin(t / 7.0))
    disk_w = (disk_mid * 0.8) + (disk_amp * 0.7) * abs(np.sin(t / 5.5))
    net_d = net_mid + net_amp * abs(np.sin(t / 9.0))
    net_u = (net_mid * 0.28) + (net_amp * 0.26) * abs(np.sin(t / 9.0 + 0.8))
    proc_count = proc_mid + proc_amp * abs(np.sin(t / 25.0))
    freq_mhz = 1350.0 + 35.0 * cpu_usage
    freq_ratio = min(100.0, max(0.0, freq_mhz / 42.0))
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
        float(gpu_usage),
        float(gpu_3d),
        float(gpu_compute),
        float(gpu_video),
        float(gpu_copy),
        float(gpu_dedicated),
        float(gpu_shared),
        1.0,
    ]

    noise = _mock_rng.normal(0.0, 0.35, size=len(base)).astype(np.float32)
    # Keep binary fields deterministic in mock mode.
    noise[17] = 0.0
    noise[18] = 0.0
    noise[-1] = 0.0
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
