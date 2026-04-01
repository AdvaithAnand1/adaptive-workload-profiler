"""
Minimal live demo GUI for telemetry -> model prediction -> oracle profile switching.
"""

from __future__ import annotations

import json
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import torch
import torch.nn.functional as F

from model import SystemStateNet
from monitor import FEATURE_NAMES, get_telemetry
from oracle_client import OracleClient, Profile, profile_for_label

MODEL_FILE = "model.pth"
CLASSES_FILE = "classes.json"
POLL_INTERVAL_MS = 500
STABILITY_WINDOW = 5
CONFIDENCE_MIN = 0.55
CONFIDENCE_MARGIN = 0.10
MIN_SWITCH_INTERVAL_SEC = 8.0
MAX_LOG_LINES = 200


def load_model_and_classes():
    model_path = Path(MODEL_FILE)
    classes_path = Path(CLASSES_FILE)
    if not model_path.exists() or not classes_path.exists():
        raise RuntimeError(
            "Missing model artifacts. Expected model.pth and classes.json."
        )

    with classes_path.open("r", encoding="utf-8") as f:
        classes = json.load(f)
    if not classes:
        raise RuntimeError("classes.json is empty; retrain the model.")

    model = SystemStateNet(len(FEATURE_NAMES), len(classes))
    state = torch.load(model_path, map_location="cpu")
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(
            "Model/feature schema mismatch. Re-record data and re-run train_model.py."
        ) from e
    model.eval()
    return model, classes


class DemoApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PerfAnalyze Demo")
        self.root.geometry("920x620")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes: list[str] = []

        self.oracle = OracleClient()
        self.running = False
        self.tick_count = 0

        self.current_profile: Profile | None = None
        self.last_candidate: str | None = None
        self.stable_count = 0
        self.last_switch_ts = 0.0
        self.manual_override: Profile | None = None

        self.dry_run_var = tk.BooleanVar(value=True)
        self.metrics_var = tk.StringVar(value="Metrics: -")
        self.prediction_var = tk.StringVar(value="Prediction: -")
        self.profile_var = tk.StringVar(value="Profile: -")
        self.status_var = tk.StringVar(value="Status: stopped")

        self.feature_index = {name: i for i, name in enumerate(FEATURE_NAMES)}
        self._build_ui()
        self._set_profile_line(target="-")

    def _build_ui(self):
        root_frame = ttk.Frame(self.root, padding=10)
        root_frame.pack(fill=tk.BOTH, expand=True)

        info_frame = ttk.LabelFrame(root_frame, text="Live State", padding=10)
        info_frame.pack(fill=tk.X)

        ttk.Label(info_frame, textvariable=self.metrics_var).pack(anchor="w")
        ttk.Label(info_frame, textvariable=self.prediction_var).pack(anchor="w")
        ttk.Label(info_frame, textvariable=self.profile_var).pack(anchor="w")
        ttk.Label(info_frame, textvariable=self.status_var).pack(anchor="w")

        controls = ttk.Frame(root_frame, padding=(0, 10, 0, 10))
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="Start", command=self.start).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(controls, text="Stop", command=self.stop).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Checkbutton(
            controls,
            text="Dry Run (no oracle call)",
            variable=self.dry_run_var,
        ).pack(side=tk.LEFT, padx=(0, 14))

        ttk.Label(controls, text="Manual:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            controls, text="Silent", command=lambda: self.manual_set("silent")
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            controls, text="Balanced", command=lambda: self.manual_set("balanced")
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            controls,
            text="Performance",
            command=lambda: self.manual_set("performance"),
        ).pack(side=tk.LEFT)
        ttk.Button(
            controls,
            text="Resume Auto",
            command=self.clear_manual_override,
        ).pack(side=tk.LEFT, padx=(8, 0))

        log_frame = ttk.LabelFrame(root_frame, text="Event Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_box = tk.Text(log_frame, height=18, wrap="word")
        self.log_box.pack(fill=tk.BOTH, expand=True)
        self.log_box.configure(state=tk.DISABLED)

    def log(self, message: str):
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {message}\n"

        self.log_box.configure(state=tk.NORMAL)
        self.log_box.insert(tk.END, line)
        line_count = int(float(self.log_box.index("end-1c").split(".")[0]))
        if line_count > MAX_LOG_LINES:
            self.log_box.delete("1.0", "2.0")
        self.log_box.see(tk.END)
        self.log_box.configure(state=tk.DISABLED)

    def _set_profile_line(self, target: str):
        applied = self.oracle.get_profile() or self.current_profile or "-"
        mode = "MANUAL" if self.manual_override else "AUTO"
        manual = self.manual_override or "-"
        self.profile_var.set(
            f"Target={target} | Applied={applied} | "
            f"Mode={mode} Manual={manual} | "
            f"Oracle={self.oracle.backend_name} | DryRun={self.dry_run_var.get()}"
        )

    def _format_metrics(self, x) -> str:
        ix = self.feature_index
        cpu = x[ix["Total CPU Usage [%]"]]
        ram = x[ix["RAM Usage [%]"]]
        battery = x[ix["On Battery [0/1]"]]
        saver = x[ix["Power Saver [0/1]"]]
        disk_r = x[ix["Disk Read [MB/s]"]]
        disk_w = x[ix["Disk Write [MB/s]"]]
        net_d = x[ix["Network Down [KB/s]"]]
        net_u = x[ix["Network Up [KB/s]"]]
        return (
            f"Metrics: CPU={cpu:5.1f}% RAM={ram:5.1f}% "
            f"Batt={battery:.0f} Saver={saver:.0f} "
            f"DiskR={disk_r:6.2f}MB/s DiskW={disk_w:6.2f}MB/s "
            f"NetD={net_d:7.1f}KB/s NetU={net_u:7.1f}KB/s"
        )

    def _set_status(self, text: str):
        self.status_var.set(f"Status: {text}")

    def start(self):
        if self.running:
            return
        try:
            self.model, self.classes = load_model_and_classes()
            self.model.to(self.device)
        except Exception as e:
            self._set_status("error")
            self.log(f"start failed: {e}")
            return

        self.running = True
        self.tick_count = 0
        self.current_profile = None
        self.last_candidate = None
        self.stable_count = 0
        self.last_switch_ts = 0.0
        self.manual_override = None
        self._set_status("running")
        self.log("demo started")
        self.root.after(POLL_INTERVAL_MS, self._tick)

    def stop(self):
        if not self.running:
            return
        self.running = False
        self._set_status("stopped")
        self.log("demo stopped")

    def manual_set(self, profile: Profile):
        # Toggle-off behavior for convenience: clicking the active manual profile
        # exits manual mode and resumes automatic switching.
        if self.manual_override == profile:
            self.clear_manual_override()
            return

        self.manual_override = profile
        result = self.oracle.set_profile(profile, dry_run=self.dry_run_var.get())
        if result.ok:
            self.current_profile = result.applied_profile or profile
            self.last_switch_ts = time.time()
            self.log(f"manual override ON -> {profile} (ok)")
        else:
            self.log(f"manual override ON -> {profile} (error: {result.message})")
        self.last_candidate = None
        self.stable_count = 0
        self._set_profile_line(target=profile)

    def clear_manual_override(self):
        if self.manual_override is None:
            self.log("manual override already OFF")
            return
        prev = self.manual_override
        self.manual_override = None
        self.last_candidate = None
        self.stable_count = 0
        self.log(f"manual override OFF (was {prev}); resumed AUTO mode")
        self._set_profile_line(target="-")

    def _tick(self):
        if not self.running or self.model is None:
            return

        try:
            x = get_telemetry()
            x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                logits = self.model(x_t)
                probs = F.softmax(logits, dim=1)
                topk = probs.topk(k=min(2, probs.shape[1]), dim=1)
                idx = int(topk.indices[0, 0].item())
                pred_label = self.classes[idx]
                confidence = float(topk.values[0, 0].item())
                runner_up = (
                    float(topk.values[0, 1].item())
                    if topk.values.shape[1] > 1
                    else 0.0
                )

            margin = confidence - runner_up
            confident = (
                confidence >= CONFIDENCE_MIN and margin >= CONFIDENCE_MARGIN
            )

            if confident:
                if pred_label == self.last_candidate:
                    self.stable_count += 1
                else:
                    self.last_candidate = pred_label
                    self.stable_count = 1
            else:
                self.last_candidate = None
                self.stable_count = 0

            target_profile = (
                self.manual_override
                if self.manual_override is not None
                else profile_for_label(pred_label)
            )
            switched = False

            if (
                self.manual_override is None
                and
                confident
                and self.stable_count >= STABILITY_WINDOW
                and target_profile != self.current_profile
                and (time.time() - self.last_switch_ts) >= MIN_SWITCH_INTERVAL_SEC
            ):
                result = self.oracle.set_profile(
                    target_profile,
                    dry_run=self.dry_run_var.get(),
                )
                if result.ok:
                    self.current_profile = result.applied_profile or target_profile
                    self.log(
                        f"switch label={pred_label} -> profile={self.current_profile} "
                        f"(confidence={confidence:.2f})"
                    )
                else:
                    self.log(
                        f"switch failed label={pred_label} -> profile={target_profile}: "
                        f"{result.message}"
                    )
                switched = True
                self.last_switch_ts = time.time()

            self.metrics_var.set(self._format_metrics(x))
            self.prediction_var.set(
                f"Prediction: label={pred_label} confidence={confidence:.2f} "
                f"margin={margin:.2f} confident={confident} "
                f"stability={self.stable_count}/{STABILITY_WINDOW}"
            )
            self._set_profile_line(target=target_profile)

            # Light heartbeat in log so demo observers can see the loop is alive.
            self.tick_count += 1
            if self.tick_count % 6 == 0 and not switched:
                self.log(
                    f"tick label={pred_label} conf={confidence:.2f} "
                    f"stable={self.stable_count}/{STABILITY_WINDOW} "
                    f"mode={'MANUAL' if self.manual_override else 'AUTO'}"
                )

        except Exception as e:
            self._set_status("error")
            self.log(f"runtime error: {e}")
            self.running = False
            return

        self.root.after(POLL_INTERVAL_MS, self._tick)


def main():
    root = tk.Tk()
    app = DemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
