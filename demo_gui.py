"""
Live demo GUI for telemetry -> model prediction -> oracle profile switching.

This GUI now supports runtime configuration editing and persistence.
"""

from __future__ import annotations

import json
import time
import tkinter as tk
from collections import Counter
from pathlib import Path
from tkinter import filedialog, ttk

import torch
import torch.nn.functional as F

from config import (
    CONFIG_FILE,
    VALID_PROFILES,
    build_config,
    default_config,
    load_config,
    write_config,
)
from model import SystemStateNet
from monitor import FEATURE_NAMES, get_telemetry
from oracle_client import OracleClient, Profile, profile_for_label
from prediction_logic import ProbabilitySmoother, summarize_probabilities
from switch_policy import describe_gate, evaluate_switch_gate

MODEL_FILE = "model.pth"
CLASSES_FILE = "classes.json"
MAX_LOG_LINES = 200

PRESET_SWITCHING: dict[str, dict[str, float | int]] = {
    "Conservative": {
        "poll_interval_sec": 0.5,
        "stability_window": 7,
        "confidence_min": 0.65,
        "confidence_margin": 0.15,
        "min_switch_interval_sec": 12.0,
        "probability_ema_alpha": 0.35,
        "downshift_extra_window": 3,
        "downshift_hold_sec": 14.0,
    },
    "Balanced": {
        "poll_interval_sec": 0.5,
        "stability_window": 5,
        "confidence_min": 0.55,
        "confidence_margin": 0.10,
        "min_switch_interval_sec": 8.0,
        "probability_ema_alpha": 0.45,
        "downshift_extra_window": 2,
        "downshift_hold_sec": 10.0,
    },
    "Responsive": {
        "poll_interval_sec": 0.4,
        "stability_window": 3,
        "confidence_min": 0.45,
        "confidence_margin": 0.05,
        "min_switch_interval_sec": 4.0,
        "probability_ema_alpha": 0.70,
        "downshift_extra_window": 1,
        "downshift_hold_sec": 5.0,
    },
}


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
        self.root.geometry("1100x820")

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

        self.runtime_config = default_config()
        self.config_source = "defaults"
        self.startup_config_warning = ""
        self.config_path = Path(CONFIG_FILE)
        self._load_runtime_config_on_startup()

        self.dry_run_var = tk.BooleanVar(value=True)
        self.metrics_var = tk.StringVar(value="Waiting for telemetry...")
        self.prediction_var = tk.StringVar(value="Waiting for prediction...")
        self.profile_var = tk.StringVar(value="applied=- | target=-")
        self.recommendation_var = tk.StringVar(value="Waiting for the first recommendation...")
        self.switch_gate_var = tk.StringVar(value="Readiness: waiting for first signal")
        self.target_profile_card_var = tk.StringVar(value="-")
        self.applied_profile_card_var = tk.StringVar(value="-")
        self.gate_card_var = tk.StringVar(value="waiting")
        self.last_action_card_var = tk.StringVar(value="none yet")
        self.status_var = tk.StringVar(value="stopped")
        self.config_var = tk.StringVar(value="source=defaults | path=perfalyze_config.json")
        self.config_hint_var = tk.StringVar(value="Config ready.")
        self.model_info_var = tk.StringVar(value="waiting for model artifacts")
        self.mapping_status_var = tk.StringVar(value="Mappings: waiting for model classes")
        self.config_summary_var = tk.StringVar(
            value="poll=0.5s | window=5 | conf>=0.55 | margin>=0.10"
        )
        self.live_summary_var = tk.StringVar(
            value="label=- | target=- | conf=0.00 | margin=0.00 | reliable=False"
        )
        self.health_var = tk.StringVar(value="model not loaded yet")
        self.path_status_var = tk.StringVar(value="Config path status: default path selected")
        self.hotkey_status_var = tk.StringVar(value="Hotkeys: waiting for configuration review")
        self.validation_status_var = tk.StringVar(value="Validation: checking form...")
        self.backend_info_var = tk.StringVar(value="Backend: checking...")
        self.actuation_var = tk.StringVar(value="Actuation: no profile action yet")
        self.session_stats_var = tk.StringVar(value="Session: auto 0 | manual 0 | failed 0 | blocks none")
        self.mode_badge_var = tk.StringVar(value="Mode AUTO")
        self.execution_badge_var = tk.StringVar(value="Exec DRY RUN")
        self.config_badge_var = tk.StringVar(value="Config SAVED")
        self.model_badge_var = tk.StringVar(value="Model MISSING")
        self.settings_error_var = tk.StringVar(value="")
        self.confidence_var = tk.DoubleVar(value=0.0)
        self.confidence_text_var = tk.StringVar(value="0.0%")
        self.stability_var = tk.DoubleVar(value=0.0)
        self.stability_text_var = tk.StringVar(value="0/0")
        self.log_filter_var = tk.StringVar(value="All")

        self.poll_interval_var = tk.StringVar(value="")
        self.stability_window_var = tk.StringVar(value="")
        self.confidence_min_var = tk.StringVar(value="")
        self.confidence_margin_var = tk.StringVar(value="")
        self.min_switch_interval_var = tk.StringVar(value="")
        self.probability_ema_alpha_var = tk.StringVar(value="")
        self.downshift_extra_window_var = tk.StringVar(value="")
        self.downshift_hold_sec_var = tk.StringVar(value="")
        self.config_path_var = tk.StringVar(value=str(self.config_path))

        self.threshold_specs = {
            "poll_interval_sec": {
                "label": "Poll Interval (s)",
                "var": self.poll_interval_var,
                "kind": "float",
                "min": 0.01,
                "max": None,
                "typical": (0.25, 0.75),
                "typical_text": "0.25–0.75s typical",
                "invalid_text": "Need a number >= 0.01",
            },
            "stability_window": {
                "label": "Stability Window",
                "var": self.stability_window_var,
                "kind": "int",
                "min": 1,
                "max": None,
                "typical": (3, 7),
                "typical_text": "3–7 typical",
                "invalid_text": "Need an integer >= 1",
            },
            "confidence_min": {
                "label": "Confidence Min",
                "var": self.confidence_min_var,
                "kind": "float",
                "min": 0.0,
                "max": 1.0,
                "typical": (0.45, 0.70),
                "typical_text": "0.45–0.70 typical",
                "invalid_text": "Need a number between 0 and 1",
            },
            "confidence_margin": {
                "label": "Confidence Margin",
                "var": self.confidence_margin_var,
                "kind": "float",
                "min": 0.0,
                "max": 1.0,
                "typical": (0.05, 0.20),
                "typical_text": "0.05–0.20 typical",
                "invalid_text": "Need a number between 0 and 1",
            },
            "min_switch_interval_sec": {
                "label": "Min Switch Interval (s)",
                "var": self.min_switch_interval_var,
                "kind": "float",
                "min": 0.0,
                "max": None,
                "typical": (4.0, 12.0),
                "typical_text": "4–12s typical",
                "invalid_text": "Need a number >= 0",
            },
            "probability_ema_alpha": {
                "label": "Probability EMA α",
                "var": self.probability_ema_alpha_var,
                "kind": "float",
                "min": 0.0,
                "max": 1.0,
                "typical": (0.35, 0.65),
                "typical_text": "0.35–0.65 typical",
                "invalid_text": "Need a number between 0 and 1",
            },
            "downshift_extra_window": {
                "label": "Downshift Extra Window",
                "var": self.downshift_extra_window_var,
                "kind": "int",
                "min": 0,
                "max": None,
                "typical": (1, 3),
                "typical_text": "1–3 typical",
                "invalid_text": "Need an integer >= 0",
            },
            "downshift_hold_sec": {
                "label": "Downshift Hold (s)",
                "var": self.downshift_hold_sec_var,
                "kind": "float",
                "min": 0.0,
                "max": None,
                "typical": (5.0, 15.0),
                "typical_text": "5–15s typical",
                "invalid_text": "Need a number >= 0",
            },
        }
        self.threshold_order = [
            "poll_interval_sec",
            "stability_window",
            "confidence_min",
            "confidence_margin",
            "min_switch_interval_sec",
            "probability_ema_alpha",
            "downshift_extra_window",
            "downshift_hold_sec",
        ]
        self.threshold_status_vars = {
            key: tk.StringVar(value=spec["typical_text"])
            for key, spec in self.threshold_specs.items()
        }
        self.threshold_status_labels: dict[str, tk.Label] = {}

        self.new_label_var = tk.StringVar(value="")
        self.mapping_filter_var = tk.StringVar(value="")
        self.mapping_filter_status_var = tk.StringVar(value="Showing 0 labels")
        self.bulk_profile_var = tk.StringVar(value="balanced")
        self.label_map_vars: dict[str, tk.StringVar] = {}
        self.auto_imported_labels: set[str] = set()
        self.hotkey_vars: dict[str, tk.StringVar] = {
            profile: tk.StringVar(value="") for profile in VALID_PROFILES
        }

        self.log_entries: list[tuple[str, str]] = []
        self._suspend_form_traces = False
        self._settings_dirty = False
        self._form_is_valid = True
        self._save_is_valid = True
        self._last_recommend_target = "-"
        self._last_pred_label = "-"
        self._last_pred_confident = False
        self._last_gate_text = "waiting for first signal"
        self._last_action_text = "no profile action yet"
        self._last_action_ts = 0.0
        self._auto_switch_count = 0
        self._manual_switch_count = 0
        self._failed_switch_count = 0
        self._gate_reason_counter: Counter[str] = Counter()
        self.prediction_smoother = ProbabilitySmoother(
            self.runtime_config.switching.probability_ema_alpha
        )

        self.feature_index = {name: i for i, name in enumerate(FEATURE_NAMES)}
        self._ensure_label_map_rows(self.runtime_config.label_to_profile.keys())

        self._build_ui()
        self._bind_form_traces()
        self._sync_form_from_runtime_config()
        self._update_config_line()
        self._set_profile_line(target="-")

        if self.startup_config_warning:
            self._set_settings_error(f"Config warning: {self.startup_config_warning}")
            self.log(
                f"config invalid, using defaults: {self.startup_config_warning}",
                level="warn",
            )
        else:
            self.log(f"config ready ({self.config_source})")

    def _load_runtime_config_on_startup(self):
        try:
            result = load_config(self.config_path)
            self.runtime_config = result.config
            self.config_source = result.source
            if result.path is not None:
                self.config_path = result.path
        except RuntimeError as e:
            self.runtime_config = default_config()
            self.config_source = "defaults"
            self.startup_config_warning = str(e)

    def _build_ui(self):
        root_frame = ttk.Frame(self.root, padding=10)
        root_frame.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(root_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        dashboard_tab = ttk.Frame(notebook, padding=10)
        controls_tab = ttk.Frame(notebook, padding=10)
        settings_tab = ttk.Frame(notebook, padding=10)
        log_tab = ttk.Frame(notebook, padding=10)

        notebook.add(dashboard_tab, text="Dashboard")
        notebook.add(controls_tab, text="Controls")
        notebook.add(settings_tab, text="Configuration")
        notebook.add(log_tab, text="Diagnostics")

        badge_row = ttk.Frame(dashboard_tab)
        badge_row.pack(fill=tk.X, pady=(0, 10))

        recommendation_frame = ttk.LabelFrame(dashboard_tab, text="Recommendation", padding=10)
        recommendation_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            recommendation_frame,
            textvariable=self.recommendation_var,
            justify=tk.LEFT,
            wraplength=900,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", fill=tk.X)
        ttk.Label(
            recommendation_frame,
            textvariable=self.switch_gate_var,
            justify=tk.LEFT,
            wraplength=900,
        ).pack(anchor="w", fill=tk.X, pady=(4, 0))

        snapshot_row = ttk.Frame(dashboard_tab)
        snapshot_row.pack(fill=tk.X, pady=(0, 10))
        snapshot_row.columnconfigure(0, weight=1)
        snapshot_row.columnconfigure(1, weight=1)
        snapshot_row.columnconfigure(2, weight=1)
        snapshot_row.columnconfigure(3, weight=1)

        for col_idx, (title, var) in enumerate([
            ("Target", self.target_profile_card_var),
            ("Applied", self.applied_profile_card_var),
            ("Gate", self.gate_card_var),
            ("Last Action", self.last_action_card_var),
        ]):
            card = ttk.LabelFrame(snapshot_row, text=title, padding=8)
            card.grid(row=0, column=col_idx, sticky="nsew", padx=(0 if col_idx == 0 else 6, 0))
            ttk.Label(
                card,
                textvariable=var,
                justify=tk.LEFT,
                wraplength=200,
                font=("Segoe UI", 9, "bold") if col_idx < 2 else None,
            ).pack(anchor="w", fill=tk.X)

        summary_row = ttk.Frame(dashboard_tab)
        summary_row.pack(fill=tk.X)
        summary_row.columnconfigure(0, weight=1)
        summary_row.columnconfigure(1, weight=1)

        self.mode_badge = tk.Label(badge_row, textvariable=self.mode_badge_var, padx=10, pady=4)
        self.mode_badge.pack(side=tk.LEFT, padx=(0, 8))
        self.execution_badge = tk.Label(badge_row, textvariable=self.execution_badge_var, padx=10, pady=4)
        self.execution_badge.pack(side=tk.LEFT, padx=(0, 8))
        self.config_badge = tk.Label(badge_row, textvariable=self.config_badge_var, padx=10, pady=4)
        self.config_badge.pack(side=tk.LEFT, padx=(0, 8))
        self.model_badge = tk.Label(badge_row, textvariable=self.model_badge_var, padx=10, pady=4)
        self.model_badge.pack(side=tk.LEFT)

        system_frame = ttk.LabelFrame(summary_row, text="System", padding=10)
        system_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        decision_frame = ttk.LabelFrame(summary_row, text="Decision Detail", padding=10)
        decision_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        system_grid = ttk.Frame(system_frame)
        system_grid.pack(fill=tk.X)
        system_grid.columnconfigure(1, weight=1)
        for row_idx, (label, var) in enumerate([
            ("Status", self.status_var),
            ("Model", self.model_info_var),
            ("Config", self.config_var),
            ("Health", self.health_var),
        ]):
            ttk.Label(system_grid, text=label).grid(
                row=row_idx,
                column=0,
                sticky="nw",
                padx=(0, 10),
                pady=2,
            )
            ttk.Label(
                system_grid,
                textvariable=var,
                justify=tk.LEFT,
                wraplength=420,
            ).grid(row=row_idx, column=1, sticky="w", pady=2)

        decision_grid = ttk.Frame(decision_frame)
        decision_grid.pack(fill=tk.X)
        decision_grid.columnconfigure(1, weight=1)
        for row_idx, (label, var) in enumerate([
            ("State", self.profile_var),
            ("Signal", self.live_summary_var),
            ("Policy", self.config_summary_var),
        ]):
            ttk.Label(decision_grid, text=label).grid(
                row=row_idx,
                column=0,
                sticky="nw",
                padx=(0, 10),
                pady=2,
            )
            ttk.Label(
                decision_grid,
                textvariable=var,
                justify=tk.LEFT,
                wraplength=420,
            ).grid(row=row_idx, column=1, sticky="w", pady=2)

        outcomes_row = ttk.Frame(dashboard_tab)
        outcomes_row.pack(fill=tk.X, pady=(10, 0))
        outcomes_row.columnconfigure(0, weight=1)
        outcomes_row.columnconfigure(1, weight=1)

        actuation_frame = ttk.LabelFrame(outcomes_row, text="Actuation", padding=10)
        actuation_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        session_frame = ttk.LabelFrame(outcomes_row, text="Session", padding=10)
        session_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        ttk.Label(
            actuation_frame,
            textvariable=self.actuation_var,
            justify=tk.LEFT,
            wraplength=420,
        ).pack(anchor="w", fill=tk.X)
        ttk.Label(
            session_frame,
            textvariable=self.session_stats_var,
            justify=tk.LEFT,
            wraplength=420,
        ).pack(anchor="w", fill=tk.X)

        live_frame = ttk.LabelFrame(dashboard_tab, text="Live State", padding=10)
        live_frame.pack(fill=tk.X, pady=(10, 0))

        live_grid = ttk.Frame(live_frame)
        live_grid.pack(fill=tk.X)
        live_grid.columnconfigure(0, weight=1)
        live_grid.columnconfigure(1, weight=1)

        metrics_card = ttk.LabelFrame(live_grid, text="Telemetry", padding=8)
        metrics_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        prediction_card = ttk.LabelFrame(live_grid, text="Inference", padding=8)
        prediction_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        ttk.Label(
            metrics_card,
            textvariable=self.metrics_var,
            justify=tk.LEFT,
            wraplength=420,
        ).pack(anchor="w", fill=tk.X)
        ttk.Label(
            prediction_card,
            textvariable=self.prediction_var,
            justify=tk.LEFT,
            wraplength=420,
        ).pack(anchor="w", fill=tk.X)

        meter_frame = ttk.Frame(live_frame)
        meter_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(meter_frame, text="Confidence").grid(row=0, column=0, sticky="w")
        self.confidence_bar = ttk.Progressbar(
            meter_frame,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.confidence_var,
            length=260,
        )
        self.confidence_bar.grid(row=0, column=1, padx=(8, 8), sticky="ew")
        ttk.Label(meter_frame, textvariable=self.confidence_text_var).grid(
            row=0,
            column=2,
            sticky="w",
        )

        ttk.Label(meter_frame, text="Stability").grid(row=1, column=0, sticky="w")
        self.stability_bar = ttk.Progressbar(
            meter_frame,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.stability_var,
            length=260,
        )
        self.stability_bar.grid(row=1, column=1, padx=(8, 8), sticky="ew")
        ttk.Label(meter_frame, textvariable=self.stability_text_var).grid(
            row=1,
            column=2,
            sticky="w",
        )
        meter_frame.columnconfigure(1, weight=1)

        ttk.Label(
            dashboard_tab,
            text="Live state is foregrounded; diagnostics stay tucked away in their own tab.",
            justify=tk.LEFT,
            wraplength=900,
        ).pack(fill=tk.X, pady=(10, 0))

        quick_actions = ttk.LabelFrame(controls_tab, text="Quick Actions", padding=10)
        quick_actions.pack(fill=tk.X)
        ttk.Button(quick_actions, text="Start", command=self.start).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(quick_actions, text="Stop", command=self.stop).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Checkbutton(
            quick_actions,
            text="Dry Run (no oracle call)",
            variable=self.dry_run_var,
        ).pack(side=tk.LEFT, padx=(0, 14))

        manual_controls = ttk.LabelFrame(controls_tab, text="Manual Override", padding=10)
        manual_controls.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(manual_controls, text="Manual:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            manual_controls,
            text="Silent",
            command=lambda: self.manual_set("silent"),
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            manual_controls,
            text="Balanced",
            command=lambda: self.manual_set("balanced"),
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            manual_controls,
            text="Performance",
            command=lambda: self.manual_set("performance"),
        ).pack(side=tk.LEFT)
        ttk.Button(
            manual_controls,
            text="Resume Auto",
            command=self.clear_manual_override,
        ).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(
            controls_tab,
            text="Manual override pauses automatic switching until Resume Auto. Dry Run previews behavior without backend writes.",
            justify=tk.LEFT,
            wraplength=900,
        ).pack(fill=tk.X, pady=(10, 0))

        settings_frame = ttk.LabelFrame(settings_tab, text="Settings", padding=10)
        settings_frame.pack(fill=tk.BOTH, expand=True)

        settings_status_row = ttk.Frame(settings_frame)
        settings_status_row.pack(fill=tk.X)
        settings_status_row.columnconfigure(0, weight=1)
        settings_status_row.columnconfigure(1, weight=1)
        settings_status_row.columnconfigure(2, weight=1)

        file_status_card = ttk.LabelFrame(settings_status_row, text="Config File", padding=8)
        file_status_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        mapping_status_card = ttk.LabelFrame(settings_status_row, text="Coverage", padding=8)
        mapping_status_card.grid(row=0, column=1, sticky="nsew", padx=6)
        validation_card = ttk.LabelFrame(settings_status_row, text="Validation", padding=8)
        validation_card.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

        ttk.Label(file_status_card, textvariable=self.config_var, justify=tk.LEFT, wraplength=300).pack(anchor="w", fill=tk.X)
        ttk.Label(file_status_card, textvariable=self.path_status_var, justify=tk.LEFT, wraplength=300).pack(anchor="w", fill=tk.X, pady=(4, 0))
        ttk.Label(mapping_status_card, textvariable=self.hotkey_status_var, justify=tk.LEFT, wraplength=300).pack(anchor="w", fill=tk.X)
        ttk.Label(
            mapping_status_card,
            textvariable=self.mapping_status_var,
            justify=tk.LEFT,
            wraplength=300,
        ).pack(anchor="w", fill=tk.X, pady=(4, 0))
        ttk.Label(
            validation_card,
            textvariable=self.validation_status_var,
            justify=tk.LEFT,
            wraplength=300,
        ).pack(anchor="w", fill=tk.X)
        tk.Label(
            settings_frame,
            textvariable=self.settings_error_var,
            fg="#9b0000",
            anchor="w",
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(8, 0))

        config_help_frame = ttk.LabelFrame(settings_frame, text="Config Workflow", padding=8)
        config_help_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(
            config_help_frame,
            text="Apply = session only · Save = write current form to disk · Reload = read from disk · Reset = restore built-in defaults",
            justify=tk.LEFT,
            wraplength=900,
        ).pack(fill=tk.X)
        ttk.Label(
            config_help_frame,
            text="After a dry-run probe, use tune_probe_config.py to generate evidence-based threshold suggestions.",
            justify=tk.LEFT,
            wraplength=900,
        ).pack(fill=tk.X, pady=(4, 0))
        ttk.Label(
            config_help_frame,
            text="Before retraining, use analyze_training_data.py to spot weak label/session/power-saver coverage.",
            justify=tk.LEFT,
            wraplength=900,
        ).pack(fill=tk.X, pady=(4, 0))

        config_path_row = ttk.Frame(settings_frame)
        config_path_row.pack(fill=tk.X, pady=(8, 8))
        ttk.Label(config_path_row, text="Config Path").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(config_path_row, textvariable=self.config_path_var).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=(0, 6),
        )
        ttk.Button(
            config_path_row,
            text="Browse",
            command=self._browse_config_path,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            config_path_row,
            text="Use Default",
            command=self._reset_config_path,
        ).pack(side=tk.LEFT)

        columns = ttk.Frame(settings_frame)
        columns.pack(fill=tk.BOTH, expand=True)
        columns.columnconfigure(0, weight=1)
        columns.columnconfigure(1, weight=1)

        left_col = ttk.Frame(columns)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right_col = ttk.Frame(columns)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        thresholds = ttk.LabelFrame(
            left_col,
            text="Switching Thresholds",
            padding=8,
        )
        thresholds.pack(fill=tk.X, pady=(0, 8))
        thresholds.columnconfigure(2, weight=1)

        ttk.Label(
            thresholds,
            text="Use typical ranges for stable demos; you can still tune outside them.",
            justify=tk.LEFT,
            wraplength=420,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

        for row, key in enumerate(self.threshold_order, start=1):
            spec = self.threshold_specs[key]
            ttk.Label(thresholds, text=spec["label"]).grid(
                row=row,
                column=0,
                sticky="w",
                padx=(0, 8),
                pady=3,
            )
            ttk.Entry(
                thresholds,
                textvariable=spec["var"],
                width=14,
            ).grid(row=row, column=1, sticky="w", pady=3)
            status_label = tk.Label(
                thresholds,
                textvariable=self.threshold_status_vars[key],
                anchor="w",
                justify=tk.LEFT,
                fg="#475569",
            )
            status_label.grid(row=row, column=2, sticky="w", padx=(10, 0), pady=3)
            self.threshold_status_labels[key] = status_label

        hotkeys_frame = ttk.LabelFrame(
            left_col,
            text="Profile Hotkeys",
            padding=8,
        )
        hotkeys_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            hotkeys_frame,
            text="Blank is OK for oracle-backed switching.",
            justify=tk.LEFT,
            wraplength=420,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        for row, profile in enumerate(VALID_PROFILES, start=1):
            ttk.Label(hotkeys_frame, text=profile).grid(
                row=row,
                column=0,
                sticky="w",
                padx=(0, 8),
                pady=2,
            )
            ttk.Entry(
                hotkeys_frame,
                textvariable=self.hotkey_vars[profile],
                width=28,
            ).grid(row=row, column=1, sticky="w", pady=2)

        mapping_frame = ttk.LabelFrame(
            right_col,
            text="Label to Profile Mapping",
            padding=8,
        )
        mapping_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        add_map = ttk.Frame(mapping_frame)
        add_map.pack(fill=tk.X)
        ttk.Label(add_map, text="New Label").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(add_map, textvariable=self.new_label_var, width=18).pack(
            side=tk.LEFT,
            padx=(0, 6),
        )
        ttk.Button(add_map, text="Add", command=self._add_label_mapping).pack(
            side=tk.LEFT
        )
        ttk.Button(
            add_map,
            text="Import Model Labels",
            command=self._import_model_labels,
        ).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(
            mapping_frame,
            text="Import classes.json to prefill mappings before you start the live demo.",
            justify=tk.LEFT,
            wraplength=420,
        ).pack(fill=tk.X, pady=(8, 0))

        mapping_tools = ttk.Frame(mapping_frame)
        mapping_tools.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(mapping_tools, text="Filter").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(mapping_tools, textvariable=self.mapping_filter_var, width=18).pack(
            side=tk.LEFT,
            padx=(0, 6),
        )
        ttk.Button(
            mapping_tools,
            text="Clear",
            command=self._clear_mapping_filter,
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(mapping_tools, text="Bulk").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Combobox(
            mapping_tools,
            textvariable=self.bulk_profile_var,
            values=list(VALID_PROFILES),
            state="readonly",
            width=12,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            mapping_tools,
            text="Apply to Visible",
            command=self._apply_bulk_profile_to_visible,
        ).pack(side=tk.LEFT)

        ttk.Label(
            mapping_frame,
            textvariable=self.mapping_filter_status_var,
            justify=tk.LEFT,
            wraplength=420,
        ).pack(fill=tk.X, pady=(6, 0))

        mapping_list_frame = ttk.Frame(mapping_frame)
        mapping_list_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.mapping_canvas = tk.Canvas(
            mapping_list_frame,
            height=220,
            highlightthickness=0,
            borderwidth=0,
        )
        mapping_scrollbar = ttk.Scrollbar(
            mapping_list_frame,
            orient="vertical",
            command=self.mapping_canvas.yview,
        )
        self.mapping_rows_frame = ttk.Frame(self.mapping_canvas)
        self.mapping_canvas_window = self.mapping_canvas.create_window(
            (0, 0),
            window=self.mapping_rows_frame,
            anchor="nw",
        )
        self.mapping_canvas.configure(yscrollcommand=mapping_scrollbar.set)
        self.mapping_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        mapping_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.mapping_rows_frame.bind(
            "<Configure>",
            lambda _e: self.mapping_canvas.configure(
                scrollregion=self.mapping_canvas.bbox("all")
            ),
        )
        self.mapping_canvas.bind(
            "<Configure>",
            lambda e: self.mapping_canvas.itemconfigure(
                self.mapping_canvas_window,
                width=e.width,
            ),
        )

        actions_row = ttk.Frame(settings_frame)
        actions_row.pack(fill=tk.X, pady=(8, 0))
        actions_row.columnconfigure(0, weight=1)
        actions_row.columnconfigure(1, weight=1)

        file_actions = ttk.LabelFrame(actions_row, text="File Actions", padding=8)
        file_actions.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        preset_actions = ttk.LabelFrame(actions_row, text="Session Presets", padding=8)
        preset_actions.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        self.apply_settings_button = ttk.Button(
            file_actions,
            text="Apply Session",
            command=self.apply_settings_session,
        )
        self.apply_settings_button.pack(side=tk.LEFT, padx=(0, 6))
        self.save_settings_button = ttk.Button(
            file_actions,
            text="Save",
            command=self.save_settings_file,
        )
        self.save_settings_button.pack(side=tk.LEFT, padx=(0, 6))
        self.reload_settings_button = ttk.Button(
            file_actions,
            text="Reload",
            command=self.reload_settings_file,
        )
        self.reload_settings_button.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            file_actions,
            text="Reset Defaults",
            command=self.reset_settings_defaults,
        ).pack(side=tk.LEFT)

        for preset_name in PRESET_SWITCHING:
            ttk.Button(
                preset_actions,
                text=preset_name,
                command=lambda n=preset_name: self._apply_preset(n),
            ).pack(side=tk.LEFT, padx=(0, 4))

        ttk.Label(
            settings_frame,
            text="Presets change the current session immediately. Use Save to persist them to disk.",
            justify=tk.LEFT,
            wraplength=900,
        ).pack(fill=tk.X, pady=(8, 0))

        ttk.Label(
            log_tab,
            text="Diagnostics are available when you need them, but kept visually out of the main demo flow.",
            justify=tk.LEFT,
            wraplength=900,
        ).pack(fill=tk.X, pady=(0, 8))
        ttk.Label(
            log_tab,
            textvariable=self.backend_info_var,
            justify=tk.LEFT,
            wraplength=900,
        ).pack(fill=tk.X, pady=(0, 8))

        log_frame = ttk.LabelFrame(log_tab, text="Diagnostics Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)

        log_controls = ttk.Frame(log_frame)
        log_controls.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(log_controls, text="View").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Combobox(
            log_controls,
            textvariable=self.log_filter_var,
            values=["All", "Switches", "Warnings"],
            width=12,
            state="readonly",
        ).pack(side=tk.LEFT)
        ttk.Button(
            log_controls,
            text="Clear",
            command=self._clear_log,
        ).pack(side=tk.LEFT, padx=(8, 0))

        self.log_filter_var.trace_add("write", lambda *_: self._refresh_log_view())

        self.log_box = tk.Text(log_frame, height=16, wrap="word")
        self.log_box.pack(fill=tk.BOTH, expand=True)
        self.log_box.configure(state=tk.DISABLED)

        self._render_label_mapping_rows()
        self._update_backend_info()

    def _set_settings_error(self, message: str):
        self.settings_error_var.set(message)
        if message:
            self.config_hint_var.set(f"Config issue: {message}")
        elif self._settings_dirty:
            self.config_hint_var.set("Configuration has unsaved changes.")
        else:
            self.config_hint_var.set("Config ready.")

    def _reset_session_outcomes(self):
        self._last_action_text = "no profile action yet"
        self._auto_switch_count = 0
        self._manual_switch_count = 0
        self._failed_switch_count = 0
        self._gate_reason_counter.clear()
        self._update_session_outcomes()

    def _record_gate_observation(self, reason: str):
        norm = str(reason).strip().lower()
        if not norm or norm in {"ready", "already-applied"}:
            return
        self._gate_reason_counter[norm] += 1
        self._update_session_outcomes()

    @staticmethod
    def _compact_text(text: str, max_len: int = 42) -> str:
        clean = " ".join(str(text).split())
        if len(clean) <= max_len:
            return clean
        return clean[: max_len - 1].rstrip() + "…"

    def _format_action_age(self) -> str:
        if self._last_action_ts <= 0.0:
            return "no timestamp"
        age_s = max(0.0, time.time() - self._last_action_ts)
        if age_s < 60.0:
            return f"{int(age_s)}s ago"
        return f"{int(age_s // 60)}m ago"

    def _update_snapshot_cards(self):
        applied = self.oracle.get_profile() or self.current_profile or "-"
        target = self._last_recommend_target or "-"
        self.target_profile_card_var.set(str(target))
        self.applied_profile_card_var.set(str(applied))
        self.gate_card_var.set(self._compact_text(self._last_gate_text, max_len=28))
        action_text = self._compact_text(self._last_action_text, max_len=30)
        age = self._format_action_age()
        self.last_action_card_var.set(action_text if self._last_action_ts <= 0.0 else f"{action_text} · {age}")

    @staticmethod
    def _describe_apply_result(result) -> str:
        if getattr(result, "verified", None) is True:
            observed = getattr(result, "observed_profile", None) or getattr(result, "applied_profile", None)
            return f"verified {observed}"
        if getattr(result, "verified", None) is False:
            observed = getattr(result, "observed_profile", None) or "unknown"
            return f"mismatch observed {observed}"
        return "reported only"

    def _set_last_action(self, message: str):
        self._last_action_text = message.strip() or "no profile action yet"
        self._last_action_ts = time.time()
        self._update_snapshot_cards()
        self._update_session_outcomes()

    def _summarize_gate_pressure(self) -> str:
        if not self._gate_reason_counter:
            return "none"
        top = self._gate_reason_counter.most_common(2)
        return ", ".join(f"{reason}×{count}" for reason, count in top)

    def _update_session_outcomes(self):
        backend = self.oracle.backend_name
        age = self._format_action_age()
        actuation = f"{backend} | {self._last_action_text}"
        if self._last_action_ts > 0.0:
            actuation += f" | last action {age}"
        self.actuation_var.set(actuation)
        self.session_stats_var.set(
            f"auto {self._auto_switch_count} | manual {self._manual_switch_count} | "
            f"failed {self._failed_switch_count} | blocks {self._summarize_gate_pressure()}"
        )
        self._update_snapshot_cards()

    def _update_backend_info(self):
        try:
            diag = self.oracle.diagnostics()
            backend_name = diag.backend_name
            selected = diag.selected_profile
            detail_parts: list[str] = []
            active_name = str(diag.details.get("active_scheme_name", "")).strip()
            if active_name:
                detail_parts.append(active_name)
            mode = str(diag.details.get("mode", "")).strip()
            if mode:
                detail_parts.append(mode)
            tuning = diag.details.get("profile_tuning")
            if isinstance(tuning, dict) and tuning:
                detail_parts.append("tuning active")
            detail_text = " | ".join(detail_parts)
            base = f"Backend: {backend_name} | Selected profile: {selected}"
            self.backend_info_var.set(base if not detail_text else f"{base} | {detail_text}")
        except Exception as e:
            self.backend_info_var.set(f"Backend: unavailable ({type(e).__name__}: {e})")

    def _resolve_config_path(self) -> Path:
        raw = self.config_path_var.get().strip()
        if not raw:
            raise RuntimeError("Config file path cannot be empty.")
        path = Path(raw).expanduser()
        if path.suffix and path.suffix.lower() != ".json":
            raise RuntimeError("Config file path must use a .json extension.")
        return path

    def _browse_config_path(self):
        initial = self.config_path_var.get().strip() or str(Path(CONFIG_FILE))
        initial_path = Path(initial).expanduser()
        selected = filedialog.asksaveasfilename(
            title="Select PerfAnalyze config file",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(initial_path.parent),
            initialfile=initial_path.name,
        )
        if selected:
            self.config_path_var.set(selected)
            self._update_config_line()

    def _reset_config_path(self):
        self.config_path = Path(CONFIG_FILE)
        self.config_path_var.set(str(self.config_path))
        self._update_config_line()
        self.log(f"config path reset: {self.config_path}")

    def _short_path(self, path: Path, max_len: int = 68) -> str:
        text = str(path)
        if len(text) <= max_len:
            return text
        parts = path.parts
        if len(parts) >= 2:
            return f"...\\{parts[-2]}\\{parts[-1]}"
        return f"...\\{path.name}"

    def _update_config_line(self):
        active_path = self.config_path_var.get().strip() or CONFIG_FILE
        path = Path(active_path).expanduser()
        self.config_var.set(
            f"source={self.config_source} | path={self._short_path(path)}"
        )
        self._update_path_status()
        self._update_config_summary()

    def _update_path_status(self):
        raw = self.config_path_var.get().strip() or CONFIG_FILE
        path = Path(raw).expanduser()
        if path.exists():
            self.path_status_var.set(
                f"path status: existing file at {self._short_path(path)}"
            )
        else:
            self.path_status_var.set(
                "path status: file does not exist yet; it will be created on save at "
                f"{self._short_path(path)}"
            )

    @staticmethod
    def _set_badge(widget: tk.Label, text: str, bg: str, fg: str = "white"):
        widget.configure(
            text=text,
            bg=bg,
            fg=fg,
            relief="ridge",
            bd=1,
            font=("Segoe UI", 9, "bold"),
        )

    def _update_dashboard_badges(self):
        mode_text = "Mode MANUAL" if self.manual_override else "Mode AUTO"
        mode_color = "#d97706" if self.manual_override else "#2563eb"
        self.mode_badge_var.set(mode_text)
        self._set_badge(self.mode_badge, mode_text, mode_color)

        execution_text = "Exec DRY RUN" if self.dry_run_var.get() else "Exec LIVE"
        execution_color = "#7c3aed" if self.dry_run_var.get() else "#15803d"
        self.execution_badge_var.set(execution_text)
        self._set_badge(self.execution_badge, execution_text, execution_color)

        config_text = "Config UNSAVED" if self._settings_dirty else "Config SAVED"
        config_color = "#b45309" if self._settings_dirty else "#15803d"
        self.config_badge_var.set(config_text)
        self._set_badge(self.config_badge, config_text, config_color)

        model_ready = bool(self.classes)
        model_text = "Model READY" if model_ready else "Model MISSING"
        model_color = "#15803d" if model_ready else "#6b7280"
        self.model_badge_var.set(model_text)
        self._set_badge(self.model_badge, model_text, model_color)

    def _update_config_summary(self):
        s = self.runtime_config.switching
        dirty = "unsaved" if self._settings_dirty else "saved"
        self.config_summary_var.set(
            f"poll={s.poll_interval_sec:g}s | "
            f"window={s.stability_window}+{s.downshift_extra_window}↓ | "
            f"conf>={s.confidence_min:g} | "
            f"margin>={s.confidence_margin:g} | "
            f"gap={s.min_switch_interval_sec:g}s | "
            f"hold={s.downshift_hold_sec:g}s | "
            f"ema={s.probability_ema_alpha:g} | "
            f"state={dirty}"
        )
        self._update_dashboard_badges()

    def _validate_threshold_field(self, key: str) -> tuple[bool, str, str]:
        spec = self.threshold_specs[key]
        raw = spec["var"].get().strip()
        if not raw:
            return False, spec["invalid_text"], "#b91c1c"

        try:
            value = int(raw) if spec["kind"] == "int" else float(raw)
        except ValueError:
            return False, spec["invalid_text"], "#b91c1c"

        min_value = spec["min"]
        max_value = spec["max"]
        if min_value is not None and value < min_value:
            return False, spec["invalid_text"], "#b91c1c"
        if max_value is not None and value > max_value:
            return False, spec["invalid_text"], "#b91c1c"

        typical_min, typical_max = spec["typical"]
        if typical_min <= value <= typical_max:
            return True, f"Good · {spec['typical_text']}", "#166534"
        return True, f"Valid · {spec['typical_text']}", "#b45309"

    def _update_threshold_field_statuses(self):
        for key in self.threshold_order:
            valid, text, color = self._validate_threshold_field(key)
            self.threshold_status_vars[key].set(text)
            if key in self.threshold_status_labels:
                self.threshold_status_labels[key].configure(fg=color)

    def _update_validation_state(self):
        form_error = ""
        save_error = ""

        self._update_threshold_field_statuses()

        try:
            self._build_config_from_form()
        except RuntimeError as e:
            form_error = str(e)

        try:
            self._resolve_config_path()
        except RuntimeError as e:
            save_error = str(e)

        self._form_is_valid = not form_error
        self._save_is_valid = self._form_is_valid and not save_error

        if form_error:
            self.validation_status_var.set(f"Fix form values before apply/save: {form_error}")
        elif save_error:
            self.validation_status_var.set(
                f"Apply is ready. Save/Reload need a valid path: {save_error}"
            )
        else:
            self.validation_status_var.set("Apply, Save, and Reload are ready.")

        if hasattr(self, "apply_settings_button"):
            self.apply_settings_button.state(
                ["!disabled"] if self._form_is_valid else ["disabled"]
            )
        if hasattr(self, "save_settings_button"):
            self.save_settings_button.state(
                ["!disabled"] if self._save_is_valid else ["disabled"]
            )
        if hasattr(self, "reload_settings_button"):
            self.reload_settings_button.state(
                ["!disabled"] if not save_error else ["disabled"]
            )

    def _set_gate_status(self, text: str):
        self._last_gate_text = text
        self.switch_gate_var.set(f"Readiness: {text}")
        self._update_snapshot_cards()

    def _update_recommendation_line(self):
        applied = self.oracle.get_profile() or self.current_profile or "-"
        target = self._last_recommend_target or "-"
        pred_label = self._last_pred_label or "-"
        self.switch_gate_var.set(f"Readiness: {self._last_gate_text}")

        if self.manual_override:
            self.recommendation_var.set(
                f"Manual override active → keep {self.manual_override}."
            )
            self._update_snapshot_cards()
            return

        if pred_label == "-" or target == "-":
            self.recommendation_var.set("Waiting for a confident signal before changing profiles.")
            self._update_snapshot_cards()
            return

        if self._last_pred_confident:
            self.recommendation_var.set(
                f"Recommend {target} from '{pred_label}'. Current applied profile: {applied}."
            )
        else:
            self.recommendation_var.set(
                f"Hold {applied} while '{pred_label}' builds enough stability for {target}."
            )
        self._update_snapshot_cards()

    def _set_live_summary(
        self,
        *,
        pred_label: str = "-",
        confidence: float = 0.0,
        margin: float = 0.0,
        target_profile: str = "-",
        confident: bool = False,
        stable_count: int = 0,
        window: int = 0,
    ):
        self._last_pred_label = pred_label
        self._last_pred_confident = confident
        self._last_recommend_target = target_profile
        stability = (
            self._format_stability_display(stable_count, window)
            if window > 0
            else "-"
        )
        self.live_summary_var.set(
            f"{pred_label} | conf {confidence:.2f} | margin {margin:.2f} | "
            f"{'reliable' if confident else 'watching'} | stable {stability}"
        )
        self._update_recommendation_line()

    def _update_model_info(self):
        if self.classes:
            labels = ", ".join(self.classes)
            self.model_info_var.set(
                f"{MODEL_FILE} | classes: {labels}"
            )
        else:
            self.model_info_var.set(
                f"waiting for {MODEL_FILE} and {CLASSES_FILE}"
            )
        self._update_mapping_status()
        self._update_dashboard_badges()

    def _update_hotkey_status(self):
        configured = [
            profile for profile in VALID_PROFILES if self.hotkey_vars[profile].get().strip()
        ]
        missing = [
            profile for profile in VALID_PROFILES if not self.hotkey_vars[profile].get().strip()
        ]
        if not missing:
            self.hotkey_status_var.set("Hotkeys: all profiles have explicit hotkey bindings.")
        elif not configured:
            self.hotkey_status_var.set(
                "Hotkeys: all bindings are blank; this is fine for oracle/demo-backed switching."
            )
        else:
            self.hotkey_status_var.set(
                "Hotkeys: configured for "
                + ", ".join(configured)
                + " | blank for "
                + ", ".join(missing)
            )

    def _update_mapping_status(self):
        labels = sorted(self.label_map_vars.keys())
        self._update_hotkey_status()
        if not self.classes:
            self.mapping_status_var.set(
                f"Mappings: {len(labels)} configured labels. Model classes will be checked after loading artifacts."
            )
            self.health_var.set(
                f"config editable, {len(labels)} label mappings loaded, model artifacts not loaded yet"
            )
            return

        missing = [
            label for label in self.classes if label.strip().lower() not in self.label_map_vars
        ]
        if missing:
            self.mapping_status_var.set(
                "Mappings: model classes missing explicit entries -> " + ", ".join(sorted(missing))
            )
            self.health_var.set(
                "model loaded, but some classes still rely on default mappings; review Configuration"
            )
        else:
            self.mapping_status_var.set(
                f"Mappings: all {len(self.classes)} model classes have profile assignments."
            )
            self.health_var.set(
                "model loaded and configuration covers all known classes"
            )

    def _bind_form_traces(self):
        watched_vars = [
            self.poll_interval_var,
            self.stability_window_var,
            self.confidence_min_var,
            self.confidence_margin_var,
            self.min_switch_interval_var,
            self.probability_ema_alpha_var,
            self.downshift_extra_window_var,
            self.downshift_hold_sec_var,
            self.config_path_var,
            self.new_label_var,
            *self.hotkey_vars.values(),
        ]
        for var in watched_vars:
            var.trace_add("write", self._on_form_edited)
        self.mapping_filter_var.trace_add("write", self._on_mapping_filter_edited)
        self.dry_run_var.trace_add("write", self._on_runtime_option_edited)

    def _on_runtime_option_edited(self, *_):
        self._set_profile_line(target=str(self.manual_override or self.current_profile or "-"))
        self._update_dashboard_badges()

    def _on_form_edited(self, *_):
        if self._suspend_form_traces:
            return
        self._settings_dirty = True
        self._update_config_summary()
        self._update_hotkey_status()
        self._update_validation_state()
        if not self.settings_error_var.get().strip():
            self.config_hint_var.set("Configuration has unsaved changes.")

    def _on_mapping_filter_edited(self, *_):
        self._render_label_mapping_rows()

    def _mark_settings_clean(self, hint: str | None = None):
        self._settings_dirty = False
        self._update_config_summary()
        self._update_validation_state()
        self._update_dashboard_badges()
        if not self.settings_error_var.get().strip():
            self.config_hint_var.set(hint or "Config ready.")

    def _ensure_label_map_rows(self, labels):
        for raw_label in labels:
            norm = str(raw_label).strip().lower()
            if not norm:
                continue
            if norm not in self.label_map_vars:
                self.label_map_vars[norm] = tk.StringVar(value="balanced")
                self.label_map_vars[norm].trace_add("write", self._on_form_edited)

    def _get_filtered_mapping_labels(self) -> list[str]:
        labels = sorted(self.label_map_vars.keys())
        query = self.mapping_filter_var.get().strip().lower()
        if not query:
            return labels
        return [label for label in labels if query in label]

    def _refresh_mapping_filter_status(self, visible_labels: list[str], total_labels: int):
        query = self.mapping_filter_var.get().strip()
        imported_visible = sum(1 for label in visible_labels if label in self.auto_imported_labels)
        if total_labels == 0:
            self.mapping_filter_status_var.set("No label mappings configured yet.")
            return
        if query:
            self.mapping_filter_status_var.set(
                f"Showing {len(visible_labels)} of {total_labels} labels for '{query}'"
                + (f" · {imported_visible} imported" if visible_labels else "")
            )
        else:
            self.mapping_filter_status_var.set(
                f"Showing all {total_labels} labels"
                + (f" · {imported_visible} imported" if imported_visible else "")
            )

    def _clear_mapping_filter(self):
        self.mapping_filter_var.set("")

    def _apply_bulk_profile_to_visible(self):
        profile = self.bulk_profile_var.get().strip().lower()
        if profile not in VALID_PROFILES:
            self._set_settings_error("Choose a valid bulk profile before applying.")
            return

        visible_labels = self._get_filtered_mapping_labels()
        if not visible_labels:
            self._set_settings_error("No visible labels match the current filter.")
            return

        self._suspend_form_traces = True
        try:
            for label in visible_labels:
                self.label_map_vars[label].set(profile)
        finally:
            self._suspend_form_traces = False

        self._settings_dirty = True
        self._set_settings_error("")
        self._update_mapping_status()
        self._update_config_summary()
        self._update_validation_state()
        self._render_label_mapping_rows()
        self.config_hint_var.set(
            f"Applied '{profile}' to {len(visible_labels)} visible labels."
        )
        self.log(
            f"bulk mapping applied -> {profile} for {len(visible_labels)} labels"
        )

    def _render_label_mapping_rows(self):
        for child in self.mapping_rows_frame.winfo_children():
            child.destroy()

        all_labels = sorted(self.label_map_vars.keys())
        labels = self._get_filtered_mapping_labels()
        self._refresh_mapping_filter_status(labels, len(all_labels))
        if not all_labels:
            ttk.Label(
                self.mapping_rows_frame,
                text="No label mappings configured.",
            ).grid(row=0, column=0, sticky="w")
            return
        if not labels:
            ttk.Label(
                self.mapping_rows_frame,
                text="No labels match the current filter.",
            ).grid(row=0, column=0, sticky="w")
            return

        ttk.Label(self.mapping_rows_frame, text="Label").grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=(0, 4),
        )
        ttk.Label(self.mapping_rows_frame, text="Profile").grid(
            row=0,
            column=1,
            sticky="w",
            padx=(0, 8),
            pady=(0, 4),
        )
        ttk.Label(self.mapping_rows_frame, text="Source").grid(
            row=0,
            column=2,
            sticky="w",
            padx=(0, 8),
            pady=(0, 4),
        )

        for row, label in enumerate(labels, start=1):
            ttk.Label(self.mapping_rows_frame, text=label).grid(
                row=row,
                column=0,
                sticky="w",
                padx=(0, 8),
                pady=2,
            )
            ttk.Combobox(
                self.mapping_rows_frame,
                textvariable=self.label_map_vars[label],
                values=list(VALID_PROFILES),
                state="readonly",
                width=14,
            ).grid(row=row, column=1, sticky="w", pady=2)

            source_text = "Imported" if label in self.auto_imported_labels else "Config"
            source_color = "#7c3aed" if label in self.auto_imported_labels else "#475569"
            tk.Label(
                self.mapping_rows_frame,
                text=source_text,
                fg=source_color,
                anchor="w",
                justify=tk.LEFT,
            ).grid(row=row, column=2, padx=(8, 8), pady=2, sticky="w")

            remove_enabled = len(all_labels) > 1
            ttk.Button(
                self.mapping_rows_frame,
                text="Remove",
                command=lambda l=label: self._remove_label_mapping(l),
                state=tk.NORMAL if remove_enabled else tk.DISABLED,
            ).grid(row=row, column=3, padx=(8, 0), pady=2, sticky="w")

    def _load_classes_from_artifact(self) -> list[str]:
        classes_path = Path(CLASSES_FILE)
        if not classes_path.exists():
            raise RuntimeError(f"'{CLASSES_FILE}' not found. Train the model first.")

        with classes_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list) or not raw:
            raise RuntimeError(f"'{CLASSES_FILE}' must contain a non-empty list of labels.")

        labels: list[str] = []
        seen: set[str] = set()
        for item in raw:
            norm = str(item).strip().lower()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            labels.append(norm)
        if not labels:
            raise RuntimeError(f"'{CLASSES_FILE}' does not contain any usable labels.")
        return labels

    def _import_model_labels(self):
        try:
            imported_labels = self._load_classes_from_artifact()
        except RuntimeError as e:
            self._set_settings_error(str(e))
            self.log(f"label import failed: {e}", level="warn")
            return

        self.classes = imported_labels
        before = set(self.label_map_vars.keys())
        self._ensure_label_map_rows(imported_labels)
        added = sorted(set(self.label_map_vars.keys()) - before)
        self.auto_imported_labels.update(added)
        self._render_label_mapping_rows()
        self._update_model_info()
        self._update_mapping_status()
        self._update_validation_state()
        if added:
            self._settings_dirty = True
            self._update_config_summary()
            self._set_settings_error("")
            self.config_hint_var.set(
                f"Imported {len(added)} model labels. Review profile assignments before saving."
            )
            self.log(
                "imported model labels from classes.json: " + ", ".join(added)
            )
        else:
            self._set_settings_error("")
            self.config_hint_var.set("Model labels already match the current mapping table.")
            self.log("model labels already present in mapping table")

    def _add_label_mapping(self):
        label = self.new_label_var.get().strip().lower()
        if not label:
            self._set_settings_error("New label cannot be empty.")
            return
        if label in self.label_map_vars:
            self._set_settings_error(f"Label '{label}' already exists.")
            return

        self.label_map_vars[label] = tk.StringVar(value="balanced")
        self.label_map_vars[label].trace_add("write", self._on_form_edited)
        self.auto_imported_labels.discard(label)
        self.new_label_var.set("")
        self._settings_dirty = True
        self._set_settings_error("")
        self._render_label_mapping_rows()
        self._update_mapping_status()
        self._update_config_summary()
        self._update_validation_state()
        self.log(f"added label mapping row: {label}")

    def _remove_label_mapping(self, label: str):
        if len(self.label_map_vars) <= 1:
            self._set_settings_error("At least one label mapping is required.")
            return
        self.label_map_vars.pop(label, None)
        self.auto_imported_labels.discard(label)
        self._settings_dirty = True
        self._set_settings_error("")
        self._render_label_mapping_rows()
        self._update_mapping_status()
        self._update_config_summary()
        self._update_validation_state()
        self.log(f"removed label mapping row: {label}")

    def _sync_form_from_runtime_config(self):
        self._suspend_form_traces = True
        try:
            self.auto_imported_labels.clear()
            s = self.runtime_config.switching
            self.poll_interval_var.set(f"{s.poll_interval_sec:g}")
            self.stability_window_var.set(str(s.stability_window))
            self.confidence_min_var.set(f"{s.confidence_min:g}")
            self.confidence_margin_var.set(f"{s.confidence_margin:g}")
            self.min_switch_interval_var.set(f"{s.min_switch_interval_sec:g}")
            self.probability_ema_alpha_var.set(f"{s.probability_ema_alpha:g}")
            self.downshift_extra_window_var.set(str(s.downshift_extra_window))
            self.downshift_hold_sec_var.set(f"{s.downshift_hold_sec:g}")
            self.config_path_var.set(str(self.config_path))

            self._ensure_label_map_rows(self.runtime_config.label_to_profile.keys())
            for label, var in self.label_map_vars.items():
                var.set(self.runtime_config.label_to_profile.get(label, "balanced"))
            self._render_label_mapping_rows()
            self._update_model_info()
            self._update_mapping_status()

            for profile in VALID_PROFILES:
                self.hotkey_vars[profile].set(
                    self.runtime_config.profile_hotkeys.get(profile, "")
                )
        finally:
            self._suspend_form_traces = False
        self._update_hotkey_status()
        self._update_validation_state()

    def _parse_float(self, raw: str, field: str) -> float:
        try:
            return float(raw.strip())
        except ValueError as e:
            raise RuntimeError(f"{field} must be a number.") from e

    def _parse_int(self, raw: str, field: str) -> int:
        value = raw.strip()
        if not value:
            raise RuntimeError(f"{field} must be an integer.")
        try:
            return int(value)
        except ValueError as e:
            raise RuntimeError(f"{field} must be an integer.") from e

    def _build_config_from_form(self):
        switching = {
            "poll_interval_sec": self._parse_float(
                self.poll_interval_var.get(),
                "Poll Interval",
            ),
            "stability_window": self._parse_int(
                self.stability_window_var.get(),
                "Stability Window",
            ),
            "confidence_min": self._parse_float(
                self.confidence_min_var.get(),
                "Confidence Min",
            ),
            "confidence_margin": self._parse_float(
                self.confidence_margin_var.get(),
                "Confidence Margin",
            ),
            "min_switch_interval_sec": self._parse_float(
                self.min_switch_interval_var.get(),
                "Min Switch Interval",
            ),
            "probability_ema_alpha": self._parse_float(
                self.probability_ema_alpha_var.get(),
                "Probability EMA Alpha",
            ),
            "downshift_extra_window": self._parse_int(
                self.downshift_extra_window_var.get(),
                "Downshift Extra Window",
            ),
            "downshift_hold_sec": self._parse_float(
                self.downshift_hold_sec_var.get(),
                "Downshift Hold",
            ),
        }

        label_to_profile = {
            label: var.get().strip().lower()
            for label, var in self.label_map_vars.items()
        }
        profile_hotkeys = {
            profile: self.hotkey_vars[profile].get().strip()
            for profile in VALID_PROFILES
        }

        raw = {
            "switching": switching,
            "label_to_profile": label_to_profile,
            "profile_hotkeys": profile_hotkeys,
        }
        return build_config(raw)

    def _apply_runtime_config(self, cfg, source: str):
        self.runtime_config = cfg
        self.config_source = source
        self.last_candidate = None
        self.stable_count = 0
        self.prediction_smoother = ProbabilitySmoother(
            cfg.switching.probability_ema_alpha
        )
        self._set_settings_error("")
        self._update_config_line()
        self._set_profile_line(target=self.manual_override or "-")
        self._set_gate_status("configuration updated")
        self._set_live_summary(target_profile=str(self.manual_override or "-"))
        self._mark_settings_clean()

    def apply_settings_session(self):
        try:
            cfg = self._build_config_from_form()
        except RuntimeError as e:
            self._set_settings_error(str(e))
            self.log(f"settings apply failed: {e}", level="warn")
            return

        self._apply_runtime_config(cfg, source="session (unsaved)")
        self._update_mapping_status()
        self._settings_dirty = True
        self._update_config_summary()
        self._update_validation_state()
        self.config_hint_var.set("Session configuration applied. Save if you want it persisted.")
        self.log("settings applied to current session")

    def save_settings_file(self):
        try:
            cfg = self._build_config_from_form()
            save_path = self._resolve_config_path()
            saved_path = write_config(cfg, save_path)
        except (RuntimeError, OSError) as e:
            self._set_settings_error(str(e))
            self.log(f"save failed: {e}", level="warn")
            return

        self.config_path = saved_path
        self.config_path_var.set(str(saved_path))
        self._apply_runtime_config(cfg, source=str(saved_path))
        self._mark_settings_clean(f"Configuration saved to {saved_path}")
        self.log(f"settings saved: {saved_path}")

    def reload_settings_file(self):
        try:
            load_path = self._resolve_config_path()
            result = load_config(load_path)
        except RuntimeError as e:
            self._set_settings_error(str(e))
            self.log(f"reload failed: {e}", level="warn")
            return

        self.runtime_config = result.config
        self.config_source = result.source
        self.config_path = result.path or load_path
        self.config_path_var.set(str(self.config_path))
        self._sync_form_from_runtime_config()
        self.last_candidate = None
        self.stable_count = 0
        self._set_settings_error("")
        self._update_config_line()
        self._set_profile_line(target=self.manual_override or "-")
        self._mark_settings_clean(f"Configuration reloaded from {self.config_path}")
        self.log(f"settings reloaded ({self.config_source})")

    def reset_settings_defaults(self):
        cfg = default_config()
        self._apply_runtime_config(cfg, source="defaults (session reset)")
        self._sync_form_from_runtime_config()
        self._update_mapping_status()
        self._mark_settings_clean(
            "Built-in defaults restored for this session. Use Save if you want to overwrite a file."
        )
        self.log("settings reset to defaults for current session")

    def _apply_preset(self, preset_name: str):
        preset = PRESET_SWITCHING.get(preset_name)
        if not preset:
            self._set_settings_error(f"Unknown preset: {preset_name}")
            return

        self.poll_interval_var.set(f"{preset['poll_interval_sec']:g}")
        self.stability_window_var.set(str(int(preset["stability_window"])))
        self.confidence_min_var.set(f"{preset['confidence_min']:g}")
        self.confidence_margin_var.set(f"{preset['confidence_margin']:g}")
        self.min_switch_interval_var.set(f"{preset['min_switch_interval_sec']:g}")
        self.probability_ema_alpha_var.set(f"{preset['probability_ema_alpha']:g}")
        self.downshift_extra_window_var.set(str(int(preset['downshift_extra_window'])))
        self.downshift_hold_sec_var.set(f"{preset['downshift_hold_sec']:g}")

        self.log(f"preset selected: {preset_name}")
        self.apply_settings_session()

    def _filter_allows(self, level: str) -> bool:
        filt = self.log_filter_var.get()
        if filt == "All":
            return True
        if filt == "Switches":
            return level == "switch"
        if filt == "Warnings":
            return level in {"warn", "error"}
        return True

    def _refresh_log_view(self):
        self.log_box.configure(state=tk.NORMAL)
        self.log_box.delete("1.0", tk.END)
        for level, line in self.log_entries:
            if self._filter_allows(level):
                self.log_box.insert(tk.END, line)
        self.log_box.see(tk.END)
        self.log_box.configure(state=tk.DISABLED)

    def _clear_log(self):
        self.log_entries.clear()
        self._refresh_log_view()
        self.log("log cleared")

    def log(self, message: str, level: str = "info"):
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {message}\n"
        self.log_entries.append((level, line))
        if len(self.log_entries) > MAX_LOG_LINES:
            self.log_entries = self.log_entries[-MAX_LOG_LINES:]
        self._refresh_log_view()

    def _set_profile_line(self, target: str):
        applied = self.oracle.get_profile() or self.current_profile or "-"
        self._last_recommend_target = target
        detail = (
            f"applied={applied} | target={target} | override={self.manual_override}"
            if self.manual_override
            else f"applied={applied} | target={target}"
        )
        self.profile_var.set(detail)
        self._update_recommendation_line()
        self._update_backend_info()
        self._update_session_outcomes()
        self._update_dashboard_badges()

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
            "System: "
            f"CPU {cpu:4.1f}% | RAM {ram:4.1f}% | "
            f"Battery {'ON' if battery >= 0.5 else 'OFF'} | "
            f"Saver {'ON' if saver >= 0.5 else 'OFF'}\n"
            "Throughput: "
            f"Disk R {disk_r:5.2f} MB/s | Disk W {disk_w:5.2f} MB/s | "
            f"Net D {net_d:6.1f} KB/s | Net U {net_u:6.1f} KB/s"
        )

    def _set_status(self, text: str):
        self.status_var.set(text)
        self._update_dashboard_badges()

    @staticmethod
    def _format_stability_display(stable_count: int, window: int) -> str:
        safe_window = max(1, int(window))
        safe_count = max(0, int(stable_count))
        capped = min(safe_count, safe_window)
        if safe_count > safe_window:
            return f"{capped}/{safe_window} (streak {safe_count})"
        return f"{capped}/{safe_window}"

    def _update_live_meters(self, confidence: float, stable_count: int, window: int):
        conf_pct = max(0.0, min(100.0, confidence * 100.0))
        self.confidence_var.set(conf_pct)
        self.confidence_text_var.set(f"{conf_pct:.1f}%")

        safe_window = max(1, int(window))
        safe_count = max(0, int(stable_count))
        capped = min(safe_count, safe_window)
        stability_pct = 100.0 * capped / safe_window
        stability_pct = max(0.0, min(100.0, stability_pct))
        self.stability_var.set(stability_pct)
        self.stability_text_var.set(self._format_stability_display(safe_count, safe_window))

    def _ensure_label_rows_for_classes(self):
        new_labels: list[str] = []
        for raw_label in self.classes:
            norm = str(raw_label).strip().lower()
            if not norm:
                continue
            if norm not in self.label_map_vars:
                self.label_map_vars[norm] = tk.StringVar(value="balanced")
                self.label_map_vars[norm].trace_add("write", self._on_form_edited)
                self.auto_imported_labels.add(norm)
                new_labels.append(norm)

        if new_labels:
            self._render_label_mapping_rows()
            self._update_mapping_status()
            joined = ", ".join(sorted(new_labels))
            self.log(
                "new model labels defaulted to balanced; review mappings: "
                f"{joined}",
                level="warn",
            )

    def start(self):
        if self.running:
            return
        try:
            self.model, self.classes = load_model_and_classes()
            self.model.to(self.device)
            self._update_model_info()
        except Exception as e:
            self._set_status("error")
            self.log(f"start failed: {e}", level="error")
            return

        self._ensure_label_rows_for_classes()
        self._reset_session_outcomes()

        self.running = True
        self.tick_count = 0
        self.current_profile = None
        self.last_candidate = None
        self.stable_count = 0
        self.last_switch_ts = 0.0
        self.manual_override = None
        self.prediction_smoother.reset()
        self._set_status("running")
        self._set_gate_status("warming up model signal")
        self._set_last_action("session started; waiting for first switch")
        self._set_live_summary(target_profile="-", confident=False)
        self.log("demo started")

        delay_ms = max(100, int(self.runtime_config.switching.poll_interval_sec * 1000))
        self.root.after(delay_ms, self._tick)

    def stop(self):
        if not self.running:
            return
        self.running = False
        self._set_status("stopped")
        self._set_gate_status("stopped")
        self._set_last_action("session stopped")
        self._set_live_summary(target_profile=str(self.current_profile or "-"), confident=False)
        self.log("demo stopped")

    def manual_set(self, profile: Profile):
        # Toggle-off behavior: clicking the active profile resumes AUTO mode.
        if self.manual_override == profile:
            self.clear_manual_override()
            return

        self.manual_override = profile
        result = self.oracle.set_profile(profile, dry_run=self.dry_run_var.get())
        if result.ok:
            self.current_profile = result.applied_profile or profile
            self.last_switch_ts = time.time()
            self._manual_switch_count += 1
            action_text = (
                f"manual override -> {self.current_profile} | {self._describe_apply_result(result)}"
            )
            if result.message:
                action_text += f" | {result.message}"
            self._set_last_action(action_text)
            self.log(f"manual override ON -> {profile}", level="switch")
        else:
            self._failed_switch_count += 1
            action_text = f"manual override failed -> {profile}"
            if result.message:
                action_text += f" | {result.message}"
            self._set_last_action(action_text)
            self.log(
                f"manual override failed -> {profile}: {result.message}",
                level="warn",
            )
        self.last_candidate = None
        self.stable_count = 0
        self._set_profile_line(target=profile)
        self._set_gate_status("manual override active")
        self._set_live_summary(target_profile=profile, confident=True)

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
        self._set_gate_status("returned to automatic control")
        self._set_last_action(f"manual override cleared (was {prev})")
        self._set_live_summary(target_profile="-", confident=False)

    def _tick(self):
        if not self.running or self.model is None:
            return

        try:
            switching = self.runtime_config.switching

            x = get_telemetry()
            x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                logits = self.model(x_t)
                probs = F.softmax(logits, dim=1)

            raw_summary = summarize_probabilities(probs[0], self.classes)
            decision_probs = self.prediction_smoother.update(probs[0])
            decision_summary = summarize_probabilities(decision_probs, self.classes)

            pred_label = decision_summary.label
            confidence = decision_summary.confidence
            runner_up = decision_summary.runner_up_confidence
            margin = decision_summary.margin
            confident = (
                confidence >= switching.confidence_min
                and margin >= switching.confidence_margin
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
                else profile_for_label(
                    pred_label,
                    label_map=self.runtime_config.label_to_profile,
                )
            )
            now_ts = time.time()
            gate = evaluate_switch_gate(
                current_profile=self.current_profile,
                target_profile=target_profile,
                confident=confident,
                stable_count=self.stable_count,
                base_window=switching.stability_window,
                downshift_extra_window=switching.downshift_extra_window,
                now_ts=now_ts,
                last_switch_ts=self.last_switch_ts,
                min_switch_interval_sec=switching.min_switch_interval_sec,
                downshift_hold_sec=switching.downshift_hold_sec,
            )
            if self.manual_override is not None:
                self._set_gate_status("manual override active")
            else:
                gate_text = describe_gate(gate)
                self._set_gate_status(gate_text)
                self._record_gate_observation(gate.reason)
            switched = False

            if self.manual_override is None and gate.allow:
                result = self.oracle.set_profile(
                    target_profile,
                    dry_run=self.dry_run_var.get(),
                )
                if result.ok:
                    self.current_profile = result.applied_profile or target_profile
                    self._auto_switch_count += 1
                    action_text = (
                        f"auto switch -> {self.current_profile} ({gate.direction}, conf {confidence:.2f})"
                        f" | {self._describe_apply_result(result)}"
                    )
                    if result.message:
                        action_text += f" | {result.message}"
                    self._set_last_action(action_text)
                    self.log(
                        f"switch label={pred_label} -> profile={self.current_profile} "
                        f"(confidence={confidence:.2f}, gate={gate.direction})",
                        level="switch",
                    )
                else:
                    self._failed_switch_count += 1
                    action_text = f"auto switch failed -> {target_profile} ({gate.direction})"
                    if result.message:
                        action_text += f" | {result.message}"
                    self._set_last_action(action_text)
                    self.log(
                        f"switch failed label={pred_label} -> profile={target_profile}: "
                        f"{result.message}",
                        level="warn",
                    )
                switched = True
                self.last_switch_ts = now_ts

            self.metrics_var.set(self._format_metrics(x))
            stability_text = self._format_stability_display(
                self.stable_count,
                switching.stability_window,
            )
            self.prediction_var.set(
                f"Raw: {raw_summary.label} ({raw_summary.confidence:.2f})\n"
                f"Decision: {pred_label} ({confidence:.2f}) | Margin {margin:.2f} | "
                f"Reliable {confident} | Stability {stability_text}"
            )
            self._update_live_meters(
                confidence,
                self.stable_count,
                gate.required_stability,
            )
            self._set_profile_line(target=target_profile)
            self._set_live_summary(
                pred_label=pred_label,
                confidence=confidence,
                margin=margin,
                target_profile=target_profile,
                confident=confident,
                stable_count=self.stable_count,
                window=gate.required_stability,
            )

            self.tick_count += 1
            if self.tick_count % 8 == 0 and not switched:
                self.log(
                    f"tick label={pred_label} conf={confidence:.2f} "
                    f"stable={self._format_stability_display(self.stable_count, gate.required_stability)} "
                    f"gate={describe_gate(gate)} mode={'MANUAL' if self.manual_override else 'AUTO'}"
                )

        except Exception as e:
            self._set_status("error")
            self.log(f"runtime error: {e}", level="error")
            self.running = False
            return

        delay_ms = max(100, int(self.runtime_config.switching.poll_interval_sec * 1000))
        self.root.after(delay_ms, self._tick)


def main():
    root = tk.Tk()
    app = DemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
