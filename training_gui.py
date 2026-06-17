from __future__ import annotations

import threading
import time
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import ttk
from uuid import uuid4

from record import record
from training.app_registry import (
    detect_foreground_process,
    load_training_app_catalog,
    match_training_app,
)
from training.models import TrainingSessionManifest
from training.session_store import iter_training_session_manifests
from ui_theme import APP_BG, RoundedButton, RoundedCard, apply_google_theme


class TrainingGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        apply_google_theme(self.root)
        self.root.title("PerfAnalyze Training")
        self.root.geometry("1100x760")
        self.root.minsize(920, 660)

        self.catalog = load_training_app_catalog()
        self.data_dir = Path("data")
        self.current_context = None
        self.current_rule = None
        self.record_thread: threading.Thread | None = None
        self.recording = False

        self.detected_var = tk.StringVar(value="Detecting foreground app...")
        self.proposed_var = tk.StringVar(value="proposed=-")
        self.approval_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="idle")
        self.progress_var = tk.StringVar(value="No session running")
        self.sessions_var = tk.StringVar(value="No sessions yet")
        self.interval_var = tk.StringVar(value="0.25")
        self.duration_var = tk.StringVar(value="180")
        self.session_label_var = tk.StringVar(value="")
        self.approved_label_var = tk.StringVar(value="browsing")

        self._build_ui()
        self._refresh_sessions()
        self._poll_foreground()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        header = RoundedCard(
            outer,
            title="Training-only App Policies",
            subtitle="Foreground process context is used only to propose labels for training data.",
        )
        header.pack(fill=tk.X)
        ttk.Label(header.body, textvariable=self.detected_var, justify=tk.LEFT).pack(anchor="w")
        ttk.Label(header.body, textvariable=self.proposed_var, justify=tk.LEFT).pack(anchor="w", pady=(4, 0))
        ttk.Label(header.body, textvariable=self.approval_var, justify=tk.LEFT).pack(anchor="w", pady=(4, 0))

        control_card = RoundedCard(outer, title="Collection Controls")
        control_card.pack(fill=tk.X, pady=(14, 0))
        control_row = ttk.Frame(control_card.body, style="Card.TFrame")
        control_row.pack(fill=tk.X)

        ttk.Label(control_row, text="Approved label", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Combobox(
            control_row,
            textvariable=self.approved_label_var,
            values=["browsing", "office", "media", "gaming", "rendering"],
            width=16,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(8, 16))
        ttk.Label(control_row, text="Interval (s)", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Entry(control_row, textvariable=self.interval_var, width=8).pack(side=tk.LEFT, padx=(8, 16))
        ttk.Label(control_row, text="Duration (s)", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Entry(control_row, textvariable=self.duration_var, width=8).pack(side=tk.LEFT, padx=(8, 16))
        ttk.Label(control_row, text="Session id", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Entry(control_row, textvariable=self.session_label_var, width=24).pack(side=tk.LEFT, padx=(8, 16))

        self.start_button = RoundedButton(
            control_card.body,
            text="Approve & Record",
            command=self._start_recording,
            variant="primary",
            canvas_bg=APP_BG,
        )
        self.start_button.pack(side=tk.LEFT, pady=(14, 0))
        RoundedButton(
            control_card.body,
            text="Refresh",
            command=self._refresh_sessions,
            canvas_bg=APP_BG,
        ).pack(side=tk.LEFT, padx=(10, 0), pady=(14, 0))
        ttk.Label(control_card.body, textvariable=self.status_var, style="Muted.Card.TLabel").pack(
            side=tk.RIGHT,
            pady=(14, 0),
        )

        live_card = RoundedCard(
            outer,
            title="Current Session",
            subtitle="Status updates while a telemetry session is being recorded.",
        )
        live_card.pack(fill=tk.X, pady=(14, 0))
        ttk.Label(live_card.body, textvariable=self.progress_var, justify=tk.LEFT).pack(anchor="w")

        sessions_card = RoundedCard(
            outer,
            title="Collected Sessions",
            subtitle="Sidecar manifests list the proposed and approved labels for each recording.",
        )
        sessions_card.pack(fill=tk.BOTH, expand=True, pady=(14, 0))
        ttk.Label(
            sessions_card.body,
            textvariable=self.sessions_var,
            style="Muted.Card.TLabel",
        ).pack(anchor="w", pady=(0, 8))
        self.sessions_box = tk.Text(
            sessions_card.body,
            height=16,
            wrap="word",
            background="#f1f3f4",
            foreground="#202124",
            relief=tk.FLAT,
            borderwidth=0,
            padx=12,
            pady=10,
            font=("Cascadia Mono", 9),
        )
        self.sessions_box.pack(fill=tk.BOTH, expand=True)
        self.sessions_box.configure(state=tk.DISABLED)

    def _poll_foreground(self):
        if not self.recording:
            self.current_context = detect_foreground_process()
            self.current_rule = match_training_app(self.catalog, self.current_context)
            self._refresh_detection_view()
        self.root.after(1000, self._poll_foreground)

    def _refresh_detection_view(self):
        if self.current_context is None:
            self.detected_var.set("No foreground app detected.")
            self.proposed_var.set("proposed=unknown")
            self.approval_var.set("Unknown apps are ignored until they match an allowlisted rule.")
            self.start_button.configure(state=tk.DISABLED)
            return

        ctx = self.current_context
        rule = self.current_rule
        detected = (
            f"pid={ctx.pid} | app={ctx.executable_name} | parent={ctx.parent_name or '-'} | "
            f"path={ctx.normalized_executable_path}"
        )
        self.detected_var.set(detected)
        if rule is None:
            self.proposed_var.set("proposed=unknown")
            self.approval_var.set("Unknown apps are ignored until they match an editable training rule.")
            self.start_button.configure(state=tk.DISABLED)
            return

        self.proposed_var.set(f"proposed={rule.label} | rule={rule.display_name}")
        self.approval_var.set(
            f"Approve {rule.label} to record telemetry, or choose a different approved label first."
        )
        self.start_button.configure(state=tk.NORMAL if not self.recording else tk.DISABLED)
        if not self.recording:
            self.approved_label_var.set(rule.label)

    def _parse_duration(self) -> float | None:
        raw = self.duration_var.get().strip()
        if not raw:
            return None
        value = float(raw)
        if value <= 0:
            raise ValueError("duration must be positive")
        return value

    def _start_recording(self):
        if self.recording:
            return
        if self.current_context is None or self.current_rule is None:
            self.status_var.set("No known app selected for collection.")
            return

        approved_label = self.approved_label_var.get().strip().lower()
        if approved_label not in {"browsing", "office", "media", "gaming", "rendering"}:
            self.status_var.set("Choose an approved label before recording.")
            return

        if approved_label != self.current_rule.label:
            self.status_var.set(
                f"Approval mismatch: proposed={self.current_rule.label} approved={approved_label}."
            )
            return

        try:
            interval = float(self.interval_var.get().strip())
            duration = self._parse_duration()
        except ValueError as exc:
            self.status_var.set(str(exc))
            return

        session_id = self.session_label_var.get().strip() or f"train_{uuid4().hex[:8]}"
        manifest = TrainingSessionManifest(
            session_id=session_id,
            label=approved_label,
            proposed_label=self.current_rule.label,
            approved_label=approved_label,
            app_rule_id=self.current_rule.rule_id,
            data_path=str(self.data_dir),
            started_at=datetime.now(timezone.utc).isoformat(),
            context=self.current_context,
        )

        self.recording = True
        self.status_var.set("Recording session...")
        self.progress_var.set(
            f"Recording {approved_label} from {self.current_rule.display_name} ({session_id})"
        )
        self.start_button.configure(state=tk.DISABLED)
        self.record_thread = threading.Thread(
            target=self._run_recording,
            args=(approved_label, interval, duration, session_id, manifest),
            daemon=True,
        )
        self.record_thread.start()

    def _run_recording(
        self,
        label: str,
        interval: float,
        duration: float | None,
        session_id: str,
        manifest: TrainingSessionManifest,
    ):
        try:
            record(
                label=label,
                interval=interval,
                duration=duration,
                session_id=session_id,
                data_dir=self.data_dir,
                session_manifest=manifest.to_dict(),
            )
            status_message = "Recording finished."
        except Exception as exc:
            status_message = f"Recording failed: {exc}"
        finally:
            self.recording = False
            progress_message = "Session complete" if "finished" in status_message.lower() else status_message
            self.root.after(0, lambda message=status_message: self.status_var.set(message))
            self.root.after(0, self._refresh_sessions)
            self.root.after(0, self._refresh_detection_view)
            self.root.after(
                0,
                lambda message=progress_message: self.progress_var.set(message),
            )

    def _refresh_sessions(self):
        manifests = list(iter_training_session_manifests(self.data_dir))
        if not manifests:
            text = "No sessions yet."
        else:
            lines = []
            for manifest in manifests[-8:]:
                lines.append(
                    f"{manifest.session_id} | approved={manifest.approved_label} | "
                    f"proposed={manifest.proposed_label} | rule={manifest.app_rule_id or '-'} | "
                    f"ended={manifest.ended_at or 'running'}"
                )
            text = "\n".join(lines)

        self.sessions_box.configure(state=tk.NORMAL)
        self.sessions_box.delete("1.0", tk.END)
        self.sessions_box.insert(tk.END, text)
        self.sessions_box.configure(state=tk.DISABLED)
        self.sessions_var.set(f"{len(manifests)} session manifests found")


def main():
    root = tk.Tk()
    TrainingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
