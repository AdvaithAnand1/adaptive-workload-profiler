from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import ObservationWindow, OutcomeComparison, PolicyOutcome


class ObservationStore:
    def __init__(self, path: str | Path = "adaptive_observations.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observation_windows (
                    window_id TEXT PRIMARY KEY,
                    predicted_label TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_outcomes (
                    window_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS outcome_comparisons (
                    comparison_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload TEXT NOT NULL
                )
                """
            )

    def write_window(self, window: ObservationWindow):
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO observation_windows VALUES (?, ?, ?, ?)",
                (window.window_id, window.predicted_label, window.plan_id, json.dumps(window.to_dict())),
            )

    def write_outcome(self, outcome: PolicyOutcome):
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO policy_outcomes VALUES (?, ?)",
                (outcome.window.window_id, json.dumps(outcome.to_dict())),
            )

    def write_comparison(self, comparison: OutcomeComparison):
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO outcome_comparisons(payload) VALUES (?)",
                (json.dumps(comparison.to_dict()),),
            )

