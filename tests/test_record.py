import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import record
from monitor import FEATURE_NAMES
from training.models import TrainingSessionManifest
from training.session_store import default_session_manifest_path, read_training_session_manifest


class RecordTests(unittest.TestCase):
    def test_default_output_is_one_csv_per_label_and_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            features = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
            with mock.patch.object(record, "get_telemetry", return_value=features):
                path = record.record(
                    "YouTube 1080p",
                    duration=0.001,
                    interval=0.001,
                    session_id="Session One",
                    data_dir=temp_dir,
                )

            expected = Path(temp_dir) / "youtube_1080p" / "session_one.csv"
            self.assertEqual(path, expected)
            self.assertTrue(expected.exists())
            self.assertGreaterEqual(
                expected.read_text(encoding="utf-8").count("\n"),
                2,
            )

    def test_writes_session_manifest_sidecar_when_provided(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            features = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
            manifest = TrainingSessionManifest(
                session_id="session_one",
                label="browsing",
                proposed_label="browsing",
                approved_label="browsing",
                app_rule_id="browser",
            )
            with mock.patch.object(record, "get_telemetry", return_value=features):
                path = record.record(
                    "browsing",
                    duration=0.001,
                    interval=0.001,
                    session_id="Session One",
                    data_dir=temp_dir,
                    session_manifest=manifest.to_dict(),
                )

            manifest_path = default_session_manifest_path(path)
            self.assertTrue(manifest_path.exists())
            loaded = read_training_session_manifest(manifest_path)
            self.assertEqual(loaded.approved_label, "browsing")
            self.assertEqual(loaded.app_rule_id, "browser")


if __name__ == "__main__":
    unittest.main()
