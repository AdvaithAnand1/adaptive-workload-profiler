import tempfile
import unittest
from pathlib import Path

from training.models import TrainingSessionManifest
from training.session_store import (
    default_session_manifest_path,
    iter_training_session_manifests,
    read_training_session_manifest,
    write_training_session_manifest,
)


class TrainingSessionStoreTests(unittest.TestCase):
    def test_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "session.manifest.json"
            manifest = TrainingSessionManifest(
                session_id="s1",
                label="browsing",
                proposed_label="browsing",
                approved_label="browsing",
                app_rule_id="browser",
            )
            write_training_session_manifest(manifest, path)
            loaded = read_training_session_manifest(path)
            self.assertEqual(loaded.session_id, "s1")
            self.assertEqual(loaded.approved_label, "browsing")

    def test_iter_manifests_finds_sidecars(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = TrainingSessionManifest(
                session_id="s2",
                label="office",
                proposed_label="office",
                approved_label="office",
                app_rule_id="office",
            )
            write_training_session_manifest(manifest, root / "data" / "office" / "s2.manifest.json")
            manifests = list(iter_training_session_manifests(root))
            self.assertEqual(len(manifests), 1)
            self.assertEqual(manifests[0].label, "office")


if __name__ == "__main__":
    unittest.main()
