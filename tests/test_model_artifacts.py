import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from model import SystemStateNet
from model_artifacts import (
    MODEL_ARTIFACT_SCHEMA,
    MODEL_METADATA_SCHEMA_VERSION,
    load_model_artifacts,
    normalize_features,
)
from monitor import FEATURE_NAMES


class ModelArtifactTests(unittest.TestCase):
    def test_loads_normalized_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.pth"
            classes_path = root / "classes.json"
            metadata_path = root / "model_metadata.json"

            model = SystemStateNet(len(FEATURE_NAMES), 2)
            torch.save(model.state_dict(), model_path)
            classes_path.write_text('["idle", "light"]', encoding="utf-8")
            metadata_path.write_text(
                __import__("json").dumps(
                    {
                        "schema_version": MODEL_METADATA_SCHEMA_VERSION,
                        "artifact_schema": MODEL_ARTIFACT_SCHEMA,
                        "feature_names": FEATURE_NAMES,
                        "feature_mean": [1.0] * len(FEATURE_NAMES),
                        "feature_std": [2.0] * len(FEATURE_NAMES),
                    }
                ),
                encoding="utf-8",
            )

            bundle = load_model_artifacts(
                model_path=model_path,
                classes_path=classes_path,
                metadata_path=metadata_path,
            )
            normalized = normalize_features(np.ones(len(FEATURE_NAMES), dtype=np.float32), bundle)
            self.assertTrue(np.allclose(normalized, np.zeros(len(FEATURE_NAMES))))

    def test_rejects_incompatible_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.pth"
            classes_path = root / "classes.json"
            metadata_path = root / "model_metadata.json"

            model = SystemStateNet(len(FEATURE_NAMES), 2)
            torch.save(model.state_dict(), model_path)
            classes_path.write_text('["idle", "light"]', encoding="utf-8")
            metadata_path.write_text(
                __import__("json").dumps(
                    {
                        "schema_version": 0,
                        "artifact_schema": "old",
                        "feature_names": FEATURE_NAMES,
                        "feature_mean": [1.0] * len(FEATURE_NAMES),
                        "feature_std": [1.0] * len(FEATURE_NAMES),
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                load_model_artifacts(
                    model_path=model_path,
                    classes_path=classes_path,
                    metadata_path=metadata_path,
                )


if __name__ == "__main__":
    unittest.main()
