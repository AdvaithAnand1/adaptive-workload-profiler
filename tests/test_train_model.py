import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from monitor import FEATURE_NAMES
from train_model import load_session_dataframe, split_by_session


class TrainModelTests(unittest.TestCase):
    def test_loads_csvs_recursively(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for label in ("idle", "youtube"):
                path = root / label / "session.csv"
                path.parent.mkdir(parents=True)
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        ["timestamp", "session_id", "label", *FEATURE_NAMES]
                    )
                    writer.writerow(
                        ["now", f"{label}_1", label, *([0.0] * len(FEATURE_NAMES))]
                    )

            frame, files = load_session_dataframe(root)
            self.assertEqual(len(files), 2)
            self.assertEqual(len(frame), 2)

    def test_split_keeps_sessions_out_of_both_sets(self):
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        sessions = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
        train_idx, test_idx = split_by_session(
            labels,
            sessions,
            test_size=0.5,
            seed=42,
        )
        self.assertTrue(set(sessions[train_idx]).isdisjoint(sessions[test_idx]))
        self.assertEqual(set(labels[train_idx]), {0, 1})
        self.assertEqual(set(labels[test_idx]), {0, 1})


if __name__ == "__main__":
    unittest.main()
