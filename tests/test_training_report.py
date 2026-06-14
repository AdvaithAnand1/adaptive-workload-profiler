import unittest

from training_report import (
    compute_accuracy_by_label,
    compute_class_counts,
    compute_class_weights,
)


class TrainingReportTests(unittest.TestCase):
    def test_compute_class_counts(self):
        counts = compute_class_counts([0, 1, 1, 2, 2, 2], 3)
        self.assertEqual(counts, [1, 2, 3])

    def test_compute_class_weights_gives_more_weight_to_rare_classes(self):
        weights = compute_class_weights([10, 50, 100])
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

    def test_compute_accuracy_by_label(self):
        stats = compute_accuracy_by_label(
            [0, 0, 1, 1, 2],
            [0, 1, 1, 1, 0],
            ["idle", "light", "heavy"],
        )
        self.assertAlmostEqual(float(stats["idle"]["accuracy"]), 0.5)
        self.assertAlmostEqual(float(stats["light"]["accuracy"]), 1.0)
        self.assertAlmostEqual(float(stats["heavy"]["accuracy"]), 0.0)


if __name__ == "__main__":
    unittest.main()
