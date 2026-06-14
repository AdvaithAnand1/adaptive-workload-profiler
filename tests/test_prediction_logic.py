import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on local runtime
    torch = None

if torch is not None:
    from prediction_logic import ProbabilitySmoother, summarize_probabilities


@unittest.skipIf(torch is None, "torch is not installed in this runtime")
class PredictionLogicTests(unittest.TestCase):
    def test_summarize_probabilities_returns_top_class_and_margin(self):
        summary = summarize_probabilities(
            torch.tensor([0.15, 0.70, 0.15]),
            ["idle", "light", "heavy"],
        )
        self.assertEqual(summary.label, "light")
        self.assertAlmostEqual(summary.confidence, 0.70, places=5)
        self.assertAlmostEqual(summary.margin, 0.55, places=5)

    def test_probability_smoother_reduces_single_step_jump(self):
        smoother = ProbabilitySmoother(0.5)
        first = smoother.update(torch.tensor([0.90, 0.10]))
        second = smoother.update(torch.tensor([0.10, 0.90]))
        self.assertAlmostEqual(float(first[0]), 0.90, places=5)
        self.assertAlmostEqual(float(second[0]), 0.50, places=5)
        self.assertAlmostEqual(float(second[1]), 0.50, places=5)


if __name__ == "__main__":
    unittest.main()
