import unittest

from config import default_config
from probe_tuning import tune_switching_config


class ProbeTuningTests(unittest.TestCase):
    def test_high_churn_tunes_toward_more_stability(self):
        current = default_config().switching
        report = tune_switching_config(
            {
                "rows": 200,
                "switches_per_10min": 6.5,
                "low_confidence_ratio": 0.10,
                "avg_decision_confidence": 0.68,
                "avg_decision_margin": 0.12,
                "dominant_label_ratio": 0.45,
                "gate_reason_counts": {"cooldown": 40},
            },
            current,
        )
        self.assertGreater(
            report.recommended_switching["stability_window"],
            current.stability_window,
        )
        self.assertGreater(
            report.recommended_switching["min_switch_interval_sec"],
            current.min_switch_interval_sec,
        )
        self.assertLess(
            report.recommended_switching["probability_ema_alpha"],
            current.probability_ema_alpha,
        )

    def test_low_confidence_can_relax_thresholds_when_data_is_not_collapsed(self):
        current = default_config().switching
        report = tune_switching_config(
            {
                "rows": 240,
                "switches_per_10min": 1.2,
                "low_confidence_ratio": 0.42,
                "avg_decision_confidence": 0.52,
                "avg_decision_margin": 0.09,
                "avg_raw_confidence": 0.62,
                "dominant_label_ratio": 0.50,
                "heavy_rate_power_saver_gap": 0.10,
                "gate_reason_counts": {"low-confidence": 80, "building-stability": 50},
            },
            current,
        )
        self.assertLess(
            report.recommended_switching["confidence_min"],
            current.confidence_min,
        )
        self.assertGreater(
            report.recommended_switching["probability_ema_alpha"],
            current.probability_ema_alpha,
        )

    def test_dataset_warnings_block_aggressive_relaxation(self):
        current = default_config().switching
        report = tune_switching_config(
            {
                "rows": 120,
                "switches_per_10min": 1.0,
                "low_confidence_ratio": 0.50,
                "avg_decision_confidence": 0.40,
                "avg_decision_margin": 0.03,
                "avg_raw_confidence": 0.44,
                "dominant_label_ratio": 0.90,
                "heavy_rate_power_saver_gap": 0.45,
                "gate_reason_counts": {"low-confidence": 60},
            },
            current,
        )
        self.assertTrue(report.warnings)
        self.assertEqual(
            report.recommended_switching["confidence_min"],
            current.confidence_min,
        )


if __name__ == "__main__":
    unittest.main()
