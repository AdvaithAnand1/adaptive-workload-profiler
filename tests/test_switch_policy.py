import unittest

from switch_policy import describe_gate, evaluate_switch_gate, required_stability_window


class SwitchPolicyTests(unittest.TestCase):
    def test_downshift_requires_extra_stability(self):
        required = required_stability_window(
            base_window=5,
            current_profile="performance",
            target_profile="balanced",
            downshift_extra_window=2,
        )
        self.assertEqual(required, 7)

    def test_downshift_hold_blocks_fast_drop(self):
        result = evaluate_switch_gate(
            current_profile="performance",
            target_profile="balanced",
            confident=True,
            stable_count=8,
            base_window=5,
            downshift_extra_window=2,
            now_ts=105.0,
            last_switch_ts=100.0,
            min_switch_interval_sec=4.0,
            downshift_hold_sec=10.0,
        )
        self.assertFalse(result.allow)
        self.assertEqual(result.reason, "downshift-hold")
        self.assertIn("holding higher profile", describe_gate(result))

    def test_upshift_can_be_ready_once_stable(self):
        result = evaluate_switch_gate(
            current_profile="balanced",
            target_profile="performance",
            confident=True,
            stable_count=5,
            base_window=5,
            downshift_extra_window=2,
            now_ts=120.0,
            last_switch_ts=100.0,
            min_switch_interval_sec=8.0,
            downshift_hold_sec=10.0,
        )
        self.assertTrue(result.allow)
        self.assertEqual(result.reason, "ready")


if __name__ == "__main__":
    unittest.main()
