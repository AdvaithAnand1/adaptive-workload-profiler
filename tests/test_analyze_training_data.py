import unittest

from analyze_training_data import summarize_training_rows


class AnalyzeTrainingDataTests(unittest.TestCase):
    def test_detects_sparse_labels_and_single_session_labels(self):
        rows = []
        for idx in range(220):
            rows.append(
                {
                    "timestamp": f"t{idx}",
                    "session_id": "idle_a" if idx < 110 else "idle_b",
                    "label": "idle",
                    "Power Saver [0/1]": "0",
                }
            )
        for idx in range(80):
            rows.append(
                {
                    "timestamp": f"l{idx}",
                    "session_id": "light_a",
                    "label": "light",
                    "Power Saver [0/1]": "1",
                }
            )
        summary = summarize_training_rows(rows, csv_path="training_data.csv")
        self.assertTrue(any("underrepresented labels" in rec for rec in summary["recommendations"]))
        self.assertTrue(any("only one session" in rec for rec in summary["recommendations"]))
        self.assertLess(summary["label_imbalance_ratio"], 0.55)

    def test_flags_power_saver_imbalance(self):
        rows = []
        for idx in range(180):
            rows.append(
                {
                    "timestamp": f"h{idx}",
                    "session_id": "heavy_a" if idx < 90 else "heavy_b",
                    "label": "heavy",
                    "Power Saver [0/1]": "1" if idx < 170 else "0",
                }
            )
        for idx in range(180):
            rows.append(
                {
                    "timestamp": f"i{idx}",
                    "session_id": "idle_a" if idx < 90 else "idle_b",
                    "label": "idle",
                    "Power Saver [0/1]": "0" if idx < 90 else "1",
                }
            )
        for idx in range(180):
            rows.append(
                {
                    "timestamp": f"l{idx}",
                    "session_id": "light_a" if idx < 90 else "light_b",
                    "label": "light",
                    "Power Saver [0/1]": "0" if idx < 90 else "1",
                }
            )
        summary = summarize_training_rows(rows, csv_path="training_data.csv")
        self.assertTrue(any("power-saver on/off coverage" in rec for rec in summary["recommendations"]))
        self.assertIn("heavy", summary["power_saver_balance_by_label"])


if __name__ == "__main__":
    unittest.main()
