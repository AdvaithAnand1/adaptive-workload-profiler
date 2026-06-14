import tempfile
import unittest
from pathlib import Path

from config import (
    VALID_PROFILES,
    build_config,
    default_config,
    load_config,
    write_config,
)


class ConfigTests(unittest.TestCase):
    def test_default_config_has_valid_profiles(self):
        cfg = default_config()
        valid = set(VALID_PROFILES)
        for profile in cfg.label_to_profile.values():
            self.assertIn(profile, valid)

    def test_build_config_rejects_invalid_profile(self):
        raw = {
            "label_to_profile": {"idle": "invalid-profile"},
        }
        with self.assertRaises(RuntimeError):
            build_config(raw)

    def test_write_and_load_roundtrip(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "perfalyze_config.json"
            write_config(cfg, path)
            loaded = load_config(path).config
            self.assertEqual(loaded.switching.stability_window, cfg.switching.stability_window)
            self.assertEqual(
                loaded.switching.probability_ema_alpha,
                cfg.switching.probability_ema_alpha,
            )
            self.assertEqual(
                loaded.switching.downshift_extra_window,
                cfg.switching.downshift_extra_window,
            )
            self.assertEqual(
                loaded.switching.downshift_hold_sec,
                cfg.switching.downshift_hold_sec,
            )
            self.assertEqual(loaded.label_to_profile, cfg.label_to_profile)
            self.assertEqual(loaded.profile_hotkeys, cfg.profile_hotkeys)

    def test_build_config_allows_blank_hotkey_strings(self):
        cfg = build_config(
            {
                "profile_hotkeys": {
                    "silent": "",
                    "balanced": "ctrl+shift+alt+f17",
                    "performance": "",
                }
            }
        )
        self.assertEqual(cfg.profile_hotkeys["silent"], "")
        self.assertEqual(cfg.profile_hotkeys["performance"], "")

    def test_write_config_creates_parent_directories(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "configs" / "perfalyze_config.json"
            write_config(cfg, path)
            self.assertTrue(path.exists())
            loaded = load_config(path).config
            self.assertEqual(loaded.label_to_profile, cfg.label_to_profile)

    def test_build_config_accepts_probability_ema_alpha(self):
        cfg = build_config(
            {
                "switching": {
                    "probability_ema_alpha": 0.7,
                    "downshift_extra_window": 3,
                    "downshift_hold_sec": 9.5,
                }
            }
        )
        self.assertAlmostEqual(cfg.switching.probability_ema_alpha, 0.7)
        self.assertEqual(cfg.switching.downshift_extra_window, 3)
        self.assertAlmostEqual(cfg.switching.downshift_hold_sec, 9.5)


if __name__ == "__main__":
    unittest.main()
