import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from config import default_config, write_config
from oracle_client import (
    CommandOracle,
    DemoOracle,
    OracleClient,
    PowerCfgProfileTuning,
    build_powercfg_tuning_commands,
    load_powercfg_profile_tuning,
    parse_powercfg_list,
    powercfg_scheme_report,
    powercfg_target_for_profile,
    profile_for_power_scheme,
    resolve_powercfg_target_token,
)


class _VerifyingBackend:
    def __init__(self, observed_profile: str):
        self._observed_profile = observed_profile

    def set_profile(self, profile: str):
        return {"ok": True, "applied_profile": profile, "message": "backend reported success"}

    def get_profile(self):
        return self._observed_profile


class OracleClientTests(unittest.TestCase):
    def test_parse_powercfg_list_extracts_schemes(self):
        output = """
Existing Power Schemes (* Active)
-----------------------------------
Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced) *
Power Scheme GUID: a1841308-3541-4fab-bc81-f71556f20b4a  (Power saver)
Power Scheme GUID: 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  (High performance)
"""
        schemes = parse_powercfg_list(output)
        self.assertEqual(len(schemes), 3)
        self.assertTrue(schemes[0].active)
        self.assertEqual(schemes[2].name, "High performance")

    def test_profile_for_power_scheme_prefers_known_guid(self):
        scheme = parse_powercfg_list(
            "Power Scheme GUID: e9a42b02-d5df-448d-aa00-03f14749eb61  (Ultimate Performance) *"
        )[0]
        self.assertEqual(profile_for_power_scheme(scheme), "performance")

    def test_powercfg_target_prefers_matching_scheme_name(self):
        schemes = parse_powercfg_list(
            """
Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced) *
Power Scheme GUID: 11111111-1111-1111-1111-111111111111  (Ultimate Performance)
Power Scheme GUID: 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  (High performance)
"""
        )
        token = powercfg_target_for_profile("performance", schemes)
        self.assertEqual(token, "11111111-1111-1111-1111-111111111111")

    def test_powercfg_target_honors_override(self):
        schemes = []
        token = powercfg_target_for_profile(
            "silent",
            schemes,
            overrides={"silent": "custom-guid"},
        )
        self.assertEqual(token, "custom-guid")

    def test_resolve_powercfg_target_token_matches_scheme_name(self):
        schemes = parse_powercfg_list(
            "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced) *"
        )
        token = resolve_powercfg_target_token("Balanced", schemes)
        self.assertEqual(token, "381b4222-f694-41f0-9685-ff5bb260df2e")

    def test_powercfg_scheme_report_includes_mapping(self):
        schemes = parse_powercfg_list(
            "Power Scheme GUID: a1841308-3541-4fab-bc81-f71556f20b4a  (Power saver) *"
        )
        report = powercfg_scheme_report(schemes)
        self.assertEqual(report[0]["mapped_profile"], "silent")

    def test_demo_oracle_diagnostics_are_available(self):
        diag = DemoOracle().diagnostics()
        self.assertEqual(diag.backend_name, "demo_stub")
        self.assertEqual(diag.selected_profile, "balanced")

    def test_load_powercfg_profile_tuning_from_env(self):
        tuning = load_powercfg_profile_tuning(
            {
                "PERFANALYZE_POWERCFG_SILENT_MAX_CPU": "55",
                "PERFANALYZE_POWERCFG_SILENT_BOOST_MODE": "disabled",
                "PERFANALYZE_POWERCFG_PERFORMANCE_MIN_CPU": "30",
            }
        )
        self.assertEqual(tuning["silent"].max_cpu_percent, 55)
        self.assertEqual(tuning["silent"].boost_mode, 0)
        self.assertEqual(tuning["performance"].min_cpu_percent, 30)

    def test_build_powercfg_tuning_commands_includes_ac_and_dc(self):
        tuning = PowerCfgProfileTuning(min_cpu_percent=20, max_cpu_percent=85, boost_mode=2)
        commands = build_powercfg_tuning_commands(
            "381b4222-f694-41f0-9685-ff5bb260df2e",
            tuning,
        )
        self.assertEqual(len(commands), 6)
        self.assertEqual(commands[0][0], "/setacvalueindex")
        self.assertEqual(commands[1][0], "/setdcvalueindex")
        self.assertEqual(commands[-1][-1], "2")

    def test_command_oracle_can_drive_mock_script(self):
        script = Path(__file__).resolve().parents[1] / "oracle_command_mock.py"
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "oracle-state.json"
            oracle = CommandOracle(
                [sys.executable, str(script), "--state", str(state)],
                timeout_sec=3.0,
            )
            result = oracle.set_profile("performance")
            self.assertEqual(result["applied_profile"], "performance")
            self.assertEqual(oracle.get_profile(), "performance")
            diag = oracle.diagnostics()
            self.assertEqual(diag.backend_name, "oracle_command_mock")
            self.assertEqual(diag.selected_profile, "performance")

    def test_command_oracle_can_drive_hotkey_bridge_in_dry_run_mode(self):
        script = Path(__file__).resolve().parents[1] / "oracle_command_hotkey.py"
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "oracle-hotkey-state.json"
            config_path = Path(tmp) / "perfalyze_config.json"
            cfg = default_config()
            cfg.profile_hotkeys["performance"] = "ctrl+shift+f11"
            write_config(cfg, config_path)
            oracle = CommandOracle(
                [
                    sys.executable,
                    str(script),
                    "--config",
                    str(config_path),
                    "--state",
                    str(state),
                ],
                timeout_sec=3.0,
            )
            with mock.patch.dict(os.environ, {"PERFANALYZE_HOTKEY_BACKEND_DRY_RUN": "1"}, clear=False):
                result = oracle.set_profile("performance")
                self.assertEqual(result["applied_profile"], "performance")
                self.assertEqual(oracle.get_profile(), "performance")
                diag = oracle.diagnostics()
                self.assertEqual(diag.backend_name, "oracle_command_hotkey")
                self.assertEqual(diag.selected_profile, "performance")
                self.assertEqual(diag.details.get("dry_run"), True)

    def test_oracle_client_verifies_matching_profile(self):
        backend = _VerifyingBackend("performance")
        with mock.patch.object(OracleClient, "_load_backend", return_value=(backend, "test_backend")):
            client = OracleClient()
            result = client.set_profile("performance")
        self.assertTrue(result.ok)
        self.assertEqual(result.observed_profile, "performance")
        self.assertTrue(result.verified)

    def test_oracle_client_fails_on_verification_mismatch(self):
        backend = _VerifyingBackend("balanced")
        with mock.patch.object(OracleClient, "_load_backend", return_value=(backend, "test_backend")):
            client = OracleClient()
            result = client.set_profile("performance")
        self.assertFalse(result.ok)
        self.assertEqual(result.observed_profile, "balanced")
        self.assertFalse(result.verified)
        self.assertIn("verification mismatch", result.message)


if __name__ == "__main__":
    unittest.main()
