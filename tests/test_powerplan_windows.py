import unittest
from unittest import mock

from powerplans.windows import PowerCfgDiscovery


class _Completed:
    def __init__(self, stdout: str):
        self.returncode = 0
        self.stdout = stdout
        self.stderr = ""


class PowerCfgDiscoveryTests(unittest.TestCase):
    @mock.patch("powerplans.windows.subprocess.run")
    def test_discovery_uses_read_only_powercfg_commands(self, run):
        run.side_effect = [
            _Completed(
                "Power Scheme GUID: "
                "381b4222-f694-41f0-9685-ff5bb260df2e (Balanced) *"
            ),
            _Completed(
                """
Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e (Balanced)
  Subgroup GUID: 5fb4938d-1ee8-4b0f-9a3c-5036b0ab995c (Graphics settings)
    GUID Alias: SUB_GRAPHICS
    Power Setting GUID: dd848b2a-8a5d-4451-9ae2-39cd41658f6c (GPU preference policy)
      GUID Alias: GPUPREFERENCEPOLICY
    Current AC Power Setting Index: 0x00000000
    Current DC Power Setting Index: 0x00000001
"""
            ),
        ]
        discovery = PowerCfgDiscovery()

        schemes = discovery.list_schemes()
        snapshot = discovery.discover_settings()

        self.assertEqual(schemes[0].name, "Balanced")
        self.assertEqual(snapshot.by_domain("graphics")[0].dc_value, 1)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0], ["powercfg", "/list"])
        self.assertEqual(commands[1], ["powercfg", "/qh", "SCHEME_CURRENT"])
        for command in commands:
            self.assertNotIn("/setactive", command)
            self.assertNotIn("/setacvalueindex", command)
            self.assertNotIn("/setdcvalueindex", command)


if __name__ == "__main__":
    unittest.main()
