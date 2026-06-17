import unittest

from powerplans.parser import parse_power_schemes, parse_power_settings


POWERCFG_SAMPLE = """
Power Scheme GUID: bc31b04a-3e13-4d42-9d4d-ca5128c4b83a  (Battery - Optimized)
  Subgroup GUID: 54533251-82be-4824-96c1-47b60b740d00  (Processor power management)
    GUID Alias: SUB_PROCESSOR
    Power Setting GUID: bc5038f7-23e0-4960-96da-33abaf5935ec  (Maximum processor state)
      GUID Alias: PROCTHROTTLEMAX
      Minimum Possible Setting: 0x00000000
      Maximum Possible Setting: 0x00000064
      Possible Settings increment: 0x00000001
      Possible Settings units: %
    Current AC Power Setting Index: 0x00000064
    Current DC Power Setting Index: 0x00000050

  Subgroup GUID: 5fb4938d-1ee8-4b0f-9a3c-5036b0ab995c  (Graphics settings)
    GUID Alias: SUB_GRAPHICS
    Power Setting GUID: dd848b2a-8a5d-4451-9ae2-39cd41658f6c  (GPU preference policy)
      GUID Alias: GPUPREFERENCEPOLICY
      Possible Setting Index: 000
      Possible Setting Friendly Name: None
      Possible Setting Index: 001
      Possible Setting Friendly Name: Low Power
    Current AC Power Setting Index: 0x00000000
    Current DC Power Setting Index: 0x00000001

  Subgroup GUID: c763b4ec-0e50-4b6b-9bed-2b92a6ee884e  (Vendor Power Extension)
    Power Setting GUID: 7ec1751b-60ed-4588-afb5-9819d3d77d90  (Overlay)
      Possible Setting Index: 000
      Possible Setting Friendly Name: Battery saver
      Possible Setting Index: 003
      Possible Setting Friendly Name: Best performance
    Current AC Power Setting Index: 0x00000003
    Current DC Power Setting Index: 0x00000000

  Subgroup GUID: 501a4d13-42af-4429-9fd1-a8218c268e20  (PCI Express)
    GUID Alias: SUB_PCIEXPRESS
    Power Setting GUID: ee12f906-d277-404b-b6da-e5fa1a576df5  (Link State Power Management)
      GUID Alias: ASPM
      Possible Setting Index: 000
      Possible Setting Friendly Name: Off
      Possible Setting Index: 002
      Possible Setting Friendly Name: Maximum power savings
    Current AC Power Setting Index: 0x00000000
    Current DC Power Setting Index: 0x00000002
"""


class PowerPlanParserTests(unittest.TestCase):
    def test_parse_power_schemes(self):
        schemes = parse_power_schemes(
            """
Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)
Power Scheme GUID: bc31b04a-3e13-4d42-9d4d-ca5128c4b83a  (Battery - Optimized) *
"""
        )
        self.assertEqual(len(schemes), 2)
        self.assertTrue(schemes[1].active)

    def test_parse_portable_and_extension_settings(self):
        snapshot = parse_power_settings(POWERCFG_SAMPLE)
        self.assertEqual(snapshot.scheme_name, "Battery - Optimized")
        self.assertEqual(len(snapshot.groups), 4)
        self.assertEqual(len(snapshot.settings), 4)

        cpu = snapshot.by_domain("cpu")[0]
        self.assertEqual(cpu.key, "SUB_PROCESSOR/PROCTHROTTLEMAX")
        self.assertEqual(cpu.maximum, 100)
        self.assertEqual(cpu.ac_value, 100)
        self.assertEqual(cpu.dc_value, 80)

        graphics = snapshot.by_domain("graphics")
        self.assertEqual(len(graphics), 2)
        gpu_policy = next(
            setting
            for setting in graphics
            if setting.key == "SUB_GRAPHICS/GPUPREFERENCEPOLICY"
        )
        self.assertEqual(
            [(option.index, option.name) for option in gpu_policy.options],
            [(0, "None"), (1, "Low Power")],
        )
        self.assertTrue(
            any(setting.key == "SUB_PCIEXPRESS/ASPM" for setting in graphics)
        )

        extension = next(
            setting
            for setting in snapshot.by_domain("other")
            if "Vendor Power Extension" in setting.group_name
        )
        self.assertEqual(extension.ac_value, 3)
        self.assertEqual(extension.dc_value, 0)

    def test_parse_rejects_output_without_scheme(self):
        with self.assertRaises(ValueError):
            parse_power_settings("No power settings here")


if __name__ == "__main__":
    unittest.main()
