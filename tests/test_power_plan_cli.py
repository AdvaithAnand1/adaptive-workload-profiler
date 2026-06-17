import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from power_plan_cli import run
from powerplans.models import PowerPlanCatalog, default_power_plan_catalog
from powerplans.parser import WindowsPowerScheme, parse_power_settings
from powerplans.store import load_power_plan_catalog, write_power_plan_catalog


BALANCED_GUID = "381b4222-f694-41f0-9685-ff5bb260df2e"
ACTIVE_GUID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
TARGET_GUID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

SETTING_SAMPLE = f"""
Power Scheme GUID: {BALANCED_GUID} (Balanced)
  Subgroup GUID: 54533251-82be-4824-96c1-47b60b740d00 (Processor power management)
    GUID Alias: SUB_PROCESSOR
    Power Setting GUID: bc5038f7-23e0-4960-96da-33abaf5935ec (Maximum processor state)
      GUID Alias: PROCTHROTTLEMAX
      Minimum Possible Setting: 0x00000000
      Maximum Possible Setting: 0x00000064
      Possible Settings increment: 0x00000001
      Possible Settings units: %
    Current AC Power Setting Index: 0x00000064
    Current DC Power Setting Index: 0x00000050
"""


class FakeDiscovery:
    def __init__(self, schemes=None):
        self.schemes = list(
            schemes
            or [
                WindowsPowerScheme(BALANCED_GUID, "Balanced"),
                WindowsPowerScheme(ACTIVE_GUID, "User Current Plan", active=True),
            ]
        )
        self.snapshot = parse_power_settings(SETTING_SAMPLE)
        self.list_calls = 0
        self.setting_calls = []

    def list_schemes(self):
        self.list_calls += 1
        return list(self.schemes)

    def discover_settings(self, scheme="SCHEME_CURRENT"):
        self.setting_calls.append(scheme)
        return self.snapshot


class PowerPlanCliTests(unittest.TestCase):
    def test_catalog_init_writes_defaults_without_windows_discovery(self):
        discovery = FakeDiscovery()
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.json"
            result = run(
                ["--catalog", str(catalog_path), "catalog", "init"],
                discovery=discovery,
                out=output,
            )
            catalog = load_power_plan_catalog(catalog_path)

        self.assertEqual(result, 0)
        self.assertIn("Wrote default catalog", output.getvalue())
        self.assertIn("Windows power settings were not changed", output.getvalue())
        self.assertIn("graphics_focus", catalog.plans)
        self.assertEqual(discovery.list_calls, 0)
        self.assertEqual(discovery.setting_calls, [])

    def test_plans_and_mappings_are_available_without_windows_discovery(self):
        discovery = FakeDiscovery()

        plans_output = io.StringIO()
        run(["plans", "list"], discovery=discovery, out=plans_output)
        mappings_output = io.StringIO()
        run(["mappings"], discovery=discovery, out=mappings_output)

        self.assertIn("cpu_focus", plans_output.getvalue())
        self.assertIn("graphics_focus", plans_output.getvalue())
        self.assertIn("gaming", mappings_output.getvalue())
        self.assertIn("graphics_focus", mappings_output.getvalue())
        self.assertEqual(discovery.list_calls, 0)
        self.assertEqual(discovery.setting_calls, [])

    def test_windows_commands_use_read_only_discovery(self):
        discovery = FakeDiscovery()
        schemes_output = io.StringIO()
        settings_output = io.StringIO()

        run(["windows", "schemes"], discovery=discovery, out=schemes_output)
        run(
            ["windows", "settings", "--scheme", BALANCED_GUID],
            discovery=discovery,
            out=settings_output,
        )

        self.assertIn("* User Current Plan", schemes_output.getvalue())
        self.assertIn("Maximum processor state", settings_output.getvalue())
        self.assertIn("AC=100 %", settings_output.getvalue())
        self.assertEqual(discovery.list_calls, 1)
        self.assertEqual(discovery.setting_calls, [BALANCED_GUID])

    def test_create_preview_is_deterministic_and_does_not_execute(self):
        discovery = FakeDiscovery()
        output = io.StringIO()

        result = run(
            [
                "preview",
                "create",
                "cpu_focus",
                "--destination-guid",
                TARGET_GUID,
            ],
            discovery=discovery,
            out=output,
        )

        rendered = output.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("DRY RUN ONLY - no commands were executed", rendered)
        self.assertIn(
            f"powercfg /duplicatescheme {BALANCED_GUID} {TARGET_GUID}",
            rendered,
        )
        self.assertIn("powercfg /setacvalueindex", rendered)
        self.assertNotIn("powercfg /setactive", rendered)
        self.assertEqual(discovery.list_calls, 1)
        self.assertEqual(discovery.setting_calls, ["SCHEME_CURRENT"])

    def test_activate_preview_requires_managed_installed_scheme(self):
        catalog = default_power_plan_catalog()
        attached_plan = replace(
            catalog.plans["cpu_focus"],
            windows_guid=TARGET_GUID,
        )
        attached_catalog = PowerPlanCatalog(
            plans={**catalog.plans, "cpu_focus": attached_plan},
            classification_to_plan=catalog.classification_to_plan,
        )
        discovery = FakeDiscovery(
            schemes=[
                WindowsPowerScheme(
                    TARGET_GUID,
                    "PerfAnalyze CPU Focus",
                    active=False,
                )
            ]
        )
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.json"
            write_power_plan_catalog(attached_catalog, catalog_path)
            result = run(
                [
                    "--catalog",
                    str(catalog_path),
                    "preview",
                    "activate",
                    "cpu_focus",
                ],
                discovery=discovery,
                out=output,
            )

        self.assertEqual(result, 0)
        self.assertIn("DRY RUN ONLY - no commands were executed", output.getvalue())
        self.assertIn(f"powercfg /setactive {TARGET_GUID}", output.getvalue())

    def test_status_reports_active_scheme_and_catalog_counts(self):
        discovery = FakeDiscovery()
        output = io.StringIO()

        run(["status"], discovery=discovery, out=output)

        rendered = output.getvalue()
        self.assertIn("Saved plans: 5", rendered)
        self.assertIn(f"User Current Plan [{ACTIVE_GUID}]", rendered)
        self.assertIn("Installed schemes: 2", rendered)


if __name__ == "__main__":
    unittest.main()
