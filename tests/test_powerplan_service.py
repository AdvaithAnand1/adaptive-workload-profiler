import unittest
from dataclasses import replace
from unittest import mock

from powerplans.commands import (
    PowerCfgCommand,
    PowerCfgCommandResult,
    PowerCfgExecutor,
)
from powerplans.models import PlanDefinition, PlanSettingValue
from powerplans.parser import parse_power_settings
from powerplans.service import ManagedPowerPlanService
from powerplans.parser import WindowsPowerScheme


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


class FakeExecutor:
    def __init__(self):
        self.commands = []

    def execute(self, command):
        self.commands.append(command)
        return PowerCfgCommandResult(command=command, executed=False)


class FakeDiscovery:
    def __init__(self, schemes):
        self.schemes = list(schemes)

    def list_schemes(self):
        return list(self.schemes)


def _schemes():
    return [
        WindowsPowerScheme(BALANCED_GUID, "Balanced"),
        WindowsPowerScheme(ACTIVE_GUID, "User Current Plan", active=True),
    ]


def _plan(**overrides):
    values = {
        "plan_id": "cpu_focus",
        "name": "CPU Focus",
        "cpu_intent": 4,
        "graphics_intent": 1,
        "total_power_level": 3,
        "settings": {
            "SUB_PROCESSOR/PROCTHROTTLEMAX": PlanSettingValue(ac=100, dc=80)
        },
    }
    values.update(overrides)
    return PlanDefinition(**values)


class PowerCfgExecutorTests(unittest.TestCase):
    @mock.patch("powerplans.commands.subprocess.run")
    def test_dry_run_never_starts_subprocess(self, run):
        executor = PowerCfgExecutor()
        command = PowerCfgCommand(("/setactive", TARGET_GUID), "activate")

        result = executor.execute(command)

        self.assertFalse(result.executed)
        run.assert_not_called()

    @mock.patch("powerplans.commands.subprocess.run")
    def test_mutation_requires_second_explicit_lock(self, run):
        executor = PowerCfgExecutor(dry_run=False, allow_mutation=False)
        command = PowerCfgCommand(("/setactive", TARGET_GUID), "activate")

        with self.assertRaises(PermissionError):
            executor.execute(command)
        run.assert_not_called()

    @mock.patch("powerplans.commands.subprocess.run")
    def test_delete_requires_separate_destructive_lock(self, run):
        executor = PowerCfgExecutor(dry_run=False, allow_mutation=True)
        command = PowerCfgCommand(
            ("/delete", TARGET_GUID),
            "delete",
            destructive=True,
        )

        with self.assertRaises(PermissionError):
            executor.execute(command)
        run.assert_not_called()

    @mock.patch("powerplans.commands.subprocess.run")
    def test_mutation_runs_only_with_both_explicit_opt_ins(self, run):
        run.return_value = mock.Mock(
            returncode=0,
            stdout="",
            stderr="",
        )
        executor = PowerCfgExecutor(dry_run=False, allow_mutation=True)
        command = PowerCfgCommand(("/setactive", TARGET_GUID), "activate")

        result = executor.execute(command)

        self.assertTrue(result.executed)
        run.assert_called_once()


class ManagedPowerPlanServiceTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = parse_power_settings(SETTING_SAMPLE)
        self.executor = FakeExecutor()
        self.service = ManagedPowerPlanService(
            executor=self.executor,
            discovery=FakeDiscovery(_schemes()),
        )

    def test_create_generates_clone_name_and_setting_commands_without_activation(self):
        operation = self.service.build_create(
            _plan(),
            schemes=_schemes(),
            snapshot=self.snapshot,
            destination_guid=TARGET_GUID,
        )

        self.assertEqual(operation.target_guid, TARGET_GUID)
        self.assertEqual(operation.resulting_plan.windows_guid, TARGET_GUID)
        arguments = [command.arguments for command in operation.commands]
        self.assertEqual(
            arguments[0],
            ("/duplicatescheme", BALANCED_GUID, TARGET_GUID),
        )
        self.assertEqual(arguments[1][0:3], ("/changename", TARGET_GUID, "PerfAnalyze CPU Focus"))
        self.assertIn("plan_id=cpu_focus", arguments[1][3])
        self.assertEqual(arguments[2][0], "/setacvalueindex")
        self.assertEqual(arguments[3][0], "/setdcvalueindex")
        self.assertNotIn("/setactive", [args[0] for args in arguments])

    def test_unsupported_portable_setting_is_reported_and_skipped(self):
        plan = _plan(
            settings={
                "SUB_PROCESSOR/PERFEPP": PlanSettingValue(ac=20, dc=80)
            }
        )
        operation = self.service.build_create(
            plan,
            schemes=_schemes(),
            snapshot=self.snapshot,
            destination_guid=TARGET_GUID,
        )

        self.assertEqual(len(operation.commands), 2)
        self.assertIn("Skipped unsupported setting", operation.warnings[0])

    def test_setting_value_is_validated_against_discovery(self):
        plan = _plan(
            settings={
                "SUB_PROCESSOR/PROCTHROTTLEMAX": PlanSettingValue(ac=101)
            }
        )
        with self.assertRaises(ValueError):
            self.service.build_create(
                plan,
                schemes=_schemes(),
                snapshot=self.snapshot,
                destination_guid=TARGET_GUID,
            )

    def test_update_refuses_unmanaged_windows_scheme(self):
        plan = replace(_plan(), windows_guid=ACTIVE_GUID)
        with self.assertRaisesRegex(ValueError, "not marked as PerfAnalyze-managed"):
            self.service.build_update(
                plan,
                schemes=_schemes(),
                snapshot=self.snapshot,
            )

    def test_update_refuses_active_managed_scheme_by_default(self):
        plan = replace(_plan(), windows_guid=TARGET_GUID)
        schemes = [
            WindowsPowerScheme(
                TARGET_GUID,
                "PerfAnalyze CPU Focus",
                active=True,
            )
        ]
        with self.assertRaisesRegex(ValueError, "is active"):
            self.service.build_update(
                plan,
                schemes=schemes,
                snapshot=self.snapshot,
            )

    def test_delete_refuses_active_managed_scheme(self):
        plan = replace(_plan(), windows_guid=TARGET_GUID)
        schemes = [
            WindowsPowerScheme(
                TARGET_GUID,
                "PerfAnalyze CPU Focus",
                active=True,
            )
        ]
        with self.assertRaisesRegex(ValueError, "active"):
            self.service.build_delete(plan, schemes=schemes)

    def test_activate_managed_refuses_unmanaged_windows_scheme(self):
        plan = replace(_plan(), windows_guid=ACTIVE_GUID)

        with self.assertRaisesRegex(ValueError, "not marked as PerfAnalyze-managed"):
            self.service.build_activate_managed(plan, schemes=_schemes())

    def test_activate_managed_targets_owned_scheme(self):
        plan = replace(_plan(), windows_guid=TARGET_GUID)
        schemes = [
            WindowsPowerScheme(
                TARGET_GUID,
                "PerfAnalyze CPU Focus",
                active=False,
            )
        ]

        operation = self.service.build_activate_managed(plan, schemes=schemes)

        self.assertEqual(operation.operation, "activate")
        self.assertEqual(operation.target_guid, TARGET_GUID)
        self.assertEqual(
            operation.commands[0].arguments,
            ("/setactive", TARGET_GUID),
        )

    def test_delete_command_is_marked_destructive(self):
        plan = replace(_plan(), windows_guid=TARGET_GUID)
        schemes = [
            WindowsPowerScheme(
                TARGET_GUID,
                "PerfAnalyze CPU Focus",
                active=False,
            )
        ]
        operation = self.service.build_delete(plan, schemes=schemes)
        self.assertTrue(operation.commands[0].destructive)

    def test_temporary_activation_restores_original_guid_after_error(self):
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with self.service.temporary_activation(
                plan_id="cpu_focus",
                target_guid=TARGET_GUID,
            ) as original:
                self.assertEqual(original.guid, ACTIVE_GUID)
                self.assertEqual(original.name, "User Current Plan")
                raise RuntimeError("body failed")

        self.assertEqual(
            [command.arguments for command in self.executor.commands],
            [
                ("/setactive", TARGET_GUID),
                ("/setactive", ACTIVE_GUID),
            ],
        )


if __name__ == "__main__":
    unittest.main()
