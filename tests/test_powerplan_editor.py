import unittest

from powerplans.editor import PowerPlansEditor
from powerplans.parser import DiscoveredPowerSetting, PowerSettingOption


class PowerPlansEditorValidationTests(unittest.TestCase):
    @staticmethod
    def _editor_with(setting):
        editor = object.__new__(PowerPlansEditor)
        editor.support_by_key = {setting.key.upper(): setting}
        return editor

    def test_discovered_option_indexes_are_enforced(self):
        setting = DiscoveredPowerSetting(
            group_guid="group",
            group_name="Graphics",
            group_alias="SUB_GRAPHICS",
            setting_guid="setting",
            setting_name="GPU preference",
            setting_alias="GPUPREFERENCEPOLICY",
            options=(
                PowerSettingOption(index=0, name="Normal"),
                PowerSettingOption(index=1, name="Low Power"),
            ),
        )
        editor = self._editor_with(setting)

        editor._validate_discovered_value(setting.key, "AC", 1)
        with self.assertRaisesRegex(ValueError, "discovered option"):
            editor._validate_discovered_value(setting.key, "AC", 2)

    def test_discovered_numeric_range_and_increment_are_enforced(self):
        setting = DiscoveredPowerSetting(
            group_guid="group",
            group_name="Processor",
            group_alias="SUB_PROCESSOR",
            setting_guid="setting",
            setting_name="Maximum processor state",
            setting_alias="PROCTHROTTLEMAX",
            minimum=20,
            maximum=100,
            increment=5,
        )
        editor = self._editor_with(setting)

        editor._validate_discovered_value(setting.key, "Battery", 80)
        with self.assertRaisesRegex(ValueError, "at least 20"):
            editor._validate_discovered_value(setting.key, "Battery", 10)
        with self.assertRaisesRegex(ValueError, "increment 5"):
            editor._validate_discovered_value(setting.key, "Battery", 82)

    def test_validation_is_portable_when_support_has_not_been_checked(self):
        editor = object.__new__(PowerPlansEditor)
        editor.support_by_key = None

        editor._validate_discovered_value(
            "SUB_PROCESSOR/PROCTHROTTLEMAX",
            "AC",
            100,
        )


if __name__ == "__main__":
    unittest.main()
