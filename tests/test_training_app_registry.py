import unittest

from training.app_registry import default_training_app_catalog, match_training_app
from training.models import DetectedProcessContext


class TrainingAppRegistryTests(unittest.TestCase):
    def test_browser_process_maps_to_browsing(self):
        catalog = default_training_app_catalog()
        context = DetectedProcessContext(
            pid=1,
            process_name="chrome.exe",
            executable_name="chrome.exe",
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            normalized_executable_path=r"c:\program files\google\chrome\application\chrome.exe",
            parent_name="explorer.exe",
        )
        rule = match_training_app(catalog, context)
        self.assertIsNotNone(rule)
        self.assertEqual(rule.label, "browsing")

    def test_generic_javaw_does_not_match_lunar_without_parent_context(self):
        catalog = default_training_app_catalog()
        context = DetectedProcessContext(
            pid=2,
            process_name="javaw.exe",
            executable_name="javaw.exe",
            executable_path=r"C:\Program Files\Java\bin\javaw.exe",
            normalized_executable_path=r"c:\program files\java\bin\javaw.exe",
            parent_name="java.exe",
        )
        rule = match_training_app(catalog, context)
        self.assertIsNone(rule)

    def test_lunar_matches_with_launcher_context(self):
        catalog = default_training_app_catalog()
        context = DetectedProcessContext(
            pid=3,
            process_name="javaw.exe",
            executable_name="javaw.exe",
            executable_path=r"C:\Games\Lunar Client\javaw.exe",
            normalized_executable_path=r"c:\games\lunar client\javaw.exe",
            parent_name="Lunar Client Launcher",
            command_line_markers=("-jar", "lunar"),
        )
        rule = match_training_app(catalog, context)
        self.assertIsNotNone(rule)
        self.assertEqual(rule.label, "gaming")


if __name__ == "__main__":
    unittest.main()
