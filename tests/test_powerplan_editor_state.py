import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from powerplans.editor_state import PowerPlanEditorState
from powerplans.models import (
    PlanDefinition,
    PowerPlanCatalog,
    default_power_plan_catalog,
)


class PowerPlanEditorStateTests(unittest.TestCase):
    def test_add_plan_generates_unlimited_unique_ids(self):
        state = PowerPlanEditorState(default_power_plan_catalog())

        added = [state.add_plan() for _ in range(12)]

        self.assertEqual(added[0].plan_id, "custom_1")
        self.assertEqual(added[-1].plan_id, "custom_12")
        self.assertEqual(len(state.catalog.plans), 17)
        self.assertTrue(state.dirty)

    def test_remove_plan_cleans_affected_classification_mappings(self):
        state = PowerPlanEditorState(default_power_plan_catalog())

        removed = state.remove_plan("graphics_focus")

        self.assertEqual(removed, ("gaming", "gpu_rendering"))
        self.assertNotIn("graphics_focus", state.catalog.plans)
        self.assertNotIn("gaming", state.catalog.classification_to_plan)

    def test_remove_refuses_to_delete_the_last_plan(self):
        plan = PlanDefinition(
            plan_id="only",
            name="Only",
            cpu_intent=1,
            graphics_intent=1,
            total_power_level=1,
        )
        state = PowerPlanEditorState(
            PowerPlanCatalog(plans={"only": plan}, classification_to_plan={})
        )

        with self.assertRaisesRegex(ValueError, "At least one"):
            state.remove_plan("only")

    def test_update_and_save_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.json"
            state = PowerPlanEditorState(default_power_plan_catalog(), path)
            updated = replace(
                state.catalog.plans["cpu_focus"],
                name="Compile Mode",
                cpu_intent=3,
            )
            state.update_plan(updated)

            saved_path = state.save()
            reloaded = PowerPlanEditorState.load(path)

        self.assertEqual(saved_path, path.resolve())
        self.assertFalse(state.dirty)
        self.assertEqual(
            reloaded.catalog.plans["cpu_focus"].name,
            "Compile Mode",
        )
        self.assertEqual(
            reloaded.catalog.plans["cpu_focus"].cpu_intent,
            3,
        )


if __name__ == "__main__":
    unittest.main()
