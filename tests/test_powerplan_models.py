import tempfile
import unittest
from pathlib import Path

from powerplans.models import (
    PlanDefinition,
    PlanSettingValue,
    PowerPlanCatalog,
    default_power_plan_catalog,
    migrate_legacy_label_mapping,
)
from powerplans.catalog import PORTABLE_SETTING_KEYS
from powerplans.store import load_power_plan_catalog, write_power_plan_catalog


class PowerPlanModelTests(unittest.TestCase):
    def test_default_catalog_has_independent_cpu_and_graphics_modes(self):
        catalog = default_power_plan_catalog()
        self.assertEqual(len(catalog.plans), 5)
        self.assertGreater(
            catalog.plans["cpu_focus"].cpu_intent,
            catalog.plans["cpu_focus"].graphics_intent,
        )
        self.assertGreater(
            catalog.plans["graphics_focus"].graphics_intent,
            catalog.plans["graphics_focus"].cpu_intent,
        )
        self.assertEqual(
            catalog.classification_to_plan["gaming"],
            "graphics_focus",
        )
        self.assertEqual(catalog.classification_to_plan["compiling"], "cpu_focus")

    def test_default_plans_use_only_the_five_high_impact_settings(self):
        catalog = default_power_plan_catalog()
        self.assertEqual(
            PORTABLE_SETTING_KEYS,
            {
                "SUB_PROCESSOR/PROCTHROTTLEMAX",
                "SUB_PROCESSOR/PERFBOOSTMODE",
                "SUB_PROCESSOR/PERFEPP",
                "SUB_GRAPHICS/GPUPREFERENCEPOLICY",
                "SUB_PCIEXPRESS/ASPM",
            },
        )
        for plan in catalog.plans.values():
            self.assertEqual(set(plan.settings), PORTABLE_SETTING_KEYS)

    def test_default_plan_values_match_the_documented_policy(self):
        catalog = default_power_plan_catalog()

        expected = {
            "quiet": {
                "SUB_PROCESSOR/PROCTHROTTLEMAX": (60, 45),
                "SUB_PROCESSOR/PERFBOOSTMODE": (0, 0),
                "SUB_PROCESSOR/PERFEPP": (75, 90),
                "SUB_GRAPHICS/GPUPREFERENCEPOLICY": (1, 1),
                "SUB_PCIEXPRESS/ASPM": (2, 2),
            },
            "everyday": {
                "SUB_PROCESSOR/PROCTHROTTLEMAX": (100, 85),
                "SUB_PROCESSOR/PERFBOOSTMODE": (1, 0),
                "SUB_PROCESSOR/PERFEPP": (35, 70),
                "SUB_GRAPHICS/GPUPREFERENCEPOLICY": (0, 1),
                "SUB_PCIEXPRESS/ASPM": (1, 2),
            },
            "cpu_focus": {
                "SUB_PROCESSOR/PROCTHROTTLEMAX": (100, 90),
                "SUB_PROCESSOR/PERFBOOSTMODE": (2, 1),
                "SUB_PROCESSOR/PERFEPP": (10, 35),
                "SUB_GRAPHICS/GPUPREFERENCEPOLICY": (1, 1),
                "SUB_PCIEXPRESS/ASPM": (1, 2),
            },
            "graphics_focus": {
                "SUB_PROCESSOR/PROCTHROTTLEMAX": (85, 75),
                "SUB_PROCESSOR/PERFBOOSTMODE": (1, 0),
                "SUB_PROCESSOR/PERFEPP": (35, 60),
                "SUB_GRAPHICS/GPUPREFERENCEPOLICY": (0, 0),
                "SUB_PCIEXPRESS/ASPM": (0, 1),
            },
            "maximum_mixed": {
                "SUB_PROCESSOR/PROCTHROTTLEMAX": (100, 90),
                "SUB_PROCESSOR/PERFBOOSTMODE": (2, 1),
                "SUB_PROCESSOR/PERFEPP": (0, 25),
                "SUB_GRAPHICS/GPUPREFERENCEPOLICY": (0, 0),
                "SUB_PCIEXPRESS/ASPM": (0, 1),
            },
        }

        for plan_id, setting_values in expected.items():
            plan = catalog.plans[plan_id]
            actual = {
                key: (value.ac, value.dc)
                for key, value in plan.settings.items()
            }
            self.assertEqual(actual, setting_values)

    def test_focus_modes_have_distinct_cpu_and_graphics_behavior(self):
        catalog = default_power_plan_catalog()
        cpu = catalog.plans["cpu_focus"].settings
        graphics = catalog.plans["graphics_focus"].settings

        self.assertGreater(
            cpu["SUB_PROCESSOR/PROCTHROTTLEMAX"].ac,
            graphics["SUB_PROCESSOR/PROCTHROTTLEMAX"].ac,
        )
        self.assertGreater(
            cpu["SUB_PROCESSOR/PERFBOOSTMODE"].ac,
            graphics["SUB_PROCESSOR/PERFBOOSTMODE"].ac,
        )
        self.assertGreater(
            cpu["SUB_GRAPHICS/GPUPREFERENCEPOLICY"].ac,
            graphics["SUB_GRAPHICS/GPUPREFERENCEPOLICY"].ac,
        )
        self.assertGreater(
            cpu["SUB_PCIEXPRESS/ASPM"].ac,
            graphics["SUB_PCIEXPRESS/ASPM"].ac,
        )

    def test_catalog_roundtrip_preserves_unlimited_named_plans_and_settings(self):
        plans = {}
        mapping = {}
        for index in range(20):
            plan_id = f"custom_{index}"
            plans[plan_id] = PlanDefinition(
                plan_id=plan_id,
                name=f"Custom {index}",
                cpu_intent=index % 5,
                graphics_intent=(index + 2) % 5,
                total_power_level=(index + 1) % 5,
                settings={
                    "SUB_PROCESSOR/PROCTHROTTLEMAX": PlanSettingValue(
                        ac=100 - index,
                        dc=80 - index,
                    )
                },
            )
            mapping[f"class_{index}"] = plan_id
        catalog = PowerPlanCatalog(plans=plans, classification_to_plan=mapping)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "catalog.json"
            write_power_plan_catalog(catalog, path)
            loaded = load_power_plan_catalog(path)

        self.assertEqual(loaded.to_dict(), catalog.to_dict())

    def test_catalog_rejects_mapping_to_unknown_plan(self):
        plan = PlanDefinition(
            plan_id="known",
            name="Known",
            cpu_intent=1,
            graphics_intent=1,
            total_power_level=1,
        )
        with self.assertRaises(ValueError):
            PowerPlanCatalog(
                plans={"known": plan},
                classification_to_plan={"gaming": "missing"},
            )

    def test_plan_rejects_boolean_intent(self):
        with self.assertRaises(ValueError):
            PlanDefinition(
                plan_id="invalid",
                name="Invalid",
                cpu_intent=True,
                graphics_intent=1,
                total_power_level=1,
            )

    def test_plan_rejects_vendor_specific_setting_keys(self):
        with self.assertRaises(ValueError):
            PlanDefinition(
                plan_id="vendor_specific",
                name="Vendor Specific",
                cpu_intent=1,
                graphics_intent=1,
                total_power_level=1,
                settings={
                    "c763b4ec-0e50-4b6b-9bed-2b92a6ee884e/"
                    "7ec1751b-60ed-4588-afb5-9819d3d77d90": PlanSettingValue(
                        ac=3,
                        dc=0,
                    )
                },
            )

    def test_legacy_mapping_migrates_to_new_default_plan_ids(self):
        migrated = migrate_legacy_label_mapping(
            {
                "idle": "silent",
                "light": "balanced",
                "heavy": "performance",
            }
        )
        self.assertEqual(
            migrated,
            {
                "idle": "quiet",
                "light": "everyday",
                "heavy": "maximum_mixed",
            },
        )


if __name__ == "__main__":
    unittest.main()
