# Power Plans Backend Contract

This document is the handoff contract for code that edits or displays
PerfAnalyze power plans. The catalog editor and Windows deployment are separate
operations. Saving a catalog must never modify or activate a Windows scheme.

## Source Of Truth

The persisted catalog defaults to:

```text
power_plan_catalog.json
```

The complete example and default values are in:

```text
power_plan_catalog.json.example
```

When `power_plan_catalog.json` does not exist,
`load_power_plan_catalog()` returns an in-memory default catalog. It does not
create the file until `write_power_plan_catalog()` or
`PowerPlanEditorState.save()` is called.

## Module Ownership

- `powerplans/catalog.py`: the five portable Windows setting definitions.
- `powerplans/models.py`: validated immutable catalog data classes and defaults.
- `powerplans/store.py`: atomic JSON loading and writing.
- `powerplans/editor_state.py`: mutable draft workflow for a GUI.
- `powerplans/windows.py`: read-only `powercfg` discovery.
- `powerplans/parser.py`: parsed Windows schemes and setting capabilities.
- `powerplans/service.py`: create, update, activate, and delete command builders.
- `powerplans/commands.py`: dry-run-first command execution safety locks.
- `power_plan_cli.py`: inspection and command-preview interface; no apply command.

GUI code should import these modules rather than recreate their validation,
serialization, or Windows command logic.

## Catalog Shape

The current schema version is `2`.

```json
{
  "schema_version": 2,
  "plans": {
    "cpu_focus": {
      "name": "PerfAnalyze CPU Focus",
      "cpu_intent": 4,
      "graphics_intent": 1,
      "total_power_level": 3,
      "enabled": true,
      "base_scheme": "SCHEME_BALANCED",
      "windows_guid": null,
      "description": "Prioritize CPU throughput.",
      "settings": {
        "SUB_PROCESSOR/PROCTHROTTLEMAX": {
          "ac": 100,
          "dc": 90
        }
      }
    }
  },
  "classification_to_plan": {
    "compiling": "cpu_focus",
    "cpu_rendering": "cpu_focus"
  }
}
```

There is no fixed plan count. `plans` is a dictionary keyed by stable
`plan_id`. Multiple classifications can point to the same plan.

## Data Classes

### `PlanSettingValue`

```python
PlanSettingValue(ac: int | None = None, dc: int | None = None)
```

- `ac` is the value used while plugged in.
- `dc` is the value used on battery.
- At least one value is required.
- Values must be non-negative integers.
- A missing AC or DC value means that side is omitted from generated commands.

### `PlanDefinition`

```python
PlanDefinition(
    plan_id: str,
    name: str,
    cpu_intent: int,
    graphics_intent: int,
    total_power_level: int,
    enabled: bool = True,
    base_scheme: str = "SCHEME_BALANCED",
    windows_guid: str | None = None,
    description: str = "",
    settings: dict[str, PlanSettingValue] = {},
)
```

- `plan_id` is the persistent identity and dictionary key. Treat it as
  immutable after creation.
- Valid IDs use lowercase letters, numbers, `_`, and `-`.
- `name` is the user-visible name.
- `cpu_intent`, `graphics_intent`, and `total_power_level` are independent
  metadata values from `0` to `4`.
- Intent values describe policy and later switching behavior. They are not
  direct Windows setting values.
- `enabled=False` means the plan should not be offered for automatic
  activation. The model currently permits mappings to disabled plans, so the
  GUI should warn about that until catalog-level validation is added.
- `base_scheme` is the Windows scheme duplicated when first deploying the plan.
  `SCHEME_BALANCED` is the portable default.
- `windows_guid=None` means the catalog plan has not been deployed to Windows.
- A non-null `windows_guid` is the link to its installed managed Windows scheme.
  The GUI must not invent or casually edit this value.
- `settings` contains only the portable setting keys listed below.

### `PowerPlanCatalog`

```python
PowerPlanCatalog(
    plans: dict[str, PlanDefinition],
    classification_to_plan: dict[str, str],
    schema_version: int = 2,
)
```

- At least one plan must remain.
- Each dictionary key must equal the embedded `PlanDefinition.plan_id`.
- Classification labels are normalized lowercase strings.
- Every mapped plan ID must exist.
- Removing a plan through `PowerPlanEditorState` also removes mappings that
  pointed to it.

## Portable Settings

Only these generic Windows aliases are accepted:

| Key | Domain | Meaning |
| --- | --- | --- |
| `SUB_PROCESSOR/PROCTHROTTLEMAX` | CPU | Maximum processor state, normally `0-100` percent |
| `SUB_PROCESSOR/PERFBOOSTMODE` | CPU | Processor boost policy index |
| `SUB_PROCESSOR/PERFEPP` | CPU | Energy/performance preference; lower favors performance |
| `SUB_GRAPHICS/GPUPREFERENCEPOLICY` | Graphics | Windows graphics preference policy |
| `SUB_PCIEXPRESS/ASPM` | Graphics | PCIe link-state power saving policy |

The JSON stores Windows setting indexes, not display text. Do not hardcode
option ranges in GUI validation. Use `PowerCfgDiscovery.discover_settings()` and
the returned `DiscoveredPowerSetting`:

- `minimum`, `maximum`, `increment`, and `units` describe numeric controls.
- `options` describes enumerated controls.
- `ac_value` and `dc_value` are the current values of the inspected scheme.
- A portable setting can be absent on a particular machine.

Unsupported settings are omitted from generated commands and returned as
warnings. Vendor-specific setting GUIDs are intentionally rejected by the data
model.

## Persistence API

Use the store functions for direct persistence:

```python
from powerplans.store import (
    load_power_plan_catalog,
    write_power_plan_catalog,
)

catalog = load_power_plan_catalog("power_plan_catalog.json")
write_power_plan_catalog(catalog, "power_plan_catalog.json")
```

Writes are atomic: JSON is written and flushed to a temporary file in the same
directory, then moved over the destination with `os.replace()`.

For a GUI draft workflow, use:

```python
from powerplans.editor_state import PowerPlanEditorState

state = PowerPlanEditorState.load("power_plan_catalog.json")
new_plan = state.add_plan()
state.update_plan(updated_plan)
removed_labels = state.remove_plan(plan_id)
saved_path = state.save()
state.reload()
state.restore_defaults()
```

`state.dirty` tracks unsaved catalog changes. Model objects are frozen
dataclasses, so edits should construct a new `PlanDefinition`, commonly with
`dataclasses.replace()`, and pass it to `state.update_plan()`.

Saving the editor state only writes JSON. It does not call `powercfg`.

## Classification Mapping

Runtime resolution is conceptually:

```python
plan_id = catalog.classification_to_plan[predicted_label.lower()]
plan = catalog.plans[plan_id]
```

The mapping editor should:

1. Present every model classification.
2. Offer enabled catalog plans as targets.
3. Allow multiple classifications to select the same plan.
4. Warn about unmapped classifications.
5. Rebuild a validated `PowerPlanCatalog` when mappings change.

Mapping CRUD helpers are not yet present in `PowerPlanEditorState`; GUI code
should avoid mutating `state.catalog.classification_to_plan` in place. Construct
a new `PowerPlanCatalog` or wait for the mapping-state API to be added.

## Windows Scheme Lifecycle

Catalog persistence and Windows deployment have different lifecycles:

1. A user creates and saves a catalog plan with `windows_guid=None`.
2. Read-only discovery checks which settings the device supports.
3. `ManagedPowerPlanService.build_create()` duplicates `base_scheme`, names it
   with the `PerfAnalyze ` ownership prefix, and generates supported settings.
4. The returned `ManagedPlanOperation.resulting_plan` contains the generated
   Windows GUID.
5. Only after every create command succeeds should that resulting plan replace
   the catalog plan and be saved.
6. Later update, activation, and deletion operations require an installed
   scheme whose name has the `PerfAnalyze ` prefix.

Do not use `windows_guid` as the catalog identity. `plan_id` survives deployment,
deletion, or recreation; `windows_guid` may change.

## Command Safety

`ManagedPowerPlanService` builds command objects. It does not imply permission
to run them.

`PowerCfgExecutor` defaults are:

```python
PowerCfgExecutor(
    dry_run=True,
    allow_mutation=False,
    allow_destructive=False,
)
```

Real mutation requires both `dry_run=False` and `allow_mutation=True`. Deletion
also requires `allow_destructive=True`.

Additional safeguards:

- Create and update never activate a plan implicitly.
- Update, activation, and deletion verify PerfAnalyze ownership.
- Active managed plans cannot be updated or deleted by default.
- Built-in Windows schemes are protected from deletion.
- `temporary_activation()` captures and restores the original scheme GUID in a
  `finally` block.
- `power_plan_cli.py` intentionally exposes no apply or execute command.

The GUI should first expose command preview and explicit confirmation. It should
not construct a mutation-enabled executor during ordinary editor startup.

## Default Plans

The default catalog contains:

- `quiet`: CPU 0, graphics 0, total 0
- `everyday`: CPU 1, graphics 1, total 1
- `cpu_focus`: CPU 4, graphics 1, total 3
- `graphics_focus`: CPU 2, graphics 4, total 3
- `maximum_mixed`: CPU 4, graphics 4, total 4

Exact AC/DC defaults live in `default_power_plan_catalog()` and
`power_plan_catalog.json.example`. Those files are authoritative; the GUI
should render data from the catalog rather than keep a second copy.

## GUI Integration Checklist

1. Load a `PowerPlanEditorState`; do not parse JSON manually.
2. Render `state.catalog.plans` without imposing a fixed count.
3. Keep `plan_id` stable and display `name` separately.
4. Build new validated dataclass instances from form input.
5. Display CPU intent, graphics intent, and total power independently.
6. Generate setting rows from `PORTABLE_SETTING_SPECS`.
7. Use read-only discovery for device-specific ranges and option labels.
8. Keep Save Catalog separate from Preview/Deploy to Windows.
9. Treat `windows_guid` as backend-managed deployment state.
10. Show validation errors without replacing the last valid draft object.
11. Warn before removing mappings when a plan is deleted.
12. Do not connect automatic switching until managed deployment and activation
    have their own explicit, tested integration path.

