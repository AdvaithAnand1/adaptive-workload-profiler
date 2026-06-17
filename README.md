# PerfAnalyze

PerfAnalyze classifies current Windows workload state from live OS telemetry and can switch G-Helper performance profiles automatically.

Current flow:
1. Sample CPU/memory/disk/network/process telemetry from the OS.
2. Convert metrics into a fixed feature vector.
3. Train a small classifier on labeled samples.
4. Run a controller loop that predicts workload and sends mapped hotkeys.

## Requirements

- Windows
- G-Helper running (only needed for `controller.py`)
- Python 3.10+ recommended

Install dependencies:

```bash
pip install -r requirements.txt
```

If `python` points to a different runtime on your machine, use the Windows launcher:

```bash
py -3.12 -m pip install -r requirements.txt
```

## Files

- `monitor.py`: OS-native telemetry sampler + feature engineering.
- `record.py`: Record one labeled telemetry CSV per session under `data/`.
- `collect_dataset.py`: Run a multi-label timed collection plan in one command.
- `model.py`: `SystemStateNet` model definition.
- `model_artifacts.py`: Load normalized runtime bundles and validate model metadata.
- `train_model.py`: Train from all session CSVs and export versioned models.
- `training_gui.py`: Separate training tool for app-guided labeling and collection approval.
- `controller.py`: Live inference loop and profile-switch hotkeys.
- `oracle_client.py`: Minimal oracle adapter (`silent|balanced|performance`) with demo fallback.
- `demo_gui.py`: GUI demo of telemetry -> prediction -> target profile -> oracle action.
- `run_probe.py`: Timed inference logger for real-world validation sessions.
- `analyze_probe.py`: Post-run scoring/diagnostics for probe logs.
- `tune_probe_config.py`: Turn probe evidence into recommended switching config changes.
- `analyze_training_data.py`: Inspect label/session/power-saver coverage before retraining.
- `inspect_oracle_backend.py`: Inspect backend selection and discovered power-plan mappings.
- `inspect_power_settings.py`: Read-only discovery of portable Windows power settings.
- `power_plan_cli.py`: Inspect catalogs and Windows support, and preview managed commands.
- `preview_power_plan.py`: Render managed-plan commands without executing them.
- `powerplans/`: Plan models, persistence, discovery, and safe command generation.
- `POWER_PLANS_BACKEND.md`: Backend and persistence contract for GUI integration.
- `power_plan_catalog.json.example`: Initial unlimited-plan catalog and mappings.

## Feature Schema (`os_native_v3`)

Training and inference use the same fixed-order datapoints from `monitor.FEATURE_NAMES`:

1. `Total CPU Usage [%]`
2. `CPU Usage EMA [%]`
3. `CPU Frequency [MHz]`
4. `CPU Frequency Ratio [%]`
5. `CPU Core Usage StdDev [%]`
6. `CPU Core Usage StdDev EMA [%]`
7. `CPU User Time [%]`
8. `CPU System Time [%]`
9. `Context Switches [/s]`
10. `Interrupts [/s]`
11. `RAM Usage [%]`
12. `RAM Available [MB]`
13. `Disk Read [MB/s]`
14. `Disk Write [MB/s]`
15. `Network Down [KB/s]`
16. `Network Up [KB/s]`
17. `Process Count`
18. `On Battery [0/1]`
19. `Power Saver [0/1]`
20. `CPU Physical Cores`
21. `CPU Logical Cores`
22. `GPU Usage [%]`
23. `GPU 3D Usage [%]`
24. `GPU Compute Usage [%]`
25. `GPU Video Usage [%]`
26. `GPU Copy Usage [%]`
27. `GPU Dedicated Memory [MB]`
28. `GPU Shared Memory [MB]`
29. `GPU Available [0/1]`

GPU data comes from Windows performance counters and is sampled in the
background. Each `record.py` run writes a separate session file so training can
keep complete sessions isolated between train and test sets. The shipping
model is normalized at train time and loaded with matching metadata; old raw
artifacts are rejected and must be retrained.

## Run

1. Record data per workload label:

```bash
python record.py idle --interval 0.25 --duration 180
python record.py youtube_1080p --interval 0.25 --duration 300
python record.py gaming --interval 0.25 --duration 300
```

Or run one collection plan:

```bash
python collect_dataset.py --sessions idle:180,youtube_1080p:300,gaming:300 --interval 0.25 --cycles 2
```

Data quality tip:
- Record at least two separate sessions per label, across different power and
  workload configurations.

2. Train model:

```bash
python train_model.py
```

Training starts from fresh weights, recursively combines every CSV under
`data/`, splits by whole sessions, saves a version under `models/`, refreshes the
root runtime artifacts, and prints only final test accuracy.

3. Run controller:

```bash
python controller.py
```

4. Run GUI demo:

```bash
python demo_gui.py
```

Or use the PowerShell launcher (prefers `.venv` automatically):

```powershell
.\run_demo_gui.ps1
```

Optional mock telemetry mode:

```powershell
.\run_demo_gui.ps1 -Mock
```

Training-only app policies live in a separate tool:

```bash
python training_gui.py
```

That tool can propose labels from local foreground process context, but those
rules are not used by production inference and are not bundled into exported
model artifacts.

Ollama control GUI (start/stop server, prompt panel, live monitoring):

```powershell
.\run_ollama_control_gui.ps1
```

The normal GUI has two tabs: Dashboard for live state and controls, and Power
Plans for catalog editing. Diagnostics are hidden by default and available in
debug mode:

```powershell
.\run_demo_gui.ps1 -Debug
```

The dashboard is organized around:
- telemetry and inference detail
- confidence and stability meters
- start/stop, dry-run, and manual profile controls

### Power Plans Editor

The `Power Plans` tab is a catalog editor for unlimited multidimensional plans.
It provides:

- a scrollable plan list with a fixed Add button
- a per-plan `...` menu for removing a plan from the draft
- independent CPU intent, GPU intent, and total power level fields
- base scheme, read-only backend-managed Windows GUID, enabled state,
  description, and mapping summary
- AC and battery values generated from `PORTABLE_SETTING_SPECS`
- read-only support discovery for the current machine

`Save Catalog` writes `power_plan_catalog.json`. The editor does not create,
modify, activate, or delete Windows power schemes. Adding future CPU, iGPU,
platform, or AMD parameters to the setting catalog automatically adds rows to
the editor without changing its layout code. Removing a mapped plan requires
confirmation and reports the classification mappings that will be removed.

## OpenClaw Prompt Launcher

Use this helper when you want to start an OpenClaw run with a custom prompt:

```powershell
.\run_openclaw_prompt.ps1 -Prompt "Read this repo and improve dashboard card spacing."
```

Useful options:

- `-PromptFile .\my_prompt.txt` to load multi-line prompts from a file.
- `-NoConfirm` to skip the interactive confirmation.
- `-Agent main` to target a specific OpenClaw agent name.

By default it:

- asks for confirmation before starting a session
- writes logs to `run_logs\openclaw_prompt_<timestamp>.log`
- runs `openclaw agent --local ... --json`

## OpenClaw Iteration Loop (Manual)

Use this when you want repeated Codex/OpenClaw passes for deeper iteration on one goal:

```powershell
.\run_openclaw_codex_loop.ps1 -Goal "Refine demo dashboard UI and validate after each pass." -Iterations 6 -Thinking xhigh
```

Useful options:

- `-GoalFile .\goal.txt` for multi-line goals.
- `-PauseSeconds 30` to add spacing between passes.
- `-SessionId <id>` to continue an existing loop session.
- `-NoConfirm` to skip the confirmation prompt.

By default it:

- keeps one shared OpenClaw session across passes
- logs everything to `run_logs\openclaw_codex_loop_<timestamp>.log`
- supports early stop by creating `.openclaw.stop` in the repo root
- does not install any scheduled task or background service

For multi-hour unattended runs (background hidden window, still manual start):

```powershell
.\start_openclaw_codex_loop_background.ps1 -Goal "Improve model prediction quality, practical profile switching, and dashboard UX each pass." -Hours 3 -PauseSeconds 180 -Thinking xhigh
```

Stop it:

```powershell
.\stop_openclaw_codex_loop.ps1
```

Force stop immediately:

```powershell
.\stop_openclaw_codex_loop.ps1 -KillProcess
```

The app uses a minimal oracle interface:

- `set_profile(profile: str) -> any`
- `get_profile() -> str | None` (optional)

Backend order:
- external command bridge first when `PERFANALYZE_ORACLE_COMMAND` is set
- external `oracle` module next (best place for G-Helper-specific integration)
- Windows `powercfg` power-plan fallback after that
- in-memory demo stub last

Useful environment overrides:

```powershell
$env:PERFANALYZE_ORACLE_BACKEND="auto"       # auto | command | oracle_module | powercfg | demo
$env:PERFANALYZE_ORACLE_COMMAND="python oracle_command_mock.py"
$env:PERFANALYZE_POWERCFG_SILENT="SCHEME_MAX"
$env:PERFANALYZE_POWERCFG_BALANCED="SCHEME_BALANCED"
$env:PERFANALYZE_POWERCFG_PERFORMANCE="SCHEME_MIN"

# Optional generic Windows tuning knobs for the powercfg fallback
$env:PERFANALYZE_POWERCFG_SILENT_MAX_CPU="60"
$env:PERFANALYZE_POWERCFG_SILENT_BOOST_MODE="disabled"
$env:PERFANALYZE_POWERCFG_BALANCED_MAX_CPU="85"
$env:PERFANALYZE_POWERCFG_PERFORMANCE_BOOST_MODE="aggressive"
```

The command bridge is the easiest way to integrate a mock backend or future G-Helper wrapper without shipping a Python module import path. The bridge is expected to support:
- `set-profile <silent|balanced|performance>`
- `get-profile`
- `diagnostics`

Included command backends:

```bash
python oracle_command_mock.py diagnostics
python oracle_command_mock.py set-profile performance

# G-Helper-compatible hotkey bridge
python oracle_command_hotkey.py diagnostics
python oracle_command_hotkey.py set-profile performance --dry-run
```

Recommended G-Helper-style setup via command bridge:

```powershell
$env:PERFANALYZE_ORACLE_BACKEND="command"
$env:PERFANALYZE_ORACLE_COMMAND="python oracle_command_hotkey.py --config perfalyze_config.json"
# optional while testing the bridge without sending real hotkeys
$env:PERFANALYZE_HOTKEY_BACKEND_DRY_RUN="1"
```

The powercfg plan overrides can be aliases, GUIDs, or plan names/unique partial names.
The optional tuning knobs make the fallback backend more measurable by changing processor-related power settings on the selected plan.

Supported tuning variables per profile:
- `PERFANALYZE_POWERCFG_<PROFILE>_MIN_CPU` (0-100)
- `PERFANALYZE_POWERCFG_<PROFILE>_MAX_CPU` (0-100)
- `PERFANALYZE_POWERCFG_<PROFILE>_BOOST_MODE` (`disabled`, `enabled`, `aggressive`, `efficient-enabled`, `efficient-aggressive`, or `0-5`)

Inspect the current backend choice and discovered power plans:

```bash
python inspect_oracle_backend.py
```

## CPU/Graphics Power-Plan Foundation

PerfAnalyze is moving from a fixed Silent/Balanced/Performance ladder to an
unlimited catalog of named plans. Each plan has independent:

- CPU intent (`0-4`)
- graphics intent (`0-4`)
- total power level (`0-4`) for switch safety and cooldown decisions
- AC and DC values for any discovered Windows power setting
- stable PerfAnalyze plan id plus an optional generated Windows scheme GUID

The initial templates are:

- `quiet`
- `everyday`
- `cpu_focus`
- `graphics_focus`
- `maximum_mixed`

The default plans intentionally use only five high-impact Windows-defined
controls:

- maximum processor state
- processor boost mode
- processor energy-performance preference
- Windows GPU preference policy
- PCI Express link-state power management

Not every machine exposes every control. PerfAnalyze discovers support first
and omits unavailable fields rather than substituting vendor-specific commands.

Default AC/DC values:

| Plan | CPU max | Boost | EPP | GPU policy | PCIe policy |
|---|---|---|---|---|---|
| Quiet | 60 / 45 | Disabled / Disabled | 75 / 90 | Low power / Low power | Maximum savings / Maximum savings |
| Everyday | 100 / 85 | Enabled / Disabled | 35 / 70 | None / Low power | Moderate / Maximum savings |
| CPU Focus | 100 / 90 | Aggressive / Enabled | 10 / 35 | Low power / Low power | Moderate / Maximum savings |
| Graphics Focus | 85 / 75 | Enabled / Disabled | 35 / 60 | None / None | Off / Moderate |
| Maximum Mixed | 100 / 90 | Aggressive / Enabled | 0 / 25 | None / None | Off / Moderate |

Each pair is `AC / battery`. Lower EPP values favor CPU performance; higher
values favor efficiency. Windows GPU policy has no cross-vendor "maximum GPU"
setting, so graphics-focused plans remove the low-power preference and reduce
PCIe link savings instead of pretending to control clocks or wattage.

Classifications map to one plan id, and multiple classifications may share the
same plan. The schema is shown in `power_plan_catalog.json.example`.

Discover the settings available on the current machine:

```bash
python inspect_power_settings.py
python inspect_power_settings.py --all
python inspect_power_settings.py --json
```

This discovery path is read-only. Saved plans accept only standard Windows
setting aliases for processor behavior, cooling, graphics preference, and PCI
Express link-state power management. Each setting is shown only when the current
machine reports support. Vendor extension groups may appear under `--all`, but
they are not saved or applied by the portable plan system.

The Windows graphics controls are policy-level controls. They do not promise
direct GPU clocks, voltages, or vendor-specific power limits. The GUI monitor
recommends named catalog plans; it does not activate them until managed
deployment has an explicit, user-confirmed integration path.

### Managed-Plan Command Safety

Use the unified backend terminal interface for catalog inspection, read-only
Windows discovery, and command previews:

```bash
python power_plan_cli.py catalog init
python power_plan_cli.py plans list
python power_plan_cli.py plans show graphics_focus
python power_plan_cli.py mappings
python power_plan_cli.py windows schemes
python power_plan_cli.py windows settings
python power_plan_cli.py --catalog power_plan_catalog.json.example status
python power_plan_cli.py --catalog power_plan_catalog.json.example preview create cpu_focus
```

`catalog init` only writes `power_plan_catalog.json`. The `windows` commands run
read-only `powercfg` discovery, and every `preview` command prints the proposed
command sequence without executing it. This CLI intentionally has no apply or
execute subcommand. `preview_power_plan.py` remains as a smaller compatibility
wrapper for create previews.

The managed-plan service is dry-run-first:

- `PowerCfgExecutor()` never starts a subprocess.
- Real mutation requires both `dry_run=False` and `allow_mutation=True`.
- Plan deletion additionally requires `allow_destructive=True`.
- Create and update operations never activate a plan implicitly.
- Existing schemes are updated or deleted only when their Windows name carries
  the `PerfAnalyze` ownership prefix.
- Active managed schemes cannot be updated or deleted by default.
- Unsupported settings are skipped with warnings instead of sending unknown
  values to the machine.

For future opt-in integration tests, `temporary_activation(...)` captures both
the active plan name and GUID, activates the requested scheme, and restores the
captured GUID from a `finally` block even when the test body raises an error.
There is currently no command-line option that enables real mutation.

## 20-Minute Functional Probe

Use this to measure how the model behaves while you use the laptop naturally.

1. Run a timed probe (safe dry-run mode, no real profile switching):

```bash
python run_probe.py --minutes 20 --out run_logs/study_probe.csv --dry-run
```

2. Analyze the resulting log:

```bash
python analyze_probe.py --log run_logs/study_probe.csv --json-out run_logs/study_probe_summary.json
```

3. Generate tuned switching recommendations from that probe:

```bash
python tune_probe_config.py --log run_logs/study_probe.csv --write-config perfalyze_config.tuned.json
```

4. Before retraining, verify dataset coverage quality:

```bash
python analyze_training_data.py --csv training_data.csv
```

The analyzer reports:
- decision-confidence / uncertainty levels
- optional raw-vs-smoothed confidence comparison
- decision margin quality
- switch churn
- gate-reason counts (for example cooldown vs low-confidence vs downshift-hold)
- label distribution collapse risk
- power-saver sensitivity gap
- a heuristic functional score (`0-100`)

Optional synthetic telemetry mode:

```bash
# PowerShell
$env:PERFANALYZE_MOCK="1"
python record.py idle --interval 0.10 --duration 120
python controller.py
```

## Configuration Notes

- Default config path: `perfalyze_config.json`
- The legacy runtime config is still loaded for the current three-profile
  controller, but its GUI editor has been removed.
- Power-plan catalog edits are saved separately to `power_plan_catalog.json`.
- `switching.probability_ema_alpha` smooths class probabilities before the app decides a target profile.
  - Lower values = steadier but slower reactions
  - Higher values = more responsive but more jitter-prone
  - `1.0` behaves like raw unsmoothed probabilities
- `switching.downshift_extra_window` requires extra stable samples before dropping to a lower-power profile.
- `switching.downshift_hold_sec` keeps a higher profile active for a short hold period before allowing a downshift.
  - Together, these make real profile switching safer and more useful under bursty workloads.
- Profile hotkeys may be left blank for oracle/demo-backed flows that do not use keyboard automation.
- `controller.py` still needs valid hotkeys for the profiles you expect it to switch when using direct hotkey/G-Helper-style control.
- `controller.py` can also use the shared oracle path instead of direct hotkeys when `PERFANALYZE_CONTROLLER_USE_ORACLE=1` or when an oracle backend is explicitly configured.
- If no external oracle module is available, PerfAnalyze can now fall back to Windows power plans via `powercfg`.

## Current Limitations

- If you already have an old `training_data.csv` from prior telemetry formats, re-record it.
- Label names are user-defined during recording; there is no label taxonomy enforcement.
- Hotkey mapping in `controller.py` must match your local G-Helper hotkey setup for any profiles you actually want it to switch.
