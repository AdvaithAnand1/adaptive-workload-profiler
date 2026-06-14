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
- `record.py`: Record labeled telemetry into `training_data.csv`.
- `collect_dataset.py`: Run a multi-label timed collection plan in one command.
- `model.py`: `SystemStateNet` model definition.
- `train_model.py`: Train model and export `model.pth` + `classes.json`.
- `controller.py`: Live inference loop and profile-switch hotkeys.
- `oracle_client.py`: Minimal oracle adapter (`silent|balanced|performance`) with demo fallback.
- `demo_gui.py`: GUI demo of telemetry -> prediction -> target profile -> oracle action.
- `run_probe.py`: Timed inference logger for real-world validation sessions.
- `analyze_probe.py`: Post-run scoring/diagnostics for probe logs.
- `tune_probe_config.py`: Turn probe evidence into recommended switching config changes.
- `analyze_training_data.py`: Inspect label/session/power-saver coverage before retraining.
- `inspect_oracle_backend.py`: Inspect backend selection and discovered power-plan mappings.

## Feature Schema (`os_native_v2`)

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

Each `record.py` run now writes a `session_id` column, so training can validate with session-aware splits.

## Run

1. Record data per workload label:

```bash
python record.py idle --interval 0.25 --duration 180
python record.py light --interval 0.25 --duration 180
python record.py heavy --interval 0.25 --duration 240
```

Or run one collection plan:

```bash
python collect_dataset.py --sessions idle:180,light:180,heavy:240 --interval 0.25 --cycles 2 --reset
```

Data quality tip:
- For each label, collect examples with power saver ON and OFF so the model learns workload intent instead of power-plan side effects.

2. Train model:

```bash
python train_model.py
```

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

Ollama control GUI (start/stop server, prompt panel, live monitoring):

```powershell
.\run_ollama_control_gui.ps1
```

The GUI uses a tabbed layout to keep the live dashboard, runtime controls, configuration, and event log separated.
That keeps the log available without making the whole app feel like a console window.

The dashboard is organized around:
- recommendation + readiness
- a compact snapshot row (target, applied, gate, last action)
- actuation status (what backend/profile action happened)
- session outcomes (auto/manual/failed actions + dominant blocks)
- telemetry and inference detail

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
$env:PERFANALYZE_POWERCFG_SILENT="SCHEME_MIN"
$env:PERFANALYZE_POWERCFG_BALANCED="SCHEME_BALANCED"
$env:PERFANALYZE_POWERCFG_PERFORMANCE="SCHEME_MAX"

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
- The GUI can now load/save a different JSON config path when you want per-machine or per-experiment settings.
- Saving to a new nested config path will create parent folders automatically.
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
