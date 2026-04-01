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

## Files

- `monitor.py`: OS-native telemetry sampler + feature engineering.
- `record.py`: Record labeled telemetry into `training_data.csv`.
- `collect_dataset.py`: Run a multi-label timed collection plan in one command.
- `model.py`: `SystemStateNet` model definition.
- `train_model.py`: Train model and export `model.pth` + `classes.json`.
- `controller.py`: Live inference loop and profile-switch hotkeys.
- `oracle_client.py`: Minimal oracle adapter (`silent|balanced|performance`) with demo fallback.
- `demo_gui.py`: GUI demo of telemetry -> prediction -> target profile -> oracle action.

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

The demo uses a minimal oracle interface:

- `set_profile(profile: str) -> any`
- `get_profile() -> str | None` (optional)

If no external `oracle` module exists yet, the GUI uses an in-memory demo stub.

Optional synthetic telemetry mode:

```bash
# PowerShell
$env:PERFANALYZE_MOCK="1"
python record.py idle --interval 0.10 --duration 120
python controller.py
```

## Current Limitations

- If you already have an old `training_data.csv` from prior telemetry formats, re-record it.
- Label names are user-defined during recording; there is no label taxonomy enforcement.
- Hotkey mapping in `controller.py` must match your local G-Helper hotkey setup.
