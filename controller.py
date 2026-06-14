# controller.py
"""
Run the trained model in a loop and apply workload-driven profiles.

Default path:
    - direct G-Helper-compatible hotkeys from config.profile_hotkeys

Optional shared backend path:
    - set PERFANALYZE_CONTROLLER_USE_ORACLE=1, or configure PERFANALYZE_ORACLE_BACKEND / PERFANALYZE_ORACLE_COMMAND
    - this lets controller use the same command/powercfg/oracle backend flow as the demo GUI and probe tools
"""

import json
import os
import time
from pathlib import Path

import keyboard  # global hotkeys; may require admin on Windows
import torch
import torch.nn.functional as F

from config import CONFIG_FILE, load_config
from model import SystemStateNet
from monitor import get_telemetry, FEATURE_NAMES
from oracle_client import OracleClient, profile_for_label
from prediction_logic import ProbabilitySmoother, summarize_probabilities
from switch_policy import describe_gate, evaluate_switch_gate

MODEL_FILE = "model.pth"
CLASSES_FILE = "classes.json"

def load_model():
    model_path = Path(MODEL_FILE)
    classes_path = Path(CLASSES_FILE)
    if not model_path.exists() or not classes_path.exists():
        raise RuntimeError(
            "Missing model artifacts. Expected both "
            f"'{MODEL_FILE}' and '{CLASSES_FILE}'. "
            "Run train_model.py after collecting training_data.csv."
        )

    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        classes = json.load(f)
    if not classes:
        raise RuntimeError(f"'{CLASSES_FILE}' is empty; retrain the model.")

    input_dim = len(FEATURE_NAMES)
    num_classes = len(classes)

    model = SystemStateNet(input_dim, num_classes)
    state = torch.load(MODEL_FILE, map_location="cpu")
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(
            "Model/feature schema mismatch. Re-record data and re-run train_model.py."
        ) from e
    model.eval()

    return model, classes


def send_profile_hotkey(profile: str, profile_hotkeys: dict[str, str]) -> bool:
    combo = profile_hotkeys.get(profile.strip().lower())
    if not combo:
        print(f"[WARN] No hotkey mapping for profile='{profile}'")
        return False
    print(f"[ACTION] Switching profile='{profile}' via hotkey '{combo}'")
    keyboard.send(combo)
    return True


def _use_oracle_backend() -> bool:
    token = os.getenv("PERFANALYZE_CONTROLLER_USE_ORACLE", "").strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if os.getenv("PERFANALYZE_ORACLE_COMMAND", "").strip():
        return True
    requested = os.getenv("PERFANALYZE_ORACLE_BACKEND", "").strip().lower()
    return requested in {"auto", "command", "oracle_command", "command_bridge", "oracle_module", "external", "powercfg", "windows_powercfg"}


def apply_profile_action(
    profile: str,
    *,
    profile_hotkeys: dict[str, str],
    oracle: OracleClient | None,
) -> tuple[bool, str]:
    if oracle is not None:
        result = oracle.set_profile(profile)  # controller is live; no dry-run path here
        return result.ok, result.message

    switched = send_profile_hotkey(profile, profile_hotkeys=profile_hotkeys)
    return switched, "direct hotkey" if switched else "direct hotkey failed"


def main_loop():
    try:
        cfg_result = load_config(CONFIG_FILE)
    except RuntimeError as e:
        print(f"[CONFIG ERROR] {e}")
        return

    cfg = cfg_result.config
    print(f"[CONFIG] Loaded: {cfg_result.source}")

    model, classes = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    oracle = OracleClient() if _use_oracle_backend() else None
    if oracle is not None:
        print(f"[BACKEND] Using oracle backend: {oracle.backend_name}")
    else:
        print("[BACKEND] Using direct hotkey switching")

    current_profile = None
    last_candidate = None
    stable_count = 0
    last_switch_ts = 0.0
    smoother = ProbabilitySmoother(cfg.switching.probability_ema_alpha)

    print("Starting controller loop. Press Ctrl+C to stop.")

    try:
        while True:
            x = get_telemetry()
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # shape (1, feat_dim)

            with torch.inference_mode():
                logits = model(x_t)
                probs = F.softmax(logits, dim=1)

            raw_summary = summarize_probabilities(probs[0], classes)
            smoothed_probs = smoother.update(probs[0])
            decision_summary = summarize_probabilities(smoothed_probs, classes)

            pred_label = decision_summary.label
            confidence = decision_summary.confidence
            runner_up = decision_summary.runner_up_confidence
            margin = decision_summary.margin
            confident = (
                confidence >= cfg.switching.confidence_min
                and margin >= cfg.switching.confidence_margin
            )

            # Hysteresis only when confidence/margin are strong enough.
            if confident:
                if pred_label == last_candidate:
                    stable_count += 1
                else:
                    last_candidate = pred_label
                    stable_count = 1
            else:
                last_candidate = None
                stable_count = 0

            target_profile = profile_for_label(
                pred_label,
                label_map=cfg.label_to_profile,
            )
            gate = evaluate_switch_gate(
                current_profile=current_profile,
                target_profile=target_profile,
                confident=confident,
                stable_count=stable_count,
                base_window=cfg.switching.stability_window,
                downshift_extra_window=cfg.switching.downshift_extra_window,
                now_ts=time.time(),
                last_switch_ts=last_switch_ts,
                min_switch_interval_sec=cfg.switching.min_switch_interval_sec,
                downshift_hold_sec=cfg.switching.downshift_hold_sec,
            )
            if gate.allow:
                switched, action_message = apply_profile_action(
                    target_profile,
                    profile_hotkeys=cfg.profile_hotkeys,
                    oracle=oracle,
                )
                if switched:
                    current_profile = target_profile
                    last_switch_ts = time.time()
                else:
                    print(f"[WARN] Failed to apply profile='{target_profile}': {action_message}")

            print(
                f"raw={raw_summary.label:<10} raw_conf={raw_summary.confidence:.2f} "
                f"decision={pred_label:<10} conf={confidence:.2f} margin={margin:.2f} "
                f"stable={stable_count:<2} target={target_profile:<11} "
                f"current={current_profile} backend={oracle.backend_name if oracle else 'hotkey'} "
                f"gate={describe_gate(gate):<28} confident={confident}",
                end="\r",
                flush=True,
            )
            time.sleep(cfg.switching.poll_interval_sec)

    except KeyboardInterrupt:
        print("\nController stopped.")


if __name__ == "__main__":
    main_loop()
