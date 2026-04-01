# controller.py
"""
Run the trained model in a loop and switch G-Helper profiles via hotkeys.

Requirements:
    pip install keyboard
    - G-Helper running
    - G-Helper hotkeys enabled for Silent/Balanced/Turbo
"""

import json
import time
from pathlib import Path

import keyboard  # global hotkeys; may require admin on Windows
import torch
import torch.nn.functional as F

from model import SystemStateNet
from monitor import get_telemetry, FEATURE_NAMES

MODEL_FILE = "model.pth"
CLASSES_FILE = "classes.json"

# Map workload label -> G-Helper hotkey combo
# Adjust keys to match your actual G-Helper setup
PROFILE_HOTKEYS = {
    "idle": "ctrl+shift+alt+f16",       # Silent
    "light": "ctrl+shift+alt+f17",      # Balanced
    "browsing": "ctrl+shift+alt+f17",   # Balanced
    "gaming": "ctrl+shift+alt+f18",     # Turbo
    "rendering": "ctrl+shift+alt+f18",  # Turbo
    "heavy": "ctrl+shift+alt+f18",      # Turbo
}

POLL_INTERVAL = 0.5      # seconds between predictions
STABILITY_WINDOW = 5     # require N consistent predictions before switching
CONFIDENCE_MIN = 0.55
CONFIDENCE_MARGIN = 0.10
MIN_SWITCH_INTERVAL = 8.0


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


def send_profile_hotkey(label: str):
    combo = PROFILE_HOTKEYS.get(label)
    if not combo:
        print(f"[WARN] No hotkey mapping for label='{label}'")
        return
    print(f"[ACTION] Switching profile mapped from '{label}' via {combo}")
    keyboard.send(combo)


def main_loop():
    model, classes = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    current_label = None
    last_candidate = None
    stable_count = 0
    last_switch_ts = 0.0

    print("Starting controller loop. Press Ctrl+C to stop.")

    try:
        while True:
            x = get_telemetry()
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # shape (1, feat_dim)

            with torch.inference_mode():
                logits = model(x_t)
                probs = F.softmax(logits, dim=1)
                topk = probs.topk(k=min(2, probs.shape[1]), dim=1)
                idx = int(topk.indices[0, 0].item())
                pred_label = classes[idx]
                confidence = float(topk.values[0, 0].item())
                runner_up = (
                    float(topk.values[0, 1].item())
                    if topk.values.shape[1] > 1
                    else 0.0
                )

            margin = confidence - runner_up
            confident = (
                confidence >= CONFIDENCE_MIN and margin >= CONFIDENCE_MARGIN
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

            can_switch = (time.time() - last_switch_ts) >= MIN_SWITCH_INTERVAL
            if (
                confident
                and stable_count >= STABILITY_WINDOW
                and pred_label != current_label
                and can_switch
            ):
                current_label = pred_label
                send_profile_hotkey(current_label)
                last_switch_ts = time.time()

            print(
                f"pred={pred_label:<10} conf={confidence:.2f} margin={margin:.2f} "
                f"stable={stable_count:<2} current={current_label} confident={confident}",
                end="\r",
                flush=True,
            )
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nController stopped.")


if __name__ == "__main__":
    main_loop()
