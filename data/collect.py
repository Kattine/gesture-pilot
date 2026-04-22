"""
data/collect.py
Guided gesture keypoint data collection script

Usage:
  python data/collect.py                        # collect all 5 gesture classes
  python data/collect.py --gesture swipe_left   # collect one class only
  python data/collect.py --samples 200          # set samples per class (default 200)

On-screen guidance is shown in both the terminal and the camera preview window.
Already-collected samples are automatically skipped (checkpoint resume supported).
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
GESTURES    = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "fist_open"]
NUM_SAMPLES = 200
SEQ_LEN     = 30
COUNTDOWN_SEC = 3
DATA_DIR    = os.path.join(os.path.dirname(__file__), "raw")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print(f"Model file not found: {MODEL_PATH}")
    print("Please run: python scripts/download_models.py")
    sys.exit(1)

# MediaPipe Hand Landmarker — IMAGE mode for collection (independent per-frame inference)
_base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
_opts = mp_vision.HandLandmarkerOptions(
    base_options=_base_opts,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = mp_vision.HandLandmarker.create_from_options(_opts)


# ─────────────────────────────────────────────
# Per-gesture collection tips
# ─────────────────────────────────────────────
GESTURE_GUIDE = {
    "swipe_left": {
        "label": "Swipe hand LEFT",
        "tips": [
            "Extend your dominant hand, palm facing the camera (fingers naturally spread)",
            "Sweep horizontally left across the frame — wrist naturally twists slightly, embrace it",
            "Full motion in 0.5–1 second",
            "Amplitude: wrist moves ~20–30 cm horizontally",
            "Avoid: vertical bounce, deliberate wrist rotation",
        ],
    },
    "swipe_right": {
        "label": "Swipe hand RIGHT",
        "tips": [
            "Mirror of swipe_left: palm facing camera, sweep right",
            "Full motion in 0.5–1 second",
            "Match amplitude of swipe_left (~20–30 cm)",
            "Avoid: wrist rotation, starting near the edge of frame",
        ],
    },
    "swipe_up": {
        "label": "Swipe hand UP",
        "tips": [
            "Palm facing camera, lift hand from chest height to eye level",
            "Strictly vertical, amplitude ~15–20 cm",
            "Full motion in 0.5–1 second",
            "Avoid: drifting left/right, wrist rotation",
        ],
    },
    "swipe_down": {
        "label": "Swipe hand DOWN",
        "tips": [
            "Mirror of swipe_up: from eye level down to chest",
            "Strictly vertical, amplitude ~15–20 cm",
            "Full motion in 0.5–1 second",
            "Avoid: wrist rotation, forward/backward movement",
        ],
    },
    "fist_open": {
        "label": "Fist → Open hand",
        "tips": [
            "Start with a natural fist (fingers curled, fist facing camera)",
            "Then quickly open hand (fingers fully extended, palm facing camera)",
            "Fist ~0.3s → open ~0.3s, total ~0.6–1 second",
            "Keep hand centred in frame",
            "Avoid: only opening without fisting first, moving too slowly",
        ],
    },
}


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────

def extract_keypoints(frame_bgr: np.ndarray):
    """
    Extract 21 keypoints from a BGR frame.
    Returns (kp_array, landmarks) or (None, None) if no hand detected.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if result.hand_landmarks:
        kp = []
        for lm in result.hand_landmarks[0]:
            kp.extend([lm.x, lm.y, lm.z])
        return np.array(kp, dtype=np.float32), result.hand_landmarks[0]
    return None, None


HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_hand(frame: np.ndarray, landmarks) -> np.ndarray:
    """Draw skeleton connections using pure OpenCV (no mediapipe.framework dependency)."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 220, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)
    return frame


def draw_overlay(frame: np.ndarray, text_lines: list[str],
                 color=(255, 255, 255), start_y=40) -> np.ndarray:
    """Render multiple text lines onto a frame with a dark outline for readability."""
    for i, line in enumerate(text_lines):
        cv2.putText(frame, line, (20, start_y + i * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, line, (20, start_y + i * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame


def print_guide(gesture: str):
    """Print gesture tips to the terminal."""
    guide = GESTURE_GUIDE[gesture]
    print("\n" + "=" * 56)
    print(f"  Gesture: {gesture}  ({guide['label']})")
    print("=" * 56)
    print("  Tips:")
    for tip in guide["tips"]:
        print(f"    • {tip}")
    print("=" * 56 + "\n")


def countdown(cap, gesture: str, idx: int, total: int):
    """Show a 3-2-1 countdown in the terminal and camera preview window."""
    guide = GESTURE_GUIDE[gesture]
    print(f"  [{idx+1}/{total}] Get ready: {gesture} ({guide['label']}) — {COUNTDOWN_SEC}s countdown")

    deadline = time.time() + COUNTDOWN_SEC
    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        remaining = int(deadline - time.time()) + 1
        lines = [
            f"Gesture: {gesture}  ({guide['label']})",
            f"Sample {idx+1}/{total}",
            f"Get ready...  {remaining}",
        ]
        frame = draw_overlay(frame, lines, color=(0, 220, 255))
        cv2.imshow("Gesture Collector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[Collection interrupted by user]")
            sys.exit(0)


def record_sequence(cap, gesture: str, idx: int, total: int) -> np.ndarray | None:
    """
    Record one 30-frame keypoint sequence.
    Returns shape (30, 63) ndarray, or None if hand was not detected.
    """
    sequence = []
    guide = GESTURE_GUIDE[gesture]

    while len(sequence) < SEQ_LEN:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        kp, landmarks = extract_keypoints(frame)

        if kp is not None:
            sequence.append(kp)
            draw_hand(frame, landmarks)

        # Progress bar
        progress = int(len(sequence) / SEQ_LEN * frame.shape[1])
        cv2.rectangle(frame, (0, frame.shape[0] - 12),
                      (progress, frame.shape[0]), (0, 255, 100), -1)

        status = "Recording..." if kp is not None else "No hand detected"
        lines = [
            f"{gesture}  ({guide['label']})",
            f"Sample {idx+1}/{total}  |  Frames: {len(sequence)}/{SEQ_LEN}",
            status,
        ]
        color = (0, 255, 100) if kp is not None else (0, 100, 255)
        draw_overlay(frame, lines, color=color)
        cv2.imshow("Gesture Collector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[Collection interrupted by user]")
            sys.exit(0)

    if len(sequence) == SEQ_LEN:
        return np.array(sequence, dtype=np.float32)
    return None


# ─────────────────────────────────────────────
# Main collection loop
# ─────────────────────────────────────────────

def collect_gesture(cap, gesture: str, num_samples: int):
    out_dir = os.path.join(DATA_DIR, gesture)
    os.makedirs(out_dir, exist_ok=True)

    # Count existing samples and determine start index (checkpoint resume)
    existing = [f for f in os.listdir(out_dir) if f.endswith(".npy")]
    start_idx = len(existing)
    if start_idx >= num_samples:
        print(f"  {gesture}: already has {start_idx} samples — skipping")
        return

    print_guide(gesture)
    input(f"  Press Enter to start collecting {gesture} (have {start_idx}/{num_samples})...")

    failed = 0
    idx = start_idx
    while idx < num_samples:
        countdown(cap, gesture, idx, num_samples)
        seq = record_sequence(cap, gesture, idx, num_samples)

        if seq is not None:
            save_path = os.path.join(out_dir, f"{idx:04d}.npy")
            np.save(save_path, seq)
            print(f"  Saved {save_path}")
            idx += 1
            failed = 0
        else:
            failed += 1
            print(f"  No hand detected — retrying (consecutive failures: {failed})")
            if failed >= 5:
                print("  5 consecutive failures. Check camera and lighting.")
                input("  Press Enter to continue, or Ctrl+C to quit...")
                failed = 0

    print(f"\n  {gesture} collection complete: {num_samples} samples\n")


def main():
    parser = argparse.ArgumentParser(description="GesturePilot — Data Collection Tool")
    parser.add_argument("--gesture", type=str, default=None,
                        choices=GESTURES, help="Collect one gesture class only")
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES,
                        help=f"Samples per class (default {NUM_SAMPLES})")
    args = parser.parse_args()

    target_gestures = [args.gesture] if args.gesture else GESTURES

    print("\n" + "=" * 56)
    print("  GesturePilot — Data Collection")
    print("=" * 56)
    print(f"  Gestures:      {target_gestures}")
    print(f"  Samples/class: {args.samples}")
    print(f"  Sequence len:  {SEQ_LEN} frames")
    print(f"  Save directory: {DATA_DIR}")
    print("\n  Controls:")
    print("    • Keep your hand in the centre of the frame")
    print("    • Start the gesture when the countdown finishes")
    print("    • Press Q at any time to quit (collected samples are saved)")
    print("=" * 56)
    input("\n  Press Enter to open camera and begin...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  Cannot open camera. Check permissions:")
        print("  System Settings → Privacy → Camera → enable Terminal")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    try:
        for gesture in target_gestures:
            collect_gesture(cap, gesture, args.samples)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    print("\n" + "=" * 56)
    print("  Collection complete!")
    print(f"  Data saved to: {DATA_DIR}")
    print("  Next step: python data/preprocess.py")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()
