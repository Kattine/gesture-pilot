"""
scripts/check_env.py
Environment verification script — run before collecting data to confirm all dependencies and permissions

Usage:
  python scripts/check_env.py

Checks:
  Python version, all dependency imports, PyTorch MPS availability,
  camera access, MediaPipe keypoint extraction, osascript permission,
  pynput Accessibility permission (safe test, no key injection).
"""

import sys
import subprocess

PASS = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠️ "


def check(label: str, fn):
    try:
        result = fn()
        if result is True or result is None:
            print(f"{PASS} {label}")
        else:
            print(f"{PASS} {label}: {result}")
    except Exception as e:
        print(f"{FAIL} {label}: {e}")


def check_python():
    v = sys.version_info
    assert v >= (3, 10), f"Python 3.10+ required, found {v.major}.{v.minor}"
    return f"{v.major}.{v.minor}.{v.micro}"

def check_cv2():
    import cv2; return cv2.__version__

def check_mediapipe():
    import mediapipe as mp; return mp.__version__

def check_sklearn():
    import sklearn; return sklearn.__version__

def check_numpy():
    import numpy as np; return np.__version__

def check_torch():
    import torch
    mps = torch.backends.mps.is_available()
    return f"v{torch.__version__}  MPS={'✓' if mps else '✗'}"

def check_pynput():
    from pynput.keyboard import Key, Controller; return True

def check_joblib():
    import joblib; return joblib.__version__

def check_rumps():
    import rumps; return True

def check_camera():
    import cv2
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), (
        "Camera cannot be opened.\n"
        "    System Settings → Privacy & Security → Camera → enable Terminal"
    )
    ret, frame = cap.read()
    cap.release()
    assert ret and frame is not None, "Camera opened but failed to read a frame"
    h, w = frame.shape[:2]
    return f"{w}×{h}"

def check_mediapipe_hands():
    import os, cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model file not found. Please run: python scripts/download_models.py"
        )

    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Camera not available"
    detected = False
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if result.hand_landmarks:
            detected = True
            break
    cap.release()
    landmarker.close()
    return "Hand landmarks detected" if detected else "(No hand detected — place hand in front of camera and retry)"

def check_osascript():
    result = subprocess.run(
        ["osascript", "-e",
         'tell application "System Events" to get name of first process '
         'whose frontmost is true'],
        capture_output=True, text=True, timeout=2,
    )
    app = result.stdout.strip()
    assert app, (
        "osascript returned no response or insufficient permissions.\n"
        "    System Settings → Privacy & Security → Accessibility → enable Terminal"
    )
    return f"Frontmost app: {app}"

def check_pynput_safe():
    from pynput.keyboard import Controller
    ctrl = Controller()
    assert ctrl is not None
    return "Instantiated successfully (Accessibility permission verified on first key injection)"


def main():
    print("\n" + "=" * 52)
    print("  GesturePilot — Environment Check")
    print("=" * 52)

    print("\n[Python]")
    check("Python version ≥ 3.10", check_python)

    print("\n[Dependencies]")
    check("opencv-python",    check_cv2)
    check("mediapipe",        check_mediapipe)
    check("scikit-learn",     check_sklearn)
    check("numpy",            check_numpy)
    check("torch + MPS",      check_torch)
    check("pynput",           check_pynput)
    check("joblib",           check_joblib)
    check("rumps",            check_rumps)

    print("\n[Hardware & Perception]")
    check("Camera open",              check_camera)
    check("MediaPipe hand detection", check_mediapipe_hands)

    print("\n[macOS Permissions]")
    check("osascript (Accessibility)", check_osascript)
    check("pynput instantiation",      check_pynput_safe)

    print("\n" + "=" * 52)
    print("  If all items show ✅, start collecting data:")
    print("    python data/collect.py")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    main()
