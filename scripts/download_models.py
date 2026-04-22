"""
scripts/download_models.py
Download the MediaPipe Hand Landmarker model file (required by the Tasks API)

Usage:
  python scripts/download_models.py

Output:
  models/hand_landmarker.task   (~8 MB)
"""

import os
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")


def download():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
        print(f"  Model already exists: {MODEL_PATH}  ({size_mb:.1f} MB)")
        return

    print(f"  Downloading hand_landmarker.task ...")
    print(f"  Source: {MODEL_URL}")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"\r  [{bar}] {pct:.0f}%", end="", flush=True)

    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=progress)
    print()

    size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
    print(f"  Download complete: {MODEL_PATH}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  MediaPipe Model Download")
    print("=" * 50)
    download()
    print("\n  Next step: python scripts/check_env.py")
    print("=" * 50 + "\n")
