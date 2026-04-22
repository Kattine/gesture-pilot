"""
data/preprocess.py
Data cleaning, label encoding, and Train/Val/Test split

Usage:
  python data/preprocess.py

Outputs (saved to data/ directory):
  keypoints_train.npy  labels_train.npy
  keypoints_val.npy    labels_val.npy
  keypoints_test.npy   labels_test.npy
  label_map.npy        # label-to-integer mapping for inference decoding
"""

import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "fist_open"]
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
DATA_DIR = os.path.dirname(__file__)
SEQ_STD_THRESHOLD = 0.005   # minimum sequence std-dev (filters near-static noise)
VAL_RATIO  = 0.1
TEST_RATIO = 0.1             # final split: 8:1:1


def is_valid(seq: np.ndarray) -> bool:
    """
    Filter invalid sequences:
      - std too small → hand barely moved (likely a false detection or forgotten gesture)
      - contains NaN / Inf
    """
    if np.any(np.isnan(seq)) or np.any(np.isinf(seq)):
        return False
    if np.std(seq) < SEQ_STD_THRESHOLD:
        return False
    return True


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Wrist-relative normalisation per frame:
      - Subtract wrist (landmark 0) to remove absolute position influence
      - Divide by max absolute coordinate to normalise scale
    seq shape: (30, 63)  →  21 points × 3 dims per frame
    """
    seq = seq.copy()
    for t in range(seq.shape[0]):
        frame = seq[t].reshape(21, 3)
        wrist = frame[0].copy()
        frame -= wrist
        scale = np.max(np.abs(frame)) + 1e-6
        frame /= scale
        seq[t] = frame.reshape(63)
    return seq


def load_dataset():
    X, y = [], []
    stats = {}

    for label_idx, gesture in enumerate(GESTURES):
        gesture_dir = os.path.join(RAW_DIR, gesture)
        files = sorted(glob.glob(os.path.join(gesture_dir, "*.npy")))

        if not files:
            print(f"  Warning: {gesture} — no data files found, skipping")
            stats[gesture] = {"total": 0, "valid": 0}
            continue

        valid_count = 0
        for f in files:
            seq = np.load(f)
            if is_valid(seq):
                seq = normalize_sequence(seq)
                X.append(seq)
                y.append(label_idx)
                valid_count += 1

        stats[gesture] = {"total": len(files), "valid": valid_count}
        print(f"  {gesture:15s}: {valid_count}/{len(files)} valid")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), stats


def main():
    print("\n" + "=" * 50)
    print("  Data Preprocessing")
    print("=" * 50)

    X, y, stats = load_dataset()

    if len(X) == 0:
        print("\n  No valid data found. Please run: python data/collect.py")
        return

    print(f"\n  Total samples: {len(X)}  shape: {X.shape}")
    print(f"  Label distribution: { {GESTURES[i]: int((y==i).sum()) for i in range(len(GESTURES))} }")

    # Train / Val / Test = 8:1:1
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=VAL_RATIO + TEST_RATIO, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    print(f"\n  Split result:")
    print(f"    Train : {len(X_train)}")
    print(f"    Val   : {len(X_val)}")
    print(f"    Test  : {len(X_test)}")

    np.save(os.path.join(DATA_DIR, "keypoints_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "keypoints_val.npy"),   X_val)
    np.save(os.path.join(DATA_DIR, "keypoints_test.npy"),  X_test)
    np.save(os.path.join(DATA_DIR, "labels_train.npy"),    y_train)
    np.save(os.path.join(DATA_DIR, "labels_val.npy"),      y_val)
    np.save(os.path.join(DATA_DIR, "labels_test.npy"),     y_test)
    np.save(os.path.join(DATA_DIR, "label_map.npy"),       np.array(GESTURES))

    print(f"\n  All files saved to {DATA_DIR}")
    print("  Next steps:")
    print("    python models/train_nondl.py   # train Random Forest")
    print("    python models/train_dl.py      # train LSTM")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
