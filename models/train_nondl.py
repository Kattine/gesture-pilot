"""
models/train_nondl.py
Non-DL planning layer: hand-crafted feature extraction + Random Forest classifier training

Usage:
  python models/train_nondl.py

Outputs:
  models/nondl_classifier.pkl    # Random Forest model (~2 MB)
  models/nondl_scaler.pkl        # feature standardiser (used at inference time)
"""

import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.dirname(__file__)
GESTURES  = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "fist_open"]


# ─────────────────────────────────────────────
# Hand-crafted feature engineering
# ─────────────────────────────────────────────

def extract_features(seq: np.ndarray) -> np.ndarray:
    """
    Extract a 35-dimensional hand-crafted feature vector from a (30, 63) keypoint sequence.

    Feature groups:
      1. Wrist trajectory direction vector           — distinguishes swipe directions
      2. Frame-to-frame wrist velocity stats         — distinguishes fast/slow motions
      3. Per-finger joint angle mean/std/change      — distinguishes fist_open vs swipe
      4. Trajectory length + endpoint displacement   — straightness of motion
      5. X/Y dominant direction ratio                — horizontal vs vertical swipe
      6. Palm openness (fingertip-to-wrist distance) — key fist_open discriminator
    """
    frames = seq.reshape(30, 21, 3)   # (T, J, 3)
    wrist  = frames[:, 0, :]          # (T, 3) wrist trajectory

    # 1. Direction vector (first → last frame wrist displacement)
    direction = wrist[-1] - wrist[0]  # (3,)

    # 2. Frame-to-frame velocity (L2 norm of consecutive wrist positions)
    vel = np.linalg.norm(np.diff(wrist, axis=0), axis=1)  # (29,)
    speed_mean = vel.mean()
    speed_std  = vel.std()
    speed_max  = vel.max()

    # 3. Joint angles (cosine of MCP-PIP-TIP angle per finger per frame)
    #    Closed fist → angle ≈ 1 (cos); open hand → angle ≈ 0
    finger_chains = [
        [1, 2, 4],    # thumb
        [5, 6, 8],    # index
        [9, 10, 12],  # middle
        [13, 14, 16], # ring
        [17, 18, 20], # pinky
    ]
    angles = []
    for t in range(30):
        for a, b, c in finger_chains:
            v1 = frames[t, a] - frames[t, b]
            v2 = frames[t, c] - frames[t, b]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7)
            angles.append(np.clip(cos_a, -1, 1))
    angles = np.array(angles).reshape(30, 5)   # (T, 5)
    angle_mean   = angles.mean(axis=0)          # (5,)
    angle_std    = angles.std(axis=0)           # (5,)
    angle_change = angles[-1] - angles[0]       # (5,) first-to-last change

    # 4. Trajectory shape features
    total_length  = vel.sum() + 1e-7
    endpoint_dist = np.linalg.norm(direction)
    straightness  = endpoint_dist / total_length   # 1.0 = perfectly straight

    # 5. X/Y dominant direction ratio (> 1 → horizontal motion dominates)
    xy_disp  = wrist[:, :2]
    xy_range = xy_disp.max(axis=0) - xy_disp.min(axis=0)
    xy_ratio = xy_range[0] / (xy_range[1] + 1e-7)

    # 6. Palm openness — fingertip-to-wrist distances over time
    fingertips = [4, 8, 12, 16, 20]
    tip_dist = np.linalg.norm(
        frames[:, fingertips, :] - frames[:, 0:1, :], axis=2
    )  # (T, 5)
    openness_mean   = tip_dist.mean(axis=0)    # (5,) average openness
    openness_change = tip_dist[-1] - tip_dist[0]  # (5,) opening/closing delta

    feature = np.concatenate([
        direction,                                       # 3
        [speed_mean, speed_std, speed_max],              # 3
        angle_mean, angle_std, angle_change,             # 15
        [total_length, endpoint_dist, straightness, xy_ratio],  # 4
        openness_mean, openness_change,                  # 10
    ])
    # Total dimensions: 3 + 3 + 15 + 4 + 10 = 35
    return feature.astype(np.float32)


def build_feature_matrix(X: np.ndarray) -> np.ndarray:
    """Batch feature extraction. X: (N, 30, 63) → (N, 35)"""
    return np.array([extract_features(seq) for seq in X])


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 50)
    print("  Non-DL Training: Random Forest")
    print("=" * 50)

    try:
        X_train = np.load(os.path.join(DATA_DIR, "keypoints_train.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "labels_train.npy"))
        X_val   = np.load(os.path.join(DATA_DIR, "keypoints_val.npy"))
        y_val   = np.load(os.path.join(DATA_DIR, "labels_val.npy"))
        X_test  = np.load(os.path.join(DATA_DIR, "keypoints_test.npy"))
        y_test  = np.load(os.path.join(DATA_DIR, "labels_test.npy"))
    except FileNotFoundError:
        print("  Data files not found. Please run: python data/preprocess.py")
        sys.exit(1)

    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print("  Extracting hand-crafted features...")

    F_train = build_feature_matrix(X_train)
    F_val   = build_feature_matrix(X_val)
    F_test  = build_feature_matrix(X_test)
    print(f"  Feature dimensionality: {F_train.shape[1]}")

    # Standardise features
    scaler  = StandardScaler()
    F_train = scaler.fit_transform(F_train)
    F_val   = scaler.transform(F_val)
    F_test  = scaler.transform(F_test)

    # Train Random Forest
    print("  Training Random Forest (100 trees, max_depth=10)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(F_train, y_train)

    # Evaluate
    val_acc  = accuracy_score(y_val,  clf.predict(F_val))
    test_acc = accuracy_score(y_test, clf.predict(F_test))
    print(f"\n  Val  accuracy: {val_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print("\n  Classification report (Test):")
    print(classification_report(y_test, clf.predict(F_test), target_names=GESTURES))

    # Feature importance (top 5)
    importances = clf.feature_importances_
    print(f"  Top-5 feature indices: {np.argsort(importances)[::-1][:5]}")

    # Save
    joblib.dump(clf,    os.path.join(MODEL_DIR, "nondl_classifier.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "nondl_scaler.pkl"))
    print(f"\n  Model saved:     models/nondl_classifier.pkl")
    print(f"  Scaler saved:    models/nondl_scaler.pkl")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
