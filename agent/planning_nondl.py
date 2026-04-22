"""
agent/planning_nondl.py
Non-DL planning layer: Random Forest gesture classifier inference

External interface:
  planner = NonDLPlanner()
  gesture, confidence = planner.predict(seq)   # seq: (30, 63)
"""

import os
import numpy as np
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
GESTURES  = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "fist_open"]


def _extract_features(seq: np.ndarray) -> np.ndarray:
    """
    Feature extraction function — must stay identical to models/train_nondl.py.
    Any divergence will silently corrupt inference results.
    """
    frames = seq.reshape(30, 21, 3)
    wrist  = frames[:, 0, :]

    direction = wrist[-1] - wrist[0]
    vel = np.linalg.norm(np.diff(wrist, axis=0), axis=1)
    speed_mean = vel.mean()
    speed_std  = vel.std()
    speed_max  = vel.max()

    finger_chains = [
        [1, 2, 4], [5, 6, 8], [9, 10, 12], [13, 14, 16], [17, 18, 20],
    ]
    angles = []
    for t in range(30):
        for a, b, c in finger_chains:
            v1 = frames[t, a] - frames[t, b]
            v2 = frames[t, c] - frames[t, b]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7)
            angles.append(np.clip(cos_a, -1, 1))
    angles = np.array(angles).reshape(30, 5)
    angle_mean   = angles.mean(axis=0)
    angle_std    = angles.std(axis=0)
    angle_change = angles[-1] - angles[0]

    total_length  = vel.sum() + 1e-7
    endpoint_dist = np.linalg.norm(direction)
    straightness  = endpoint_dist / total_length
    xy_disp = wrist[:, :2]
    xy_range = xy_disp.max(axis=0) - xy_disp.min(axis=0)
    xy_ratio = xy_range[0] / (xy_range[1] + 1e-7)

    fingertips = [4, 8, 12, 16, 20]
    tip_dist = np.linalg.norm(
        frames[:, fingertips, :] - frames[:, 0:1, :], axis=2
    )
    openness_mean   = tip_dist.mean(axis=0)
    openness_change = tip_dist[-1] - tip_dist[0]

    return np.concatenate([
        direction, [speed_mean, speed_std, speed_max],
        angle_mean, angle_std, angle_change,
        [total_length, endpoint_dist, straightness, xy_ratio],
        openness_mean, openness_change,
    ]).astype(np.float32)


class NonDLPlanner:
    def __init__(self):
        clf_path    = os.path.join(MODEL_DIR, "nondl_classifier.pkl")
        scaler_path = os.path.join(MODEL_DIR, "nondl_scaler.pkl")
        if not os.path.exists(clf_path):
            raise FileNotFoundError(
                "Random Forest model not found. Please run: python models/train_nondl.py"
            )
        self.clf    = joblib.load(clf_path)
        self.scaler = joblib.load(scaler_path)
        print("  [NonDL] Random Forest model loaded")

    def predict(self, seq: np.ndarray) -> tuple[str, float]:
        """
        Args:
            seq: (30, 63) keypoint sequence
        Returns:
            (gesture_name, confidence)  confidence ∈ [0, 1]
        """
        feat = _extract_features(seq).reshape(1, -1)
        feat = self.scaler.transform(feat)
        proba = self.clf.predict_proba(feat)[0]
        label_idx = int(proba.argmax())
        confidence = float(proba[label_idx])
        return GESTURES[label_idx], confidence
