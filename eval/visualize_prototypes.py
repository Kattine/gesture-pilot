"""
eval/visualize_prototypes.py
For each gesture class, find the test sample with the highest model confidence,
then render its hand skeleton at 3 key frames (start / mid / end) plus the
wrist trajectory across all 30 frames.

This shows what motion pattern the model considers "most representative" for
each gesture — useful for diagnosing left/right confusion.

Usage:
  python eval/visualize_prototypes.py --mode dl
  python eval/visualize_prototypes.py --mode nondl
  python eval/visualize_prototypes.py --mode dl --save   # save as PNG
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "fist_open"]
LABEL_MAP = {g: i for i, g in enumerate(GESTURES)}

# MediaPipe Hand 21-keypoint skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm base
]

# Colour per finger group
FINGER_COLORS = {
    "thumb":  "#FF6B6B",
    "index":  "#4ECDC4",
    "middle": "#45B7D1",
    "ring":   "#96CEB4",
    "pinky":  "#FFEAA7",
    "palm":   "#DDA0DD",
}

CONN_COLOR_MAP = {
    (0,1):(FINGER_COLORS["thumb"]),  (1,2):(FINGER_COLORS["thumb"]),
    (2,3):(FINGER_COLORS["thumb"]),  (3,4):(FINGER_COLORS["thumb"]),
    (0,5):(FINGER_COLORS["index"]),  (5,6):(FINGER_COLORS["index"]),
    (6,7):(FINGER_COLORS["index"]),  (7,8):(FINGER_COLORS["index"]),
    (0,9):(FINGER_COLORS["middle"]), (9,10):(FINGER_COLORS["middle"]),
    (10,11):(FINGER_COLORS["middle"]),(11,12):(FINGER_COLORS["middle"]),
    (0,13):(FINGER_COLORS["ring"]),  (13,14):(FINGER_COLORS["ring"]),
    (14,15):(FINGER_COLORS["ring"]), (15,16):(FINGER_COLORS["ring"]),
    (0,17):(FINGER_COLORS["pinky"]), (17,18):(FINGER_COLORS["pinky"]),
    (18,19):(FINGER_COLORS["pinky"]),(19,20):(FINGER_COLORS["pinky"]),
    (5,9):(FINGER_COLORS["palm"]),   (9,13):(FINGER_COLORS["palm"]),
    (13,17):(FINGER_COLORS["palm"]),
}

GESTURE_EMOJI = {
    "swipe_left":  "←",
    "swipe_right": "→",
    "swipe_up":    "↑",
    "swipe_down":  "↓",
    "fist_open":   "✊→🖐",
}


def load_data(data_dir: str):
    X = np.load(os.path.join(data_dir, "keypoints_test.npy"))
    y = np.load(os.path.join(data_dir, "labels_test.npy"))
    return X, y


def get_softmax_confidences(X, y, mode: str):
    """
    Run inference on the entire test set, return (pred_labels, confidences).
    confidences[i] = probability of the predicted class for sample i.
    """
    if mode == "dl":
        import torch
        from agent.planning_dl import DLPlanner
        planner = DLPlanner()
        device = next(planner.model.parameters()).device
        tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = planner.model(tensor)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
    else:
        from agent.planning_nondl import NonDLPlanner
        import joblib
        planner = NonDLPlanner()
        feats = np.array([planner._extract_features(x) for x in X])
        probs = planner.clf.predict_proba(feats)   # (N, 5)

    pred_labels  = np.argmax(probs, axis=1)
    confidences  = np.max(probs, axis=1)
    return pred_labels, confidences, probs


def normalize_seq(seq_kp):
    """
    Auto-normalize a (30, 21, 3) sequence so that all x/y values map into
    [0.05, 0.95], regardless of whether the data uses absolute coordinates
    (MediaPipe normalized [0,1]) or relative coordinates (offsets from wrist).
    z is ignored for 2-D display.
    """
    xy = seq_kp[:, :, :2].copy()          # (30, 21, 2)
    mn = xy.min()
    mx = xy.max()
    rng = mx - mn
    if rng < 1e-6:
        rng = 1.0
    xy = (xy - mn) / rng                  # → [0, 1]
    xy = xy * 0.90 + 0.05                 # → [0.05, 0.95]
    # Flip y so hand appears upright (image coords → display coords)
    xy[:, :, 1] = 1.0 - xy[:, :, 1]
    return xy                              # (30, 21, 2)


def draw_hand(ax, frame_xy, alpha=1.0, lw=2):
    """
    Draw hand skeleton on ax.
    frame_xy: (21, 2) already normalised display coordinates.
    """
    xs = frame_xy[:, 0]
    ys = frame_xy[:, 1]

    for (a, b) in HAND_CONNECTIONS:
        color = CONN_COLOR_MAP.get((a, b), "#888888")
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]],
                color=color, lw=lw, alpha=alpha, solid_capstyle="round")

    ax.scatter(xs, ys, s=18, c="#FFFFFF", zorder=5, alpha=alpha,
               linewidths=0.5, edgecolors="#333333")
    ax.scatter(xs[0], ys[0], s=50, c="#FF4444", zorder=6, alpha=alpha)


def plot_gesture_prototype(ax_frames, ax_traj, seq, gesture_name, confidence):
    """
    ax_frames: list of 3 Axes for start / mid / end frames
    ax_traj:   1 Axes for wrist trajectory
    seq:       (30, 63) keypoint sequence
    """
    kp    = seq.reshape(30, 21, 3)         # (frames, joints, xyz)
    xy    = normalize_seq(kp)              # (30, 21, 2)  display coords

    frame_indices = [0, 14, 29]
    frame_labels  = ["Start", "Mid", "End"]
    alphas        = [0.6, 0.8, 1.0]

    for ax, fi, label, alpha in zip(ax_frames, frame_indices, frame_labels, alphas):
        ax.set_facecolor("#1A1A2E")
        draw_hand(ax, xy[fi], alpha=alpha)
        ax.set_title(label, fontsize=8, color="#AAAAAA", pad=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")

    # ── Wrist trajectory ────────────────────────────────────────────────
    # Use the wrist keypoint (index 0) across all 30 frames
    wrist_x = xy[:, 0, 0]
    wrist_y = xy[:, 0, 1]

    ax_traj.set_facecolor("#1A1A2E")

    n = len(wrist_x)
    for i in range(n - 1):
        alpha_seg = 0.15 + 0.85 * (i / n)
        ax_traj.plot(wrist_x[i:i+2], wrist_y[i:i+2],
                     color="#00D4FF", lw=2.5, alpha=alpha_seg)

    ax_traj.scatter(wrist_x[0],  wrist_y[0],  s=80,  c="#00FF88",
                    zorder=5, label="start")
    ax_traj.scatter(wrist_x[-1], wrist_y[-1], s=100, c="#FF8800",
                    zorder=5, marker="*", label="end")

    dx = wrist_x[-1] - wrist_x[0]
    dy = wrist_y[-1] - wrist_y[0]
    if abs(dx) + abs(dy) > 0.01:
        ax_traj.annotate(
            "", xy=(wrist_x[-1], wrist_y[-1]),
            xytext=(wrist_x[-1] - dx * 0.25, wrist_y[-1] - dy * 0.25),
            arrowprops=dict(arrowstyle="->", color="#FF8800", lw=2.0),
        )

    ax_traj.set_xlim(0, 1)
    ax_traj.set_ylim(0, 1)
    ax_traj.set_aspect("equal")
    ax_traj.set_title(f"Wrist trajectory\nconf={confidence:.3f}",
                      fontsize=8, color="#AAAAAA", pad=2)
    ax_traj.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="dl", choices=["dl", "nondl"])
    parser.add_argument("--save", action="store_true",
                        help="Save figure as PNG instead of showing interactively")
    args = parser.parse_args()

    data_dir = os.path.join(ROOT, "data")
    print(f"\nLoading test data from {data_dir} ...")
    X, y = load_data(data_dir)
    print(f"  Test set: {len(X)} samples")

    print(f"Running {args.mode.upper()} inference on test set ...")
    pred_labels, confidences, probs = get_softmax_confidences(X, y, args.mode)

    # ── Find the most confident CORRECT sample per class ───────────────
    prototypes = {}
    for gesture in GESTURES:
        cls_idx  = LABEL_MAP[gesture]
        # Mask: true label == gesture AND predicted correctly
        correct_mask = (y == cls_idx) & (pred_labels == cls_idx)
        if not np.any(correct_mask):
            # Fall back to highest confidence for this class regardless of correctness
            mask = (y == cls_idx)
            print(f"  ⚠️  {gesture}: no correct predictions found, using highest-conf sample")
        else:
            mask = correct_mask

        class_conf = np.where(mask, probs[:, cls_idx], -1)
        best_idx   = np.argmax(class_conf)
        prototypes[gesture] = {
            "seq":        X[best_idx],
            "confidence": probs[best_idx, cls_idx],
            "pred":       GESTURES[pred_labels[best_idx]],
        }
        print(f"  {gesture:<18} best_conf={probs[best_idx, cls_idx]:.3f}  "
              f"pred={GESTURES[pred_labels[best_idx]]}")

    # ── Plot ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12), facecolor="#0D0D1A")
    fig.suptitle(
        f"Most Confident Sample per Gesture  [{args.mode.upper()}]\n"
        "● = start   ★ = end   — trail shows wrist path",
        color="white", fontsize=13, y=0.98,
    )

    n_gestures = len(GESTURES)
    outer = gridspec.GridSpec(n_gestures, 1, hspace=0.55, figure=fig,
                              top=0.93, bottom=0.03, left=0.02, right=0.98)

    for row, gesture in enumerate(GESTURES):
        p   = prototypes[gesture]
        seq = p["seq"]
        conf = p["confidence"]

        # Each row: label | frame0 | frame14 | frame29 | trajectory
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=outer[row],
            width_ratios=[0.6, 1, 1, 1, 1.2], wspace=0.08
        )

        # Gesture label panel
        ax_label = fig.add_subplot(inner[0])
        ax_label.set_facecolor("#0D0D1A")
        ax_label.axis("off")
        ax_label.text(0.5, 0.6, GESTURE_EMOJI.get(gesture, ""),
                      ha="center", va="center", fontsize=22, color="white",
                      transform=ax_label.transAxes)
        ax_label.text(0.5, 0.2, gesture.replace("_", "\n"),
                      ha="center", va="center", fontsize=8,
                      color="#AAAAAA", transform=ax_label.transAxes)

        ax_frames = [fig.add_subplot(inner[i]) for i in range(1, 4)]
        ax_traj   = fig.add_subplot(inner[4])

        plot_gesture_prototype(ax_frames, ax_traj, seq, gesture, conf)

    if args.save:
        out_path = os.path.join(ROOT, "eval", "results",
                                f"prototypes_{args.mode}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\n  Saved → {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
