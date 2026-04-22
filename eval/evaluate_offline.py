"""
eval/evaluate_offline.py
Offline evaluation: compare Non-DL and DL classification performance on the test set

Usage:
  python eval/evaluate_offline.py

Outputs:
  eval/results/confusion_nondl.png
  eval/results/confusion_dl.png
  eval/results/f1_comparison.png
  eval/results/offline_report.txt
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR   = os.path.join(ROOT, "data")
MODEL_DIR  = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "eval", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

GESTURES     = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "fist_open"]
LABELS_SHORT = ["L←", "R→", "U↑", "D↓", "Fist"]


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────

def load_test_data():
    X_test = np.load(os.path.join(DATA_DIR, "keypoints_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "labels_test.npy"))
    return X_test, y_test


# ─────────────────────────────────────────────
# Run inference
# ─────────────────────────────────────────────

def predict_nondl(X_test):
    from agent.planning_nondl import NonDLPlanner
    planner = NonDLPlanner()
    preds, confs = [], []
    for seq in X_test:
        g, c = planner.predict(seq)
        preds.append(GESTURES.index(g))
        confs.append(c)
    return np.array(preds), np.array(confs)


def predict_dl(X_test):
    from agent.planning_dl import DLPlanner
    planner = DLPlanner()
    preds, confs = [], []
    for seq in X_test:
        g, c = planner.predict(seq)
        preds.append(GESTURES.index(g))
        confs.append(c)
    return np.array(preds), np.array(confs)


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title: str, save_path: str):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f",
        xticklabels=LABELS_SHORT, yticklabels=LABELS_SHORT,
        cmap="Blues", ax=ax, vmin=0, vmax=1,
        annot_kws={"size": 12},
    )
    # Overlay raw counts
    for i in range(len(GESTURES)):
        for j in range(len(GESTURES)):
            ax.text(j + 0.5, i + 0.72, f"({cm[i,j]})",
                    ha="center", va="center", fontsize=8, color="gray")

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_f1_comparison(y_test, preds_nondl, preds_dl, save_path: str):
    f1_nondl = f1_score(y_test, preds_nondl, average=None)
    f1_dl    = f1_score(y_test, preds_dl,    average=None)

    x     = np.arange(len(GESTURES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(x - width/2, f1_nondl, width, label="Non-DL (RF)",  color="#4C8BF5", alpha=0.85)
    bars2 = ax.bar(x + width/2, f1_dl,    width, label="DL (LSTM)",    color="#FF6B35", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS_SHORT, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Per-class F1 Score: Non-DL vs DL", fontsize=13)
    ax.legend(fontsize=10)
    ax.axhline(y=0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  Offline Evaluation: Non-DL vs DL")
    print("=" * 55)

    X_test, y_test = load_test_data()
    print(f"  Test set size: {len(X_test)}")

    # Non-DL
    print("\n  [Non-DL] Running inference...")
    preds_nondl, _ = predict_nondl(X_test)
    acc_nondl = accuracy_score(y_test, preds_nondl)
    print(f"  Non-DL accuracy: {acc_nondl:.4f}")

    # DL
    print("\n  [DL] Running inference...")
    preds_dl, _ = predict_dl(X_test)
    acc_dl = accuracy_score(y_test, preds_dl)
    print(f"  DL accuracy: {acc_dl:.4f}")

    # Confusion matrices
    print("\n  Generating confusion matrices...")
    plot_confusion_matrix(
        y_test, preds_nondl,
        f"Non-DL (Random Forest)  Acc={acc_nondl:.3f}",
        os.path.join(RESULT_DIR, "confusion_nondl.png"),
    )
    plot_confusion_matrix(
        y_test, preds_dl,
        f"DL (LSTM)  Acc={acc_dl:.3f}",
        os.path.join(RESULT_DIR, "confusion_dl.png"),
    )

    # F1 comparison bar chart
    plot_f1_comparison(
        y_test, preds_nondl, preds_dl,
        os.path.join(RESULT_DIR, "f1_comparison.png"),
    )

    # Text report
    report_path = os.path.join(RESULT_DIR, "offline_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 55 + "\n")
        f.write("Non-DL (Random Forest) Classification Report\n")
        f.write("=" * 55 + "\n")
        f.write(classification_report(y_test, preds_nondl, target_names=GESTURES))
        f.write("\n\n" + "=" * 55 + "\n")
        f.write("DL (LSTM) Classification Report\n")
        f.write("=" * 55 + "\n")
        f.write(classification_report(y_test, preds_dl, target_names=GESTURES))
    print(f"  Saved: {report_path}")

    print("\n  Offline evaluation complete.")
    print(f"  Results directory: {RESULT_DIR}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
