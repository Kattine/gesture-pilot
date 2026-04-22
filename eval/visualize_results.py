"""
eval/visualize_results.py
Generate all evaluation charts (for 10-minute video screenshots)

Usage:
  python eval/visualize_results.py

Prerequisites:
  - eval/evaluate_offline.py has been run (generates confusion matrices + F1 chart)
  - eval/evaluate_online.py  has been run (generates latency + accuracy data)

Additional outputs:
  eval/results/latency_boxplot.png     # latency distribution boxplot (DL vs Non-DL)
  eval/results/robustness_table.png    # robustness test results visualisation
  eval/results/training_curve.png      # DL training curve (Loss + Val Acc)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_DIR  = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "eval", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

GESTURES     = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "fist_open"]
LABELS_SHORT = ["L←", "R→", "U↑", "D↓", "Fist"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────
# Latency boxplot
# ─────────────────────────────────────────────

def plot_latency_boxplot():
    paths = {
        "Non-DL": os.path.join(RESULT_DIR, "online_results_nondl.npy"),
        "DL":     os.path.join(RESULT_DIR, "online_results_dl.npy"),
    }
    data = {}
    for label, path in paths.items():
        if not os.path.exists(path):
            print(f"  Skipping {label}: file not found at {path}")
            continue
        res = np.load(path, allow_pickle=True).item()
        lats = []
        for g in GESTURES:
            lats.extend(res[g]["latencies"])
        data[label] = lats

    if not data:
        print("  Latency boxplot: no data found, skipping")
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#4C8BF5", "#FF6B35"]
    bp = ax.boxplot(
        list(data.values()),
        tick_labels=list(data.keys()),
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("End-to-End Latency (ms)", fontsize=11)
    ax.set_title("Inference Latency: Non-DL vs DL", fontsize=13)
    ax.axhline(y=200, color="red", linestyle="--", linewidth=1,
               alpha=0.7, label="200ms target")
    ax.legend(fontsize=9)

    save_path = os.path.join(RESULT_DIR, "latency_boxplot.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# Robustness test chart
# ─────────────────────────────────────────────

def plot_robustness_table():
    """
    Load robustness results from eval/results/robustness_results.npy if available,
    otherwise fall back to the hardcoded placeholder values below.

    Run eval/robustness_eval.py first to generate real data.
    Format: {condition: {mode: accuracy}}
    """
    npy_path = os.path.join(RESULT_DIR, "robustness_results.npy")
    if os.path.exists(npy_path):
        robustness_data = np.load(npy_path, allow_pickle=True).item()
        print("  Robustness: loaded from robustness_results.npy")
    else:
        # ⚠️ Replace with your actual test results if not using robustness_eval.py
        robustness_data = {
            "Normal\n(baseline)":    {"Non-DL": 0.88, "DL": 0.93},
            "Low Light":             {"Non-DL": 0.82, "DL": 0.87},
            "Complex\nBackground":   {"Non-DL": 0.80, "DL": 0.85},
            "Different\nUser":       {"Non-DL": 0.75, "DL": 0.82},
        }
        print("  Robustness: using placeholder values (run robustness_eval.py for real data)")

    conditions = list(robustness_data.keys())

    # Detect which modes are present (supports single-mode runs)
    sample    = robustness_data[conditions[0]]
    has_nondl = "Non-DL" in sample
    has_dl    = "DL"     in sample

    nondl_accs = [robustness_data[c].get("Non-DL", 0) for c in conditions] if has_nondl else []
    dl_accs    = [robustness_data[c].get("DL",     0) for c in conditions] if has_dl    else []

    x     = np.arange(len(conditions))
    width = 0.35 if (has_nondl and has_dl) else 0.5
    fig, ax = plt.subplots(figsize=(8, 4))

    if has_nondl and has_dl:
        ax.bar(x - width/2, nondl_accs, width, label="Non-DL (RF)", color="#4C8BF5", alpha=0.85)
        ax.bar(x + width/2, dl_accs,    width, label="DL (LSTM)",   color="#FF6B35", alpha=0.85)
        for i, (a, b) in enumerate(zip(nondl_accs, dl_accs)):
            ax.text(i - width/2, a + 0.01, f"{a:.0%}", ha="center", va="bottom", fontsize=9)
            ax.text(i + width/2, b + 0.01, f"{b:.0%}", ha="center", va="bottom", fontsize=9)
    elif has_dl:
        bars = ax.bar(x, dl_accs, width, label="DL (LSTM)", color="#FF6B35", alpha=0.85)
        for bar, a in zip(bars, dl_accs):
            ax.text(bar.get_x() + bar.get_width()/2, a + 0.01,
                    f"{a:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    else:
        bars = ax.bar(x, nondl_accs, width, label="Non-DL (RF)", color="#4C8BF5", alpha=0.85)
        for bar, a in zip(bars, nondl_accs):
            ax.text(bar.get_x() + bar.get_width()/2, a + 0.01,
                    f"{a:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Robustness Test: Accuracy by Condition", fontsize=13)
    ax.legend(fontsize=10)
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    save_path = os.path.join(RESULT_DIR, "robustness_table.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# DL training curve
# ─────────────────────────────────────────────

def plot_training_curve():
    log_path = os.path.join(MODEL_DIR, "training_log.npy")
    if not os.path.exists(log_path):
        print("  Training curve: training_log.npy not found, skipping")
        return

    log    = np.load(log_path, allow_pickle=True).item()
    epochs = range(1, len(log["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, log["train_loss"], color="#FF6B35", linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training Loss")
    ax1.grid(axis="y", alpha=0.3)

    ax2.plot(epochs, log["val_acc"], color="#4C8BF5", linewidth=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Accuracy"); ax2.set_title("Validation Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(y=max(log["val_acc"]), color="gray", linestyle="--", linewidth=1,
                label=f"Best: {max(log['val_acc']):.3f}")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("LSTM Training Curve", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, "training_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 50)
    print("  Generating Evaluation Charts")
    print("=" * 50)

    plot_latency_boxplot()
    plot_robustness_table()
    plot_training_curve()

    print(f"\n  All charts saved to {RESULT_DIR}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
