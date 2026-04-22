"""
eval/visualize_results.py
生成所有评估图表（用于 10 分钟视频截图）

运行方式：
  python eval/visualize_results.py

前提：
  - eval/evaluate_offline.py 已运行（生成混淆矩阵 + F1 图）
  - eval/evaluate_online.py  已运行（生成延迟 + 准确率数据）

额外生成：
  eval/results/latency_boxplot.png     # 延迟分布箱线图（DL vs Non-DL）
  eval/results/robustness_table.png    # 鲁棒性测试结果可视化
  eval/results/training_curve.png      # DL 训练曲线（Loss + Val Acc）
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

GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "fist_open"]
LABELS_SHORT = ["L←", "R→", "U↑", "D↓", "Fist"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────
# 延迟箱线图
# ─────────────────────────────────────────────

def plot_latency_boxplot():
    paths = {
        "Non-DL": os.path.join(RESULT_DIR, "online_results_nondl.npy"),
        "DL":     os.path.join(RESULT_DIR, "online_results_dl.npy"),
    }
    data = {}
    for label, path in paths.items():
        if not os.path.exists(path):
            print(f"  跳过 {label}：未找到 {path}")
            continue
        res = np.load(path, allow_pickle=True).item()
        lats = []
        for g in GESTURES:
            lats.extend(res[g]["latencies"])
        data[label] = lats

    if not data:
        print("  延迟箱线图：无数据，跳过")
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#4C8BF5", "#FF6B35"]
    bp = ax.boxplot(
        list(data.values()),
        labels=list(data.keys()),
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
    print(f"  保存：{save_path}")


# ─────────────────────────────────────────────
# 鲁棒性测试表格可视化
# ─────────────────────────────────────────────

def plot_robustness_table():
    """
    手动输入鲁棒性测试数据（运行在线测试后填写）。
    格式：{条件: {模式: 准确率}}
    """
    # ⚠️ 将此处替换为你的实际测试数据
    robustness_data = {
        "Normal\n(baseline)":    {"Non-DL": 0.88, "DL": 0.93},
        "Low Light":             {"Non-DL": 0.82, "DL": 0.87},
        "Complex\nBackground":   {"Non-DL": 0.80, "DL": 0.85},
        "Different\nUser":       {"Non-DL": 0.75, "DL": 0.82},
    }

    conditions = list(robustness_data.keys())
    nondl_accs = [robustness_data[c]["Non-DL"] for c in conditions]
    dl_accs    = [robustness_data[c]["DL"]     for c in conditions]

    x = np.arange(len(conditions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, nondl_accs, width, label="Non-DL", color="#4C8BF5", alpha=0.85)
    ax.bar(x + width/2, dl_accs,    width, label="DL",     color="#FF6B35", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Robustness Test: Accuracy by Condition", fontsize=13)
    ax.legend(fontsize=10)
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    for i, (a, b) in enumerate(zip(nondl_accs, dl_accs)):
        ax.text(i - width/2, a + 0.01, f"{a:.0%}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width/2, b + 0.01, f"{b:.0%}", ha="center", va="bottom", fontsize=9)

    save_path = os.path.join(RESULT_DIR, "robustness_table.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存：{save_path}")


# ─────────────────────────────────────────────
# DL 训练曲线
# ─────────────────────────────────────────────

def plot_training_curve():
    log_path = os.path.join(MODEL_DIR, "training_log.npy")
    if not os.path.exists(log_path):
        print("  训练曲线：未找到 training_log.npy，跳过")
        return

    log = np.load(log_path, allow_pickle=True).item()
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
    print(f"  保存：{save_path}")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 50)
    print("  生成评估图表")
    print("=" * 50)

    plot_latency_boxplot()
    plot_robustness_table()
    plot_training_curve()

    print(f"\n  ✅ 所有图表已保存到 {RESULT_DIR}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
