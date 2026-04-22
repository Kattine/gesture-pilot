"""
eval/robustness_eval.py
Quick robustness test: 3 conditions × 5 gestures × 3 trials = 45 trials (~10 min)

Conditions tested:
  1. Normal       — normal indoor lighting, plain background (baseline)
  2. Low Light    — main lights off, screen glow only
  3. Complex BG   — bookshelf / patterned wall behind you

Usage:
  python eval/robustness_eval.py --mode dl
  python eval/robustness_eval.py --mode nondl
  python eval/robustness_eval.py --mode dl --trials 3   # default: 3 per gesture per condition

Outputs:
  eval/results/robustness_results.npy   ← loaded automatically by visualize_results.py
  eval/results/robustness_table.png     ← chart generated immediately after collection
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULT_DIR = os.path.join(ROOT, "eval", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "fist_open"]

GESTURE_GUIDE = {
    "swipe_left":  "Swipe LEFT",
    "swipe_right": "Swipe RIGHT",
    "swipe_up":    "Swipe UP",
    "swipe_down":  "Swipe DOWN",
    "fist_open":   "Fist → Open",
}

# 3 conditions — change lighting/background between conditions
CONDITIONS = [
    {
        "key":   "Normal\n(baseline)",
        "label": "NORMAL (baseline)",
        "instruction": "Normal indoor lighting, plain background. No changes needed.",
    },
    {
        "key":   "Low Light",
        "label": "LOW LIGHT",
        "instruction": "Turn off main lights. Only screen glow / ambient light.",
    },
    {
        "key":   "Complex\nBackground",
        "label": "COMPLEX BACKGROUND",
        "instruction": "Move in front of a bookshelf, window, or patterned wall.",
    },
]

# Per-gesture confidence thresholds — must match control.py
CONFIDENCE_THRESHOLDS = {
    "swipe_left":  0.70,
    "swipe_right": 0.75,
    "swipe_up":    0.75,
    "swipe_down":  0.75,
    "fist_open":   0.55,
}
DEFAULT_THRESHOLD = 0.75

# Seconds to drain the perception buffer before each trial (>= SEQ_LEN/fps = 1.0s)
FLUSH_DURATION_S = 1.5


def _flush_and_countdown(perception, gesture: str, trial: int, trials: int, condition_label: str):
    """Drain the perception buffer during 3-2-1 countdown so each trial gets fresh frames."""
    for count in [3, 2, 1]:
        deadline = time.time() + 1.0
        while time.time() < deadline:
            perception.get_sequence()
            frame = perception.get_debug_frame()
            if frame is not None:
                text = f"[{condition_label}]  [{trial+1}/{trials}]  {GESTURE_GUIDE[gesture]}   {count}"
                cv2.putText(frame, text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.imshow("Robustness Eval", frame)
                cv2.waitKey(1)
            time.sleep(0.04)


def run_condition(planner, perception, condition: dict, trials: int) -> dict:
    """Run all gesture trials for one condition. Returns per-gesture result dicts."""
    results = {g: {"correct": 0, "wrong": 0, "missed_lowconf": 0, "missed_timeout": 0}
               for g in GESTURES}

    print(f"\n  ── Condition: {condition['label']} ──")
    print(f"  Setup: {condition['instruction']}")
    input("  Press Enter when ready...")

    for gesture in GESTURES:
        print(f"\n    [{gesture}]  {GESTURE_GUIDE[gesture]}")

        for trial in range(trials):
            _flush_and_countdown(perception, gesture, trial, trials, condition["label"])

            # Show GO
            frame = perception.get_debug_frame()
            if frame is not None:
                cv2.putText(frame, f"GO!  {GESTURE_GUIDE[gesture]}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)
                cv2.imshow("Robustness Eval", frame)
                cv2.waitKey(1)

            # Wait for fresh sequence
            timeout = time.time() + 5.0
            seq = None
            while seq is None and time.time() < timeout:
                seq = perception.get_sequence()
                frame = perception.get_debug_frame()
                if frame is not None:
                    cv2.imshow("Robustness Eval", frame)
                    cv2.waitKey(1)
                time.sleep(0.01)

            if seq is None:
                results[gesture]["missed_timeout"] += 1
                print(f"      [{trial+1}/{trials}] ✗ Timeout")
                continue

            pred_gesture, confidence = planner.predict(seq)
            threshold = CONFIDENCE_THRESHOLDS.get(gesture, DEFAULT_THRESHOLD)

            if confidence < threshold:
                results[gesture]["missed_lowconf"] += 1
                print(f"      [{trial+1}/{trials}] ✗ Low conf  pred={pred_gesture}  conf={confidence:.3f}")
            elif pred_gesture == gesture:
                results[gesture]["correct"] += 1
                print(f"      [{trial+1}/{trials}] ✓ Correct   conf={confidence:.3f}")
            else:
                results[gesture]["wrong"] += 1
                print(f"      [{trial+1}/{trials}] ✗ Wrong → {pred_gesture}  conf={confidence:.3f}")

    return results


def condition_accuracy(results: dict, trials: int) -> float:
    """Overall accuracy for one condition (correct / total trials across all gestures)."""
    correct = sum(r["correct"] for r in results.values())
    total   = len(GESTURES) * trials
    return correct / total if total > 0 else 0.0


def run_robustness_eval(mode: str, trials: int):
    from agent.perception import Perception

    if mode == "dl":
        from agent.planning_dl import DLPlanner
        planner = DLPlanner()
    else:
        from agent.planning_nondl import NonDLPlanner
        planner = NonDLPlanner()

    perception = Perception()

    print(f"\n  Robustness Evaluation  mode={mode.upper()}  {trials} trials/gesture/condition")
    print(f"  Total trials: {len(CONDITIONS)} conditions × {len(GESTURES)} gestures × {trials} = "
          f"{len(CONDITIONS) * len(GESTURES) * trials}")
    print("  Estimated time: ~10 minutes\n")

    all_results = {}   # condition_key → per-gesture results

    for condition in CONDITIONS:
        results = run_condition(planner, perception, condition, trials)
        all_results[condition["key"]] = results
        acc = condition_accuracy(results, trials)
        print(f"\n  {condition['label']} overall accuracy: {acc:.1%}")

    perception.release()
    cv2.destroyAllWindows()
    return all_results


def build_robustness_data(all_results: dict, trials: int, mode: str) -> dict:
    """
    Convert raw per-gesture results into the format expected by visualize_results.py:
      {condition_key: {"Non-DL": acc, "DL": acc}}

    When running only one mode, the other is left as None and excluded from the chart.
    """
    robustness_data = {}
    mode_key = "DL" if mode == "dl" else "Non-DL"
    for cond_key, results in all_results.items():
        acc = condition_accuracy(results, trials)
        robustness_data[cond_key] = {mode_key: acc}
    return robustness_data


def plot_and_save(all_results: dict, trials: int, mode: str):
    import matplotlib.pyplot as plt

    # Build accuracy per condition
    conditions = [c["key"] for c in CONDITIONS]
    accs = [condition_accuracy(all_results[c["key"]], trials) for c in CONDITIONS]
    color = "#FF6B35" if mode == "dl" else "#4C8BF5"
    label = "DL (LSTM)" if mode == "dl" else "Non-DL (RF)"

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(conditions, accs, color=color, alpha=0.85, width=0.5)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(f"Robustness Test — {label}", fontsize=13)
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{acc:.0%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Per-gesture breakdown as a table below the chart
    col_labels = [g.replace("_", "\n") for g in GESTURES]
    row_labels  = [c["label"] for c in CONDITIONS]
    cell_data   = []
    for c in CONDITIONS:
        row = []
        for g in GESTURES:
            r = all_results[c["key"]][g]
            total = r["correct"] + r["wrong"] + r["missed_lowconf"] + r["missed_timeout"]
            acc_g = r["correct"] / total if total > 0 else 0
            row.append(f"{acc_g:.0%}")
        cell_data.append(row)

    # Add table below the bar chart
    plt.subplots_adjust(bottom=0.28)
    table = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="bottom",
        bbox=[0, -0.55, 1, 0.4],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    save_path = os.path.join(RESULT_DIR, "robustness_table.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved: {save_path}")

    # Merge with existing file instead of overwriting —
    # DL and Non-DL runs accumulate into the same file.
    npy_path = os.path.join(RESULT_DIR, "robustness_results.npy")
    new_data = build_robustness_data(all_results, trials, mode)
    if os.path.exists(npy_path):
        existing = np.load(npy_path, allow_pickle=True).item()
        for cond_key, mode_dict in new_data.items():
            if cond_key in existing:
                existing[cond_key].update(mode_dict)
            else:
                existing[cond_key] = mode_dict
        merged = existing
    else:
        merged = new_data
    np.save(npy_path, merged)
    print(f"  Raw data saved (merged): {npy_path}")


def print_summary(all_results: dict, trials: int, mode: str):
    print("\n" + "=" * 65)
    print(f"  Robustness Summary   mode={mode.upper()}")
    print("=" * 65)
    print(f"  {'Condition':<22} {'Acc':>6}  per-gesture breakdown")
    print("  " + "-" * 63)

    for condition in CONDITIONS:
        key     = condition["key"]
        results = all_results[key]
        acc     = condition_accuracy(results, trials)
        per_g   = []
        for g in GESTURES:
            r = results[g]
            total = r["correct"] + r["wrong"] + r["missed_lowconf"] + r["missed_timeout"]
            acc_g = r["correct"] / total if total > 0 else 0
            per_g.append(f"{g.split('_')[1][0].upper()}:{acc_g:.0%}")
        label_clean = condition["label"].replace("\n", " ")
        print(f"  {label_clean:<22} {acc:>6.1%}  {' '.join(per_g)}")

    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="GesturePilot — Robustness Evaluation")
    parser.add_argument("--mode",   default="dl", choices=["dl", "nondl"])
    parser.add_argument("--trials", type=int, default=3,
                        help="Trials per gesture per condition (default: 3)")
    args = parser.parse_args()

    all_results = run_robustness_eval(args.mode, args.trials)
    print_summary(all_results, args.trials, args.mode)
    plot_and_save(all_results, args.trials, args.mode)


if __name__ == "__main__":
    main()
