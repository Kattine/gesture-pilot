"""
eval/evaluate_online.py
Online evaluation: end-to-end latency + real-world accuracy + robustness test

Usage:
  python eval/evaluate_online.py --mode dl
  python eval/evaluate_online.py --mode nondl --trials 20

Flow:
  1. Open camera; guide the user to perform each gesture N times.
  2. Record per-trial: predicted label, confidence, end-to-end latency.
  3. Classify each trial as: correct / wrong / missed_lowconf / missed_timeout.
  4. Print latency stats + accuracy table.

Key fixes vs previous version:
  - Buffer is properly flushed during a 3-2-1 countdown before EVERY trial,
    ensuring the captured sequence contains only post-prompt frames.
  - Per-gesture confidence thresholds (same as control.py) are applied so
    the evaluation reflects true agent behaviour, not raw model output.
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
    "swipe_left":  "Swipe hand LEFT",
    "swipe_right": "Swipe hand RIGHT",
    "swipe_up":    "Swipe hand UP",
    "swipe_down":  "Swipe hand DOWN",
    "fist_open":   "Fist → Open hand",
}

# Must match CONFIDENCE_THRESHOLDS in agent/control.py
CONFIDENCE_THRESHOLDS = {
    "swipe_left":  0.70,
    "swipe_right": 0.75,
    "swipe_up":    0.75,
    "swipe_down":  0.75,
    "fist_open":   0.55,
}
DEFAULT_THRESHOLD = 0.75

# How long (seconds) to flush the perception buffer before each trial.
# Must be >= SEQ_LEN / camera_fps (30 frames / 30 fps = 1.0 s) so the
# sliding window contains only post-prompt frames when GO is shown.
FLUSH_DURATION_S = 1.5


def _flush_and_countdown(perception, gesture: str, trial: int, trials: int):
    """
    Show a 3-2-1-GO countdown while continuously draining the perception
    buffer.  After this function returns, get_sequence() will return a window
    that consists entirely of frames captured AFTER the countdown started,
    eliminating stale-frame contamination.
    """
    for count in [3, 2, 1]:
        deadline = time.time() + 1.0
        while time.time() < deadline:
            perception.get_sequence()          # drain buffer every loop tick
            frame = perception.get_debug_frame()
            if frame is not None:
                label = f"[{trial+1}/{trials}]  {GESTURE_GUIDE[gesture]}   GET READY: {count}"
                cv2.putText(frame, label, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
                cv2.imshow("Online Eval", frame)
                cv2.waitKey(1)
            time.sleep(0.04)

    # Extra flush to ensure the window is completely fresh
    extra_deadline = time.time() + (FLUSH_DURATION_S - 3.0) if FLUSH_DURATION_S > 3.0 else time.time()
    while time.time() < extra_deadline:
        perception.get_sequence()
        time.sleep(0.04)


def run_online_eval(mode: str, trials: int = 20):
    from agent.perception import Perception

    if mode == "dl":
        from agent.planning_dl import DLPlanner
        planner = DLPlanner()
    else:
        from agent.planning_nondl import NonDLPlanner
        planner = NonDLPlanner()

    perception = Perception()
    results = {
        g: {"correct": 0, "wrong": 0, "missed_lowconf": 0,
            "missed_timeout": 0, "latencies": []}
        for g in GESTURES
    }

    print(f"\n  Online evaluation started  (mode={mode.upper()}, {trials} trials per gesture)")
    print("  Each trial: 3-2-1 countdown, then perform the gesture.")
    print("  Press Q in the camera window to skip the current trial.\n")

    for gesture in GESTURES:
        print(f"\n  ─── {gesture}  ({GESTURE_GUIDE[gesture]}) ───")
        input(f"  Press Enter to start ({trials} trials)...")

        for trial in range(trials):
            # ── Flush buffer + countdown ───────────────────────────────────
            # This is the critical fix: drain stale frames so the sequence
            # captured after GO contains only frames from this trial.
            _flush_and_countdown(perception, gesture, trial, trials)

            # Show GO prompt
            frame = perception.get_debug_frame()
            if frame is not None:
                cv2.putText(frame, f"GO!  {GESTURE_GUIDE[gesture]}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 100), 2)
                cv2.imshow("Online Eval", frame)
                cv2.waitKey(1)

            # ── Wait for a fresh sequence (timeout = 5 s) ──────────────────
            timeout = time.time() + 5.0
            seq = None
            skipped = False
            while seq is None and time.time() < timeout:
                seq = perception.get_sequence()
                frame = perception.get_debug_frame()
                if frame is not None:
                    cv2.putText(frame, f"Performing...  {GESTURE_GUIDE[gesture]}",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 255, 100), 2)
                    cv2.imshow("Online Eval", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        skipped = True
                        break
                time.sleep(0.01)

            if skipped:
                print(f"    [{trial+1}/{trials}] Skipped")
                continue

            if seq is None:
                results[gesture]["missed_timeout"] += 1
                print(f"    [{trial+1}/{trials}] ✗ Timeout — no valid sequence detected")
                continue

            # ── Planning: inference + timing ───────────────────────────────
            t0 = time.perf_counter()
            pred_gesture, confidence = planner.predict(seq)
            latency_ms = (time.perf_counter() - t0) * 1000

            # ── Apply per-gesture confidence threshold ─────────────────────
            # Mirrors the behaviour of execute_with_guard in control.py so
            # the evaluation reflects true agent performance, not raw model
            # output.  Low-confidence predictions are classified as missed.
            threshold = CONFIDENCE_THRESHOLDS.get(gesture, DEFAULT_THRESHOLD)

            if confidence < threshold:
                results[gesture]["missed_lowconf"] += 1
                print(f"    [{trial+1}/{trials}] ✗ Low confidence  "
                      f"pred={pred_gesture}  conf={confidence:.3f} < {threshold}  "
                      f"latency={latency_ms:.1f}ms")
                continue

            results[gesture]["latencies"].append(latency_ms)

            if pred_gesture == gesture:
                results[gesture]["correct"] += 1
                print(f"    [{trial+1}/{trials}] ✓ Correct  "
                      f"conf={confidence:.3f}  latency={latency_ms:.1f}ms")
            else:
                results[gesture]["wrong"] += 1
                print(f"    [{trial+1}/{trials}] ✗ Wrong → {pred_gesture}  "
                      f"conf={confidence:.3f}  latency={latency_ms:.1f}ms")

    perception.release()
    cv2.destroyAllWindows()
    return results


def print_results(results: dict, mode: str):
    print("\n" + "=" * 75)
    print(f"  Online Evaluation Results   mode={mode.upper()}")
    print("=" * 75)
    print(f"  {'Gesture':<18} {'Correct':>7} {'Wrong':>7} "
          f"{'LowConf':>8} {'Timeout':>8} {'Acc':>7} {'Lat mean':>10} {'Lat P95':>9}")
    print("  " + "-" * 73)

    all_latencies = []
    for gesture in GESTURES:
        r = results[gesture]
        total = r["correct"] + r["wrong"] + r["missed_lowconf"] + r["missed_timeout"]
        acc   = r["correct"] / total if total > 0 else 0
        lats  = r["latencies"]
        lat_mean = np.mean(lats)       if lats else 0.0
        lat_p95  = np.percentile(lats, 95) if lats else 0.0
        all_latencies.extend(lats)
        print(f"  {gesture:<18} {r['correct']:>7} {r['wrong']:>7} "
              f"{r['missed_lowconf']:>8} {r['missed_timeout']:>8} "
              f"{acc:>7.1%} {lat_mean:>8.1f}ms {lat_p95:>7.1f}ms")

    total_correct = sum(r["correct"] for r in results.values())
    total_trials  = sum(
        r["correct"] + r["wrong"] + r["missed_lowconf"] + r["missed_timeout"]
        for r in results.values()
    )
    print("  " + "-" * 73)
    print(f"  {'Total':<18} {total_correct:>7}  /  {total_trials:<6}"
          f"  {total_correct/total_trials:>7.1%}"
          f"  {np.mean(all_latencies) if all_latencies else 0:>8.1f}ms")
    print("=" * 75 + "\n")

    # Save raw data
    save_path = os.path.join(RESULT_DIR, f"online_results_{mode}.npy")
    np.save(save_path, results)
    print(f"  Raw data saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Gesture Agent — Online Evaluation")
    parser.add_argument("--mode",   default="dl", choices=["dl", "nondl"])
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of trials per gesture (default: 20)")
    args = parser.parse_args()

    results = run_online_eval(args.mode, args.trials)
    print_results(results, args.mode)


if __name__ == "__main__":
    main()
