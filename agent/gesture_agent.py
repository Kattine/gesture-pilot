"""
agent/gesture_agent.py
Gesture Control Agent — main entry point

Usage:
  python agent/gesture_agent.py                  # default: DL mode, no GUI
  python agent/gesture_agent.py --mode nondl     # Non-DL random-forest mode
  python agent/gesture_agent.py --tray           # with macOS system-tray icon (rumps blocks main thread)
  python agent/gesture_agent.py --debug          # show camera debug window

Background (demo recommended):
  nohup python agent/gesture_agent.py --tray > agent.log 2>&1 &
  tail -f agent.log

Stop:
  kill $(cat agent.pid)   or Ctrl+C
"""

import os
import sys
import time
import signal
import logging
import argparse
import threading
import cv2

# Add project root to Python path (allows running from any directory)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.perception     import Perception
from agent.planning_nondl import NonDLPlanner
from agent.planning_dl    import DLPlanner
from agent.control        import execute_with_guard, stop_monitoring

# ─────────────────────────────────────────────
# Logging configuration
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Main Agent class
# ─────────────────────────────────────────────

class GestureAgent:

    def __init__(self, mode: str = "dl", show_debug: bool = False):
        self.mode             = mode           # "dl" | "nondl"
        self.show_debug       = show_debug
        self.active           = True           # can be paused by tray icon
        self.last_gesture     = "—"
        self.last_confidence  = 0.0
        self._stop_event      = threading.Event()

        logger.info(f"[Agent] Initializing, mode: {mode.upper()}")

        # Perception layer
        self.perception = Perception()
        logger.info("[Agent] Perception layer started")

        # Planning layer
        self._planner_nondl = None
        self._planner_dl    = None
        self._load_planner(mode)

    def _load_planner(self, mode: str):
        if mode == "nondl" and self._planner_nondl is None:
            self._planner_nondl = NonDLPlanner()
        elif mode == "dl" and self._planner_dl is None:
            self._planner_dl = DLPlanner()
        self.mode = mode

    def set_mode(self, mode: str):
        """Switch DL / Non-DL mode at runtime (called by tray menu)."""
        logger.info(f"[Agent] Switching mode: {self.mode} → {mode}")
        self._load_planner(mode)

    def _predict(self, seq):
        if self.mode == "dl":
            return self._planner_dl.predict(seq)
        else:
            return self._planner_nondl.predict(seq)

    def run(self):
        """Main loop: perception → planning → control"""
        logger.info("[Agent] Main loop started. Press Ctrl+C to exit.")

        # Write PID file for easy process management
        with open("agent.pid", "w") as f:
            f.write(str(os.getpid()))

        try:
            while not self._stop_event.is_set():
                # Pause check
                if not self.active:
                    time.sleep(0.1)
                    continue

                # ── Perception ────────────────────────
                seq = self.perception.get_sequence()

                # Debug window (optional)
                if self.show_debug:
                    frame = self.perception.get_debug_frame()
                    if frame is not None:
                        if self.last_gesture != "—":
                            cv2.putText(
                                frame,
                                f"{self.last_gesture}  {self.last_confidence:.2f}",
                                (10, frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 200), 2,
                            )
                        cv2.imshow("Gesture Agent Debug", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                if seq is None:
                    time.sleep(0.01)
                    continue

                # ── Planning ──────────────────────────
                t0 = time.perf_counter()
                gesture, confidence = self._predict(seq)
                latency_ms = (time.perf_counter() - t0) * 1000

                logger.info(
                    f"[Predict] {gesture:<15} conf={confidence:.3f}  "
                    f"latency={latency_ms:.1f}ms  mode={self.mode}"
                )
                self.last_gesture    = gesture
                self.last_confidence = confidence

                # ── Control ───────────────────────────
                executed = execute_with_guard(gesture, confidence)
                if executed:
                    logger.info(f"[Control] ✅ Executed: {gesture}")

        except KeyboardInterrupt:
            logger.info("[Agent] Ctrl+C received, shutting down...")
        finally:
            self._cleanup()

    def _cleanup(self):
        self.perception.release()
        stop_monitoring()
        cv2.destroyAllWindows()
        if os.path.exists("agent.pid"):
            os.remove("agent.pid")
        logger.info("[Agent] Exited cleanly.")

    def stop(self):
        self._stop_event.set()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gesture Control Agent")
    parser.add_argument("--mode",  default="dl",
                        choices=["dl", "nondl"],
                        help="Planning layer mode (default: dl)")
    parser.add_argument("--tray",  action="store_true",
                        help="Enable macOS system-tray icon")
    parser.add_argument("--debug", action="store_true",
                        help="Show camera debug window")
    args = parser.parse_args()

    agent = GestureAgent(mode=args.mode, show_debug=args.debug)

    # Signal handling
    signal.signal(signal.SIGTERM, lambda *_: agent.stop())

    if args.tray:
        # Tray runs on main thread (rumps requirement); agent loop runs in worker thread
        from agent.tray_icon import TrayApp
        agent_thread = threading.Thread(target=agent.run, daemon=True)
        agent_thread.start()
        tray = TrayApp(agent_ref=agent)
        tray.run()   # blocks main thread
    else:
        agent.run()


if __name__ == "__main__":
    main()
