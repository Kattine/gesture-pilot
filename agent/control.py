"""
agent/control.py
Control layer: bidirectional App history navigation + pynput keyboard injection + osascript App activation

Core improvement (v2):
  Uses a history list with a current pointer instead of a simple queue.
  swipe_left / swipe_right traverse the list in opposite directions;
  rapid consecutive triggers behave stably and predictably.

  Example history: [Chrome, VSCode, Finder, WeChat]
                                               ↑ ptr (current)
  swipe_left  (go back)   : ptr-1 → Finder
  swipe_right (go forward): ptr+1 → next newer App (no-op if already at head)

External interface:
  from agent.control import execute_with_guard
  execute_with_guard("swipe_left", confidence=0.92)

Permission requirements:
  System Settings → Privacy & Security → Accessibility → enable Terminal
"""

import subprocess
import time
import threading
import logging
from pynput.keyboard import Key, Controller

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Global configuration
# ─────────────────────────────────────────────

# Per-gesture confidence thresholds.
# fist_open has precision=1.00, so a lower threshold is safe and improves recall.
CONFIDENCE_THRESHOLDS = {
    "swipe_left":  0.70,   # slightly relaxed — harder gesture to recognise
    "swipe_right": 0.75,
    "swipe_up":    0.75,
    "swipe_down":  0.75,
    "fist_open":   0.55,   # precision=1.00 — lowering threshold won't increase false positives
}
DEFAULT_CONFIDENCE_THRESHOLD = 0.75   # fallback for any unlisted gesture

COOLDOWN_MS          = 500
# ★ FIX: guard duration must be LONGER than APP_POLL_INTERVAL (0.3 s).
#   Previously 0.25 s < 0.3 s poll interval caused a race where the monitor
#   thread woke up after the flag was already cleared and recorded the
#   agent-triggered switch as a user switch, polluting the history.
APP_SWITCH_GUARD_S   = 0.6    # seconds to hold agent_switching=True after activation
APP_POLL_INTERVAL    = 0.3    # monitor polling interval (seconds)
APP_HISTORY_MAX      = 10    # reduced — keeps history concise

_keyboard = Controller()

# Process names excluded from App history
EXCLUDED_APPS = {
    "gesture_agent", "Terminal", "iTerm2", "iTerm",
    "Python", "python3", "python",
}

# ─────────────────────────────────────────────
# Bidirectional App History Manager
# ─────────────────────────────────────────────

class AppHistory:
    """
    Maintains an ordered App history list and a current pointer.

    Rules:
      - When the user switches Apps normally, the new App is appended to the
        tail and the pointer is moved to the tail.
      - Agent-initiated switches (while _agent_switching is True) do NOT
        append to history; only the pointer moves. This prevents the switch
        itself from polluting the history list.
      - When the list exceeds APP_HISTORY_MAX, the head is trimmed.
    """

    def __init__(self):
        self._history: list[str] = []
        self._ptr: int = -1
        self._lock = threading.Lock()
        self._agent_switching = False   # blocks monitor writes during agent switch

    def push(self, app: str):
        """Called by the monitor thread when the user switches Apps normally."""
        with self._lock:
            if self._agent_switching:
                return
            if self._history and self._history[-1] == app:
                return
            # Remember whether the pointer was already at the newest end
            # BEFORE appending. Only follow the new entry if it was —
            # if the user had navigated back via swipe_left, keep ptr in place.
            was_at_end = (self._ptr == len(self._history) - 1)
            self._history.append(app)
            if len(self._history) > APP_HISTORY_MAX:
                trim = len(self._history) - APP_HISTORY_MAX
                self._history = self._history[trim:]
                self._ptr = max(0, self._ptr - trim)
            elif was_at_end:
                # Normal browsing: follow the newly appended entry
                self._ptr = len(self._history) - 1
            # else: user had navigated back — leave ptr where it is
            logger.debug(f"[AppHistory] push={app}  ptr={self._ptr}  "
                        f"history={self._history}")
    
    def go_back(self) -> str | None:
        """
        swipe_left: move pointer toward the past, return target App name.
        Returns None if already at the oldest position.
        """
        with self._lock:
            if self._ptr <= 0 or len(self._history) < 2:
                logger.info("[AppHistory] go_back: already at oldest end")
                return None
            self._ptr -= 1
            target = self._history[self._ptr]
            logger.info(f"[AppHistory] go_back → {target}  ptr={self._ptr}")
            return target

    def go_forward(self) -> str | None:
        """
        swipe_right: move pointer toward the future, return target App name.
        Returns None if already at the newest position.
        """
        with self._lock:
            if self._ptr >= len(self._history) - 1:
                logger.info("[AppHistory] go_forward: already at newest end")
                return None
            self._ptr += 1
            target = self._history[self._ptr]
            logger.info(f"[AppHistory] go_forward → {target}  ptr={self._ptr}")
            return target

    def set_agent_switching(self, val: bool):
        with self._lock:
            self._agent_switching = val

    def debug_state(self) -> str:
        with self._lock:
            marked = [
                f"[{a}]" if i == self._ptr else a
                for i, a in enumerate(self._history)
            ]
            return " → ".join(marked)


_app_history = AppHistory()
_monitoring  = True
_last_trigger_time = 0.0


# ─────────────────────────────────────────────
# System query / activation
# ─────────────────────────────────────────────

def get_current_app() -> str:
    try:
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of first process '
             'whose frontmost is true'],
            capture_output=True, text=True, timeout=1,
        )
        return result.stdout.strip()
    except Exception as e:
        logger.warning(f"get_current_app failed: {e}")
        return ""


def activate_app(app_name: str) -> bool:
    if not app_name:
        return False
    try:
        # Block the monitor thread while the OS focus is transitioning.
        # ★ FIX: guard duration (APP_SWITCH_GUARD_S = 0.6 s) is now safely
        #   longer than APP_POLL_INTERVAL (0.3 s), eliminating the race
        #   condition that caused agent-triggered switches to be recorded
        #   as user switches and pollute the history.
        _app_history.set_agent_switching(True)
        subprocess.run(
            ["osascript", "-e", f'tell application "{app_name}" to activate'],
            capture_output=True, timeout=2,
        )
        time.sleep(APP_SWITCH_GUARD_S)   # wait for OS focus to settle
        _app_history.set_agent_switching(False)
        logger.info(f"[Control] Activated App: {app_name}")
        return True
    except Exception as e:
        _app_history.set_agent_switching(False)
        logger.warning(f"activate_app failed: {e}")
        return False


# ─────────────────────────────────────────────
# Background monitor thread
# ─────────────────────────────────────────────

def _monitor_app_focus():
    while _monitoring:
        current = get_current_app()
        if current and current not in EXCLUDED_APPS:
            _app_history.push(current)
        time.sleep(APP_POLL_INTERVAL)


_monitor_thread = threading.Thread(target=_monitor_app_focus, daemon=True)
_monitor_thread.start()


# ─────────────────────────────────────────────
# Gesture control actions
# ─────────────────────────────────────────────

def swipe_left():
    """swipe_left: switch to an older App in history (go back)"""
    target = _app_history.go_back()
    if target:
        logger.info(f"[swipe_left] → {target}  "
                    f"history: {_app_history.debug_state()}")
        activate_app(target)
    else:
        logger.info("[swipe_left] Already at oldest history entry, cannot go further back")


def swipe_right():
    """swipe_right: switch to a newer App in history (go forward)"""
    target = _app_history.go_forward()
    if target:
        logger.info(f"[swipe_right] → {target}  "
                    f"history: {_app_history.debug_state()}")
        activate_app(target)
    else:
        logger.info("[swipe_right] Already at newest history entry, cannot go further forward")


def swipe_up():
    """swipe_up: Page Up"""
    _keyboard.tap(Key.page_up)
    logger.info("[swipe_up] Page Up")


def swipe_down():
    """swipe_down: Page Down"""
    _keyboard.tap(Key.page_down)
    logger.info("[swipe_down] Page Down")


def fist_open():
    """fist_open: toggle fullscreen (Ctrl+Cmd+F)"""
    with _keyboard.pressed(Key.ctrl):
        with _keyboard.pressed(Key.cmd):
            _keyboard.tap('f')
    logger.info("[fist_open] Toggle fullscreen")


# ─────────────────────────────────────────────
# Unified gesture action map
# ─────────────────────────────────────────────
GESTURE_ACTIONS = {
    "swipe_left":  swipe_left,
    "swipe_right": swipe_right,
    "swipe_up":    swipe_up,
    "swipe_down":  swipe_down,
    "fist_open":   fist_open,
}


# ─────────────────────────────────────────────
# Three-layer false-trigger guard
# ─────────────────────────────────────────────

def execute_with_guard(gesture: str, confidence: float) -> bool:
    """
    Layer 1: per-gesture confidence threshold
    Layer 2: cooldown silence window
    Layer 3: action existence check
    """
    global _last_trigger_time

    # Layer 1: per-gesture confidence threshold
    threshold = CONFIDENCE_THRESHOLDS.get(gesture, DEFAULT_CONFIDENCE_THRESHOLD)
    if confidence < threshold:
        logger.debug(f"[Guard] Confidence too low: {gesture} {confidence:.3f} < {threshold}")
        return False

    # Layer 2: cooldown
    now_ms = time.time() * 1000
    if now_ms - _last_trigger_time < COOLDOWN_MS:
        logger.debug(f"[Guard] Cooldown active: {now_ms - _last_trigger_time:.0f}ms elapsed")
        return False

    # Layer 3: action existence
    action = GESTURE_ACTIONS.get(gesture)
    if action is None:
        logger.warning(f"[Guard] Unknown gesture: {gesture}")
        return False

    _last_trigger_time = now_ms
    logger.info(f"[Execute] ✅ {gesture}  conf={confidence:.3f}")
    action()
    return True


def stop_monitoring():
    global _monitoring
    _monitoring = False
