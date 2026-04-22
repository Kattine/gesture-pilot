"""
agent/tray_icon.py
macOS system tray icon (rumps)

Usage (called from gesture_agent.py):
  from agent.tray_icon import TrayApp
  app = TrayApp(agent_ref)
  app.run()   # blocking — must run on the main thread
"""

import rumps
import logging

logger = logging.getLogger(__name__)


class TrayApp(rumps.App):
    """
    Menu-bar Agent status icon.
    ✋ = running,  🤚 = paused
    """

    def __init__(self, agent_ref=None):
        super().__init__("✋", quit_button="Quit GesturePilot")
        self.agent = agent_ref
        self.menu = [
            "Pause / Resume",
            None,                   # separator
            "Switch to Non-DL mode",
            "Switch to DL mode",
            None,
            "Show current confidence",
        ]

    @rumps.clicked("Pause / Resume")
    def toggle_active(self, _):
        if self.agent:
            self.agent.active = not self.agent.active
            self.title = "✋" if self.agent.active else "🤚"
            status = "Running" if self.agent.active else "Paused"
            rumps.notification("GesturePilot", "", f"Agent {status}")
            logger.info(f"[Tray] Agent {status}")

    @rumps.clicked("Switch to Non-DL mode")
    def switch_nondl(self, _):
        if self.agent:
            self.agent.set_mode("nondl")
            rumps.notification("GesturePilot", "", "Switched to Non-DL (Random Forest) mode")

    @rumps.clicked("Switch to DL mode")
    def switch_dl(self, _):
        if self.agent:
            self.agent.set_mode("dl")
            rumps.notification("GesturePilot", "", "Switched to DL (LSTM) mode")

    @rumps.clicked("Show current confidence")
    def show_confidence(self, _):
        if self.agent:
            info = (
                f"Mode: {self.agent.mode.upper()}\n"
                f"Last gesture: {self.agent.last_gesture}\n"
                f"Last confidence: {self.agent.last_confidence:.3f}"
            )
            rumps.notification("GesturePilot", "Current Status", info)
