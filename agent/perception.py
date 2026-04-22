"""
agent/perception.py
Perception layer: OpenCV camera capture + MediaPipe keypoint extraction + 30-frame sliding window
Compatible with MediaPipe >= 0.10.13 Tasks API

External interface:
  perception = Perception()
  seq = perception.get_sequence()   # returns (30, 63) ndarray or None
  perception.get_debug_frame()      # returns annotated debug frame
  perception.release()

Prerequisite:
  Run `python scripts/download_models.py` to download models/hand_landmarker.task
"""

import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from collections import deque
import threading
import time

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "hand_landmarker.task"
)


def _landmarks_to_array(landmarks) -> np.ndarray:
    """
    Convert hand_landmarks[0] from HandLandmarkerResult to a (63,) float32 array.
    landmarks: list of NormalizedLandmark
    Applies wrist-relative normalisation: subtract wrist position, divide by max abs value.
    """
    kp = []
    for lm in landmarks:
        kp.extend([lm.x, lm.y, lm.z])
    arr = np.array(kp, dtype=np.float32).reshape(21, 3)
    wrist = arr[0].copy()
    arr -= wrist
    scale = np.max(np.abs(arr)) + 1e-6
    arr /= scale
    return arr.reshape(63)


_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def _draw_landmarks_on_frame(frame: np.ndarray, landmarks) -> np.ndarray:
    """Draw 21 keypoints and skeleton connections on frame using pure OpenCV."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 220, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)
    return frame


class Perception:
    """
    Real-time perception layer (MediaPipe Tasks API).

    Uses VIDEO running mode for per-frame synchronous inference —
    no callbacks required, simplest possible implementation.
    """

    def __init__(self, camera_id=0, seq_len=30,
                 width=640, height=480, fps=30):
        self.seq_len = seq_len
        self._buffer: deque = deque(maxlen=seq_len)
        self._debug_frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"MediaPipe model not found: {MODEL_PATH}\n"
                "Please run: python scripts/download_models.py"
            )

        # MediaPipe Hand Landmarker (VIDEO mode: per-frame synchronous inference)
        base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        self._frame_ts_ms = 0    # monotonically increasing timestamp (required by VIDEO mode)

        # Camera
        self._cap = cv2.VideoCapture(camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(
                "Cannot open camera. Please check:\n"
                "  System Settings → Privacy & Security → Camera → enable Visual Studio Code"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS,          fps)

        # Background capture thread
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._running = True
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # VIDEO mode requires a monotonically increasing timestamp
            self._frame_ts_ms += 33   # ~30 fps
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

            kp = None
            debug = frame.copy()

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                kp = _landmarks_to_array(landmarks)
                debug = _draw_landmarks_on_frame(debug, landmarks)

            with self._lock:
                if kp is not None:
                    self._buffer.append(kp)
                buf_len = len(self._buffer)
                cv2.putText(debug, f"Buffer: {buf_len}/{self.seq_len}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0) if buf_len == self.seq_len else (0, 150, 255), 2)
                self._debug_frame = debug

    def get_sequence(self) -> np.ndarray | None:
        """Return a (30, 63) keypoint sequence when the buffer is full, else None.
        Clears the buffer after returning so the next call waits for fresh frames."""
        with self._lock:
            if len(self._buffer) < self.seq_len:
                return None
            seq = np.array(self._buffer, dtype=np.float32)
            self._buffer.clear()
            return seq

    def get_debug_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._debug_frame.copy() if self._debug_frame is not None else None

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self._cap.release()
        self._landmarker.close()
