"""
agent/planning_dl.py
DL planning layer: GestureLSTM inference (Apple Silicon MPS accelerated)

External interface:
  planner = DLPlanner()
  gesture, confidence = planner.predict(seq)   # seq: (30, 63)
"""

import os
import numpy as np
import torch
import torch.nn as nn

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
GESTURES  = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "fist_open"]


class GestureLSTM(nn.Module):
    """Identical architecture to models/train_dl.py (shared for inference)."""
    def __init__(self, input_size=63, hidden_size=128,
                 num_layers=2, num_classes=5, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


class DLPlanner:
    def __init__(self):
        # Device selection: prefer MPS (Apple Silicon), then CUDA, then CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        model_path = os.path.join(MODEL_DIR, "lstm_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "LSTM model not found. Please run: python models/train_dl.py"
            )

        self.model = GestureLSTM(num_classes=len(GESTURES)).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        print(f"  [DL] LSTM model loaded (device: {self.device})")

    @torch.no_grad()
    def predict(self, seq: np.ndarray) -> tuple[str, float]:
        """
        Args:
            seq: (30, 63) keypoint sequence
        Returns:
            (gesture_name, confidence)  confidence ∈ [0, 1]
        """
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.model(x)
        proba  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        label_idx  = int(proba.argmax())
        confidence = float(proba[label_idx])
        return GESTURES[label_idx], confidence
