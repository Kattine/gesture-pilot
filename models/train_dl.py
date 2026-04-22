"""
models/train_dl.py
DL planning layer: LSTM sequence model training (Apple Silicon MPS accelerated)

Usage:
  python models/train_dl.py
  python models/train_dl.py --epochs 80 --lr 0.001   # custom hyperparameters

Outputs:
  models/lstm_model.pth      # best validation weights (~5 MB)
  models/training_log.npy    # training curve data (for visualisation)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.dirname(__file__)
GESTURES  = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "fist_open"]


# ─────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────

class GestureLSTM(nn.Module):
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

    def forward(self, x):           # x: (batch, 30, 63)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])   # use last hidden state


# ─────────────────────────────────────────────
# Online data augmentation (keypoint space)
# ─────────────────────────────────────────────

def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Random augmentation applied during training only:
      - Gaussian coordinate noise (simulates hand tremor)
      - Random time-crop + interpolate back to 30 frames (simulates motion speed variation)
      - Random horizontal flip (within-class diversity; labels unchanged)

    Note: horizontal flip here is for within-class diversity only —
    it does NOT swap left/right labels. Cross-class flip augmentation
    is handled offline in data/augment.py.
    """
    batch = x.clone()

    # Gaussian noise
    batch += torch.randn_like(batch) * 0.005

    # Random horizontal flip (x-coordinate negation)
    flip_mask = torch.rand(batch.shape[0]) > 0.5
    for i in range(batch.shape[0]):
        if flip_mask[i]:
            frame = batch[i].reshape(30, 21, 3)
            frame[:, :, 0] = -frame[:, :, 0]
            batch[i] = frame.reshape(30, 63)

    return batch


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train(args):
    # Device selection: prefer MPS (Apple Silicon), then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Device: Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("  Device: CUDA GPU")
    else:
        device = torch.device("cpu")
        print("  Device: CPU (training will be slower)")

    # Load data
    try:
        X_train = np.load(os.path.join(DATA_DIR, "keypoints_train.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "labels_train.npy"))
        X_val   = np.load(os.path.join(DATA_DIR, "keypoints_val.npy"))
        y_val   = np.load(os.path.join(DATA_DIR, "labels_val.npy"))
        X_test  = np.load(os.path.join(DATA_DIR, "keypoints_test.npy"))
        y_test  = np.load(os.path.join(DATA_DIR, "labels_test.npy"))
    except FileNotFoundError:
        print("  Data files not found. Please run: python data/preprocess.py")
        sys.exit(1)

    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # Convert to tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_v  = torch.tensor(X_val,   dtype=torch.float32)
    y_v  = torch.tensor(y_val,   dtype=torch.long)
    X_te = torch.tensor(X_test,  dtype=torch.float32)
    y_te = torch.tensor(y_test,  dtype=torch.long)

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds   = TensorDataset(X_v,  y_v)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    # Model
    model = GestureLSTM(num_classes=len(GESTURES)).to(device)
    print(f"\n  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class-weighted loss to handle sample imbalance
    class_counts = np.bincount(y_train, minlength=len(GESTURES))
    weights = 1.0 / (class_counts + 1e-6)
    weights = torch.tensor(weights / weights.sum() * len(GESTURES),
                           dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    log = {"train_loss": [], "val_acc": []}

    print(f"\n  Training (epochs={args.epochs}, lr={args.lr})\n")
    print(f"  {'Epoch':>6}  {'Train Loss':>10}  {'Val Acc':>8}")
    print("  " + "-" * 30)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = augment_batch(xb).to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(1)
                correct += (preds == yb).sum().item()
        val_acc  = correct / len(X_val)
        avg_loss = total_loss / len(train_loader)
        log["train_loss"].append(avg_loss)
        log["val_acc"].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>6}  {avg_loss:>10.4f}  {val_acc:>8.4f}"
                  + (" ← best" if val_acc > best_val_acc else ""))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "lstm_model.pth"))

    print(f"\n  Best Val accuracy: {best_val_acc:.4f}")

    # Test set evaluation
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "lstm_model.pth"), weights_only=True,
                   map_location=device)
    )
    model.eval()
    with torch.no_grad():
        preds = model(X_te.to(device)).argmax(1).cpu().numpy()
    test_acc = (preds == y_test).mean()
    print(f"  Test accuracy: {test_acc:.4f}")

    from sklearn.metrics import classification_report
    print("\n  Classification report (Test):")
    print(classification_report(y_test, preds, target_names=GESTURES))

    # Save training log
    np.save(os.path.join(MODEL_DIR, "training_log.npy"), log)
    print("  Training log saved: models/training_log.npy")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=5e-4)
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  DL Training: GestureLSTM")
    print("=" * 50)
    train(args)
