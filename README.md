# GesturePilot

**About This Agent**
Real-time camera-based gesture recognition → macOS system-level window control

---

## Project Overview

GesturePilot is a hands-free interaction agent that maps five dynamic hand gestures captured from a built-in webcam to macOS system commands — no touch device required. It implements the full Perception → Planning → Control pipeline in two versions (Non-DL and DL) for direct comparison.

| Agent Layer | Implementation |
|-------------|---------------|
| **Perception** | OpenCV 30fps capture → MediaPipe Hand Landmarker → 21-keypoint (x,y,z) extraction |
| **Planning** | 30-frame sliding window → Random Forest (Non-DL) or LSTM (DL) classifier |
| **Control** | `pynput` keyboard injection + `osascript` App activation |

---

## Gesture Set

| Gesture | Action | macOS Implementation |
|---------|--------|---------------------|
| Swipe Left | Switch to previous App | `osascript` bidirectional history |
| Swipe Right | Switch to next App | `osascript` bidirectional history |
| Swipe Up | Page Up | `pynput Key.page_up` |
| Swipe Down | Page Down | `pynput Key.page_down` |
| Fist → Open | Toggle fullscreen | `Ctrl+Cmd+F` |

---

## Evaluation Results

### Offline (Test Set — 820 samples)

| Model | Accuracy | swipe_left F1 | swipe_right F1 | swipe_up F1 | swipe_down F1 | fist_open F1 |
|-------|----------|--------------|----------------|-------------|---------------|--------------|
| Non-DL (RF) | **82%** | 0.70 | 0.74 | 0.90 | 0.88 | **1.00** |
| DL (LSTM)   | **80%** | 0.66 | 0.73 | 0.86 | 0.84 | **0.99** |

### Online (Real-World — 20 trials × 5 gestures)

| Model | Overall Acc | swipe_left | swipe_right | swipe_up | swipe_down | fist_open |
|-------|-------------|-----------|-------------|----------|-----------|-----------|
| DL (LSTM)   | **67%** | 75% | 75% | 60% | 45% | 80% |
| Non-DL (RF) | **30%** | 50% | 0%  | 0%  | 0%  | 100% |

> Non-DL online accuracy is lower because RF confidence scores are more spread out compared to LSTM softmax probabilities — most predictions fall below the 0.75 threshold and are classified as missed (low confidence). fist_open works perfectly because its feature signature is unambiguous.

### End-to-End Latency

| Model | Mean | P95 |
|-------|------|-----|
| Non-DL (RF) | < 2ms | < 5ms |
| DL (LSTM)   | ~8ms  | ~18ms |

Both modes run well within the 200ms perceptual budget.

---

## Repository Structure

```
gesture-pilot/
├── README.md
├── requirements.txt
│
├── scripts/
│   ├── download_models.py   # download MediaPipe model (one-time)
│   └── check_env.py         # verify environment before collecting
│
├── data/
│   ├── collect.py           # guided data collection with countdown UI
│   ├── augment.py           # offline augmentation (flip/warp/noise/scale)
│   ├── preprocess.py        # cleaning + 8:1:1 train/val/test split
│   └── raw/                 # raw .npy files (gitignored)
│
├── models/
│   ├── train_nondl.py       # Random Forest training
│   ├── train_dl.py          # LSTM training (MPS accelerated)
│   ├── hand_landmarker.task # MediaPipe model (~8 MB)
│   ├── nondl_classifier.pkl
│   ├── nondl_scaler.pkl
│   └── lstm_model.pth
│
├── agent/
│   ├── gesture_agent.py     # main entry point + run loop
│   ├── perception.py        # OpenCV + MediaPipe Tasks API
│   ├── planning_nondl.py    # Non-DL planning layer
│   ├── planning_dl.py       # DL planning layer
│   ├── control.py           # App history + pynput + osascript
│   └── tray_icon.py         # macOS menu-bar icon (rumps)
│
└── eval/
    ├── evaluate_offline.py      # confusion matrix + F1 report
    ├── evaluate_online.py       # real-world accuracy + latency
    └── visualize_prototypes.py  # render most-confident sample per class
```

---

## Setup

### Step 0 — Environment

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install mediapipe opencv-python scikit-learn pynput numpy scipy \
    rumps matplotlib seaborn joblib
pip install torch torchvision   # Apple Silicon MPS support built-in
python scripts/download_models.py
python scripts/check_env.py
```

### Step 1 — macOS Permissions (one-time, manual)

| Permission | Location | Required |
|------------|----------|---------|
| Camera | System Settings → Privacy → Camera → VS Code | **Yes** |
| Accessibility | System Settings → Privacy → Accessibility → VS Code | **Yes** — pynput won't work without this |
| Screen Recording | System Settings → Privacy → Screen Recording → VS Code | Optional (demo recording) |

> ⚠️ Accessibility permission does **not** auto-prompt — you must add it manually.

### Step 2 — Data Collection

```bash
python data/collect.py                        # collect all 5 gestures (200 samples each)
python data/collect.py --gesture swipe_left   # collect one gesture only (resumes from checkpoint)
python data/collect.py --samples 50           # quick test with fewer samples
```

**Collection tips by gesture:**

| Gesture | Key tip |
|---------|---------|
| swipe_left | Palm facing camera, horizontal sweep left ~20cm. Wrist naturally twists slightly — embrace it. |
| swipe_right | Mirror of swipe_left. Match amplitude. |
| swipe_up | Vertical lift from chest to eye level, ~15cm. Stay vertical. |
| swipe_down | Mirror of swipe_up. |
| fist_open | Full fist first (~0.3s) → fast open. Incomplete fist = missed detection. |

Vary your speed and distance across recordings to improve generalisation.

### Step 3 — Data Augmentation

```bash
python data/augment.py --dry-run   # preview counts without writing
python data/augment.py             # write augmented files
```

Augmentation strategies: horizontal flip with label swap (swipe_left ↔ swipe_right), time warp (×2), Gaussian noise (×2), scale jitter (×2).

> **Important**: `NUM_ORIGINAL` in `augment.py` must match your actual raw sample count per class (default 200). Only raw files with index < `NUM_ORIGINAL` are augmented — previously augmented files are skipped.

### Step 4 — Preprocessing

```bash
python data/preprocess.py   # clean + 8:1:1 split
```

### Step 5 — Training

```bash
python models/train_nondl.py              # Random Forest (~1 min)
python models/train_dl.py                 # LSTM (~1 hr on M4 MPS)
python models/train_dl.py --epochs 80     # extended training
```

### Step 6 — Run the Agent

```bash
python agent/gesture_agent.py --debug          # DL mode + camera window
python agent/gesture_agent.py --mode nondl     # Non-DL mode
python agent/gesture_agent.py --tray           # with menu-bar icon

# Background (demo recommended)
nohup python agent/gesture_agent.py --tray > agent.log 2>&1 &
tail -f agent.log
kill $(cat agent.pid)
```

### Step 7 — Evaluation

```bash
python eval/evaluate_offline.py                       # confusion matrix + F1
python eval/evaluate_online.py --mode dl --trials 20  # real-world accuracy
python eval/evaluate_online.py --mode nondl --trials 20
python eval/visualize_prototypes.py --mode dl --save  # most-confident sample per class
```

---

## Design Decisions

### Why 5 gestures?

5 classes fully satisfy course requirements (perception/planning/control + DL/Non-DL comparison). Fewer classes means more samples per class, more stable models, and higher demo success rate.

### Why self-collected data?

All 5 gestures are dynamic sequences; HaGRID (the largest public hand gesture dataset) only contains static poses. Self-collected data also matches the deployment environment, giving better real-world accuracy.

### Why Random Forest for Non-DL?

- Feature importance is interpretable (meets the "justify decisions" requirement)
- Stable on small samples (~160 training examples per class)
- Inference < 2ms, well within the 200ms budget

### Control layer: bidirectional App history

Uses a pointer into an ordered history list instead of a simple queue. swipe_left/right move the pointer without modifying the list — rapid consecutive triggers behave stably. App switches triggered by the agent set a 0.6s write-lock on the monitor thread (> 0.3s poll interval) to prevent the agent's own switch from polluting the history.

---

## Key Bugs Fixed During Development

| Bug | Symptom | Fix |
|-----|---------|-----|
| `flip_horizontal` used `-x` instead of `1-x` | swipe_left recognised as swipe_right at high confidence | Changed to `1.0 - s[:,:,0]` |
| `augment.py` re-augmented previously augmented files | swipe_right support doubled on second run | Added `NUM_ORIGINAL` guard — only files with index < threshold are processed |
| `evaluate_online.py` grabbed stale frames before gesture | missed = 0, wrong rate ~70% | Added 3-2-1 countdown that drains the perception buffer before each trial |
| `activate_app` guard (0.25s) < monitor poll interval (0.3s) | Agent-triggered App switches appeared in history | Extended guard to 0.6s |
| `push()` reset ptr to tail on every call | swipe_right disabled immediately after swipe_left | Only advance ptr if it was already at the tail before append |

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Camera won't open | Permission not granted | System Settings → Camera → enable VS Code |
| Controls have no effect | Accessibility permission missing | System Settings → Accessibility → enable VS Code, restart |
| Left/right confusion | Training data distribution asymmetry | Re-run `augment.py` with corrected flip formula; vary gesture motion during collection |
| LSTM fist_open not triggering | Confidence below threshold | `fist_open` threshold is already set to 0.55; vary fist speed during collection |
| History grows long | Apps recorded on every poll | Expected — history caps at 10 entries; use swipe_right to navigate forward |
| Camera occupied | Zoom/Teams exclusive lock | Quit other camera-using apps |
