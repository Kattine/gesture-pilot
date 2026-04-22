"""
eval/rebuild_robustness.py
Reconstruct robustness chart from terminal output numbers.
Run from project root: python eval/rebuild_robustness.py
"""
import os, sys, numpy as np, matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "eval", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "fist_open"]
TRIALS   = 3

# ── DL correct counts per gesture [L←, R→, U↑, D↓, Fist] ──
DL = {
    "Normal\n(baseline)":  [3, 0, 1, 0, 3],   # overall 47%
    "Low Light":           [0, 2, 0, 0, 3],   # overall 33%
    "Complex\nBackground": [0, 0, 0, 0, 2],   # overall 13%
}

# ── Non-DL correct counts per gesture ──
NonDL = {
    "Normal\n(baseline)":  [2, 0, 0, 0, 3],   # overall 33%
    "Low Light":           [0, 0, 0, 0, 2],   # overall 13%
    "Complex\nBackground": [3, 0, 0, 0, 3],   # overall 40%
}

conditions = list(DL.keys())
def to_acc(d): return sum(d) / (len(d) * TRIALS)

dl_accs    = [to_acc(DL[c])    for c in conditions]
nondl_accs = [to_acc(NonDL[c]) for c in conditions]

# Save merged npy so visualize_results.py loads real data
robustness_data = {c: {"DL": to_acc(DL[c]), "Non-DL": to_acc(NonDL[c])} for c in conditions}
np.save(os.path.join(RESULT_DIR, "robustness_results.npy"), robustness_data)
print("Saved: robustness_results.npy")

# ── Chart ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x, width = np.arange(len(conditions)), 0.35

b1 = ax.bar(x - width/2, nondl_accs, width, label="Non-DL (RF)", color="#4C8BF5", alpha=0.85)
b2 = ax.bar(x + width/2, dl_accs,    width, label="DL (LSTM)",   color="#FF6B35", alpha=0.85)

for bar, a in zip(b1, nondl_accs):
    ax.text(bar.get_x()+bar.get_width()/2, a+0.01, f"{a:.0%}",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#2255bb")
for bar, a in zip(b2, dl_accs):
    ax.text(bar.get_x()+bar.get_width()/2, a+0.01, f"{a:.0%}",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#cc4400")

ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Robustness Test: Non-DL vs DL  (3 trials x 5 gestures per condition)", fontsize=13)
ax.legend(fontsize=11)
ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Per-gesture breakdown table
col_labels = ["L<-", "R->", "U^", "Dv", "Fist"]
row_labels, cell_data = [], []
for c in conditions:
    row_labels.append(f"DL  - {c.replace(chr(10), ' ')}")
    cell_data.append([f"{v/TRIALS:.0%}" for v in DL[c]])
for c in conditions:
    row_labels.append(f"RF  - {c.replace(chr(10), ' ')}")
    cell_data.append([f"{v/TRIALS:.0%}" for v in NonDL[c]])

plt.subplots_adjust(bottom=0.38)
table = ax.table(cellText=cell_data, rowLabels=row_labels, colLabels=col_labels,
                 cellLoc="center", loc="bottom", bbox=[0, -0.72, 1, 0.52])
table.auto_set_font_size(False)
table.set_fontsize(8.5)
for (row, col), cell in table.get_celld().items():
    if row == 0:                           cell.set_facecolor("#dddddd")
    elif 1 <= row <= len(conditions):      cell.set_facecolor("#fff2ec")
    else:                                  cell.set_facecolor("#ecf0ff")

out = os.path.join(RESULT_DIR, "robustness_table.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Chart saved: {out}")

print("\nSummary:")
print(f"  {'Condition':<24} {'DL':>6}  {'Non-DL':>8}")
for c, da, na in zip(conditions, dl_accs, nondl_accs):
    print(f"  {c.replace(chr(10),' '):<24} {da:>6.0%}  {na:>8.0%}")
