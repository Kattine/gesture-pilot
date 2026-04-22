"""
data/augment.py
离线数据增强 — 在预处理之前对原始 .npy 文件做扩充

解决的核心问题：
  swipe_left / swipe_right 样本量少时容易混淆。
  水平翻转 swipe_left → 生成等效的 swipe_right 样本（反之亦然），
  在不额外采集的前提下将两类样本量翻倍，且物理含义严格对应。

增强策略：
  1. 水平翻转（swipe_left ↔ swipe_right 互换）
  2. 时序抖动（随机裁剪 + 重采样回 30 帧，模拟动作快慢）
  3. 坐标噪声（每帧关键点加微小高斯噪声，模拟手部抖动）
  4. 关键点缩放（整体手势缩放，模拟远近距离变化）

运行方式：
  python data/augment.py            # 增强并保存回 data/raw/
  python data/augment.py --dry-run  # 只打印统计，不写文件

在 preprocess.py 之前运行此脚本。
"""

import os
import glob
import argparse
import numpy as np
from scipy.interpolate import interp1d

RAW_DIR      = os.path.join(os.path.dirname(__file__), "raw")
GESTURES     = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "fist_open"]
SEQ_LEN      = 30
NUM_ORIGINAL = 250          # ★ FIX2：每类原始采集样本固定数量，增强时只读这些文件，有 200个就增强 200 个
MIRROR_PAIRS = [("swipe_left", "swipe_right")]   # 水平翻转时互换标签


# ─────────────────────────────────────────────
# 增强函数（每个函数接收 (30,63) 返回 (30,63)）
# ─────────────────────────────────────────────

def flip_horizontal(seq: np.ndarray) -> np.ndarray:
    """
    水平翻转：x 坐标镜像。
    MediaPipe 输出的 x 已归一化到 [0, 1]，镜像公式为 1 - x（不是 -x）。
    对 swipe_left 做此操作等价于 swipe_right，标签需同步互换。

    ★ FIX1：原代码写 -s[:,:,0]，导致翻转后 x 全为负数，
             超出 [0,1] 正常范围，模型学到的是无效坐标。
             改为 1.0 - s[:,:,0] 才是真正的水平镜像。
    """
    s = seq.copy().reshape(SEQ_LEN, 21, 3)
    s[:, :, 0] = 1.0 - s[:, :, 0]   # ★ FIX1：1-x，而非 -x
    return s.reshape(SEQ_LEN, 63)


def time_warp(seq: np.ndarray,
              crop_ratio: float = 0.85) -> np.ndarray:
    """
    时序抖动：随机从序列中裁剪 crop_ratio 比例的帧，再插值回 SEQ_LEN 帧。
    模拟动作快慢差异。crop_ratio ∈ (0.7, 1.0)
    """
    T = SEQ_LEN
    keep = max(int(T * crop_ratio), 10)
    start = np.random.randint(0, T - keep)
    cropped = seq[start:start + keep]               # (keep, 63)
    old_t = np.linspace(0, 1, keep)
    new_t = np.linspace(0, 1, T)
    warped = np.zeros((T, 63), dtype=np.float32)
    for d in range(63):
        f = interp1d(old_t, cropped[:, d], kind="linear")
        warped[:, d] = f(new_t)
    return warped


def add_noise(seq: np.ndarray, sigma: float = 0.008) -> np.ndarray:
    """坐标高斯噪声"""
    return seq + np.random.randn(*seq.shape).astype(np.float32) * sigma


def scale_keypoints(seq: np.ndarray,
                    scale_range=(0.85, 1.15)) -> np.ndarray:
    """
    关键点整体缩放（模拟手离摄像头远近）。
    以腕关节为原点缩放，不改变相对坐标的方向。
    """
    scale = np.random.uniform(*scale_range)
    s = seq.copy().reshape(SEQ_LEN, 21, 3)
    wrist = s[:, 0:1, :].copy()
    s = (s - wrist) * scale + wrist
    return s.reshape(SEQ_LEN, 63)


# ─────────────────────────────────────────────
# 增强策略组合
# ─────────────────────────────────────────────

def augment_sample(seq: np.ndarray,
                   gesture: str) -> list[tuple[np.ndarray, str]]:
    """
    对单个样本生成多条增强样本。
    返回 [(augmented_seq, label), ...]
    """
    results = []

    # 策略1：时序抖动（所有类别）
    for ratio in [0.80, 0.90]:
        results.append((time_warp(seq, ratio), gesture))

    # 策略2：坐标噪声（所有类别）
    results.append((add_noise(seq, sigma=0.006), gesture))
    results.append((add_noise(seq, sigma=0.012), gesture))

    # 策略3：缩放（所有类别）
    results.append((scale_keypoints(seq, (0.85, 1.0)), gesture))
    results.append((scale_keypoints(seq, (1.0, 1.15)), gesture))

    # 策略4：水平翻转 + 标签互换（仅 swipe_left / swipe_right）
    for a, b in MIRROR_PAIRS:
        if gesture == a:
            results.append((flip_horizontal(seq), b))
        elif gesture == b:
            results.append((flip_horizontal(seq), a))

    return results


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def augment_gesture(gesture: str, dry_run: bool = False):
    src_dir = os.path.join(RAW_DIR, gesture)

    # ★ FIX2：只读取原始文件（索引 0000 ~ NUM_ORIGINAL-1），
    #          不把上次增强生成的文件再次增强（否则翻转的翻转会产生错误标签）
    all_files = sorted(glob.glob(os.path.join(src_dir, "*.npy")))
    files = [
        f for f in all_files
        if int(os.path.splitext(os.path.basename(f))[0]) < NUM_ORIGINAL
    ]

    if not files:
        print(f"  ⚠️  {gesture}：无原始数据，跳过")
        return 0

    # 起始索引 = 当前目录全部已有文件数（含上次未清理的增强样本，若有）
    next_idx = len(all_files)

    # 用字典缓存跨手势目录的下一个索引，避免在同一次运行里重复 glob
    cross_next: dict[str, int] = {}

    new_count = 0
    for f in files:
        seq = np.load(f)
        augmented = augment_sample(seq, gesture)

        for aug_seq, aug_label in augmented:
            dst_dir = os.path.join(RAW_DIR, aug_label)
            os.makedirs(dst_dir, exist_ok=True)

            if aug_label == gesture:
                # 同目录：使用递增的 next_idx
                save_path = os.path.join(src_dir, f"{next_idx:04d}.npy")
                next_idx += 1
            else:
                # 跨目录（翻转互换）：用缓存计数器，避免每次 glob
                if aug_label not in cross_next:
                    cross_next[aug_label] = len(
                        glob.glob(os.path.join(dst_dir, "*.npy"))
                    )
                save_path = os.path.join(
                    dst_dir, f"{cross_next[aug_label]:04d}.npy"
                )
                cross_next[aug_label] += 1

            if not dry_run:
                np.save(save_path, aug_seq)
            new_count += 1

    return new_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印统计，不写文件")
    args = parser.parse_args()

    print("\n" + "=" * 52)
    print("  数据增强")
    print("=" * 52)
    if args.dry_run:
        print("  [DRY RUN 模式，不写文件]\n")

    total = 0
    for gesture in GESTURES:
        before = len(glob.glob(os.path.join(RAW_DIR, gesture, "*.npy")))
        added  = augment_gesture(gesture, dry_run=args.dry_run)
        after  = before + added
        print(f"  {gesture:<18}: {before:>4} → +{added:<4} = {after}")
        total += added

    print(f"\n  新增样本总数：{total}")
    if not args.dry_run:
        print("  ✅ 增强完成，下一步：python data/preprocess.py")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    main()
