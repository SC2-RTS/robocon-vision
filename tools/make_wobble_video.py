#!/usr/bin/env python
"""合成 D6 用的测试视频：目标缓慢平移 + 每帧高频抖动 + 胡椒噪声。

设计意图（为什么不用真实视频）：
    1. 你的 detect 是针对「暗背景 + 低对比度灰白目标」调的，套在别的视频上必然失效
    2. 只有知道「真实运动轨迹」，才能量化「平滑到底把抖动压掉了多少」
    所以这里：真值 = 缓慢平移（平滑运动），叠加的抖动 = 要消掉的高频分量。

产出：
    data/samples/wobble.mp4         150 帧，30fps，800x600
    data/samples/wobble_truth.csv   每帧每片叶子的真值中心（不含抖动）

用法：
    python tools/make_wobble_video.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import low_contrast_demo as demo  # noqa: E402

OUT_DIR = ROOT / "data" / "samples"
VIDEO = OUT_DIR / "wobble.mp4"
TRUTH = OUT_DIR / "wobble_truth.csv"

W, H = 800, 600
FRAMES = 150
FPS = 30
PAN_X = 60.0          # 整段视频里缓慢平移 60px（真值运动）
PAN_Y = 8.0           # 轻微上下起伏
JITTER_SIGMA = 3.5    # 每帧抖动标准差（要消掉的高频分量）


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2024)
    base = demo.make_scene(leaf=78, seed=7)          # 与 D5 用的图同源，参数才通用
    leaves = demo.LEAVES

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(VIDEO), fourcc, FPS, (W, H))
    if not writer.isOpened():
        print("✗ mp4v 编码器不可用，改存 MJPG/avi")
        return 1

    rows: list[tuple[int, int, float, float]] = []
    for i in range(FRAMES):
        t = i / (FRAMES - 1)
        pan_x = PAN_X * t
        pan_y = PAN_Y * np.sin(t * np.pi)

        # 抖动：每帧独立采样（模拟手持/机械振动）
        jx = rng.normal(0.0, JITTER_SIGMA)
        jy = rng.normal(0.0, JITTER_SIGMA)

        mat = np.float32([[1, 0, pan_x + jx], [0, 1, pan_y + jy]])
        shifted = cv2.warpAffine(base, mat, (W, H), borderMode=cv2.BORDER_REPLICATE)

        # 每帧独立撒椒盐（模拟传感器噪声，不是固定图案）
        frame = demo.add_salt_pepper(shifted, seed=1000 + i)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

        # 真值只记「平滑运动」，不含抖动 —— 这样"误差下降"就等于"抖动被压掉"
        for lid, ((cx, cy), _ax, _ang) in enumerate(leaves, 1):
            rows.append((i, lid, cx + pan_x, cy + pan_y))

    writer.release()

    with TRUTH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "leaf_id", "cx", "cy"])
        w.writerows(rows)

    size_mb = VIDEO.stat().st_size / 1024 / 1024
    print(f"已生成视频：{VIDEO.relative_to(ROOT)}　{FRAMES} 帧 / {FPS}fps / {size_mb:.1f} MB")
    print(f"已生成真值：{TRUTH.relative_to(ROOT)}　{len(rows)} 行（{FRAMES} 帧 × {len(leaves)} 片叶子）")
    print(f"真值运动：x 平移 {PAN_X:.0f}px，y 起伏 ±{PAN_Y:.0f}px；抖动 σ={JITTER_SIGMA}px（每帧独立）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
