#!/usr/bin/env python
"""从 wobble.mp4 自动生成 YOLO 检测数据集（图 + 标签 + data.yaml）。

为什么自动生成，而不是手标：
    · 这段视频是我们合成的，目标的真实位置已知
    · 用你自己验证过的 find_targets() 算框，标签精度 ≈1px，比手标还准
    · 省下的 2 小时用来练 X-AnyLabeling（30 张真实照片，见计划第 3 天路 B）

关键点（和手标完全一致）：
    标签格式 = `类别 cx cy w h`，全部归一化到 0~1，中心点表示
    划分 = **按时间切分**（前 80% 训练，后 20% 验证），绝不随机打乱 → 避免相邻帧泄漏

用法：
    python tools/make_yolo_dataset.py
    python tools/make_yolo_dataset.py --frames 120 --split 0.8
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classic.detect_targets import Params, find_targets  # noqa: E402

VIDEO = ROOT / "data" / "samples" / "wobble.mp4"
CLASS_NAME = "leaf"


def mask_bboxes(mask: np.ndarray, min_area: float) -> list[tuple[float, float, float, float]]:
    """从二值 mask 的连通域算**精确**的轴对齐 bbox（x1, y1, x2, y2）。

    为什么不用 minAreaRect 反推：旋转矩形的轴对齐外接框会比目标本身大 15~20%，
    标签框太松会让模型学出"永远偏大"的框。
    连通域的 boundingRect 才是目标真实占据的像素范围。
    """
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: list[tuple[float, float, float, float]] = []
    for i in range(1, n):                       # 0 是背景
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        boxes.append((float(x), float(y), float(x + w), float(y + h)))
    return boxes


def to_yolo_line(box: tuple[float, float, float, float], w_img: int, h_img: int) -> str:
    """一行 YOLO 标签：0 cx cy w h（全部 / 图尺寸，中心点表示）。"""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / w_img
    cy = (y1 + y2) / 2 / h_img
    bw = (x2 - x1) / w_img
    bh = (y2 - y1) / h_img
    vals = [max(0.0, min(1.0, v)) for v in (cx, cy, bw, bh)]
    return "0 " + " ".join(f"{v:.6f}" for v in vals)


def main() -> int:
    ap = argparse.ArgumentParser(description="从 wobble.mp4 生成 YOLO 数据集")
    ap.add_argument("--video", default=str(VIDEO))
    ap.add_argument("--out", default=str(ROOT / "datasets" / "leaves"))
    ap.add_argument("--frames", type=int, default=0, help="最多用几帧，0=全部")
    ap.add_argument("--split", type=float, default=0.8, help="训练集比例（按时间切分）")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)                      # 重跑时清空，避免旧标签残留
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频：{args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_use = min(total, args.frames) if args.frames > 0 else total
    n_train = max(1, int(n_use * args.split))

    n_box_per_frame: list[int] = []
    box_px: list[tuple[float, float]] = []
    n_train_img = n_val_img = 0
    mismatch = 0

    for idx in range(n_use):
        ok, frame = cap.read()
        if not ok:
            break
        h_img, w_img = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        targets, _mask = find_targets(gray, Params())
        boxes = mask_bboxes(_mask, Params().min_area)

        is_train = idx < n_train
        split = "train" if is_train else "val"
        stem = f"frame_{idx:04d}"

        cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        lines = []
        for box in boxes:
            lines.append(to_yolo_line(box, w_img, h_img))
            box_px.append((box[2] - box[0], box[3] - box[1]))
        (out / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

        n_box_per_frame.append(len(boxes))
        if len(boxes) != len(targets):
            mismatch += 1
        if is_train:
            n_train_img += 1
        else:
            n_val_img += 1

    cap.release()

    (out / "data.yaml").write_text(
        # 注意：不要写 `path: .` —— ultralytics 会把相对 path 当成"数据集下载根"来解析，
        # 结果去找 <仓库根>/images/val 然后报错。省略 path 时，它用 yaml 所在目录当根，最稳。
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n",
        encoding="utf-8",
    )

    # ---------- 自检 ----------
    print("=== 数据集自检 ===")
    dist = Counter(n_box_per_frame)
    print(f"帧数 {len(n_box_per_frame)}　每帧框数分布 {dict(sorted(dist.items()))}")
    print(f"图：train {n_train_img} / val {n_val_img}（按时间切分，前 80% 训练）")
    print(f"标签总数 {sum(n_box_per_frame)}")
    print(f"连通域数与 find_targets 结果不一致的帧：{mismatch}")

    if box_px:
        ws = np.array([b[0] for b in box_px])
        hs = np.array([b[1] for b in box_px])
        print(f"框宽 px: min {ws.min():.0f} / mean {ws.mean():.0f} / max {ws.max():.0f}")
        print(f"框高 px: min {hs.min():.0f} / mean {hs.mean():.0f} / max {hs.max():.0f}")

    # 抽查 3 个标签文件：必须全在 [0,1]，且 w/h > 0.01
    bad = []
    for p in sorted((out / "labels" / "train").glob("*.txt"))[:3]:
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                bad.append(f"{p.name}: 字段数 {len(parts)} != 5")
                continue
            vals = [float(v) for v in parts[1:]]
            if not all(0.0 <= v <= 1.0 for v in vals):
                bad.append(f"{p.name}: 超出 [0,1] -> {vals}")
            if vals[2] < 0.01 or vals[3] < 0.01:
                bad.append(f"{p.name}: 框太小 -> {vals}")
    print("标签格式检查：" + ("通过" if not bad else "有问题"))
    for b in bad:
        print("  ✗", b)

    sample = sorted((out / "labels" / "train").glob("*.txt"))
    if sample:
        print(f"\n样例 {sample[0].name}：")
        print("  " + sample[0].read_text(encoding="utf-8").strip().replace("\n", "\n  "))

    print(f"\n产出目录：{out.relative_to(ROOT)}")
    print("下一步：")
    print(f"  yolo train data={out.relative_to(ROOT)}/data.yaml model=yolov8n.pt "
          f"epochs=1 imgsz=640")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
