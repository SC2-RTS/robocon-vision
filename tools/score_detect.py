#!/usr/bin/env python
"""给 D5 的 detect() 打分：数量 / 中心误差 / 角度误差 / 单帧耗时。

它只做一件事：**用真值检查你的输出**，不关心你怎么实现。

用法：
    python tools/score_detect.py                        # 跑 data/samples 下所有 *_truth.json
    python tools/score_detect.py --image data/samples/low_contrast_leaf.png

约定（你的 src/classic/detect_targets.py 必须满足）：
    def detect(image_path: str) -> list[dict]:
        # 每个 dict 至少包含 cx, cy, w, h, angle
        # 没有目标就返回 []
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"

# 及格线（和计划里写的一致）
MAX_CENTER_ERR = 5.0     # px
MAX_ANGLE_ERR = 15.0     # 度
MAX_TIME_MS = 20.0       # 单帧
REQUIRED_KEYS = ("cx", "cy", "w", "h", "angle")


def load_detect():
    sys.path.insert(0, str(ROOT))
    try:
        from src.classic.detect_targets import detect  # noqa: PLC0415
    except ImportError as exc:
        print("✗ 找不到 src/classic/detect_targets.py 里的 detect()")
        print(f"  原因：{exc}")
        print("\n先做这一步（10 行以内就能跑）：")
        print("  1. 新建 src/classic/detect_targets.py")
        print("  2. 写一个先返回空列表的骨架：")
        print("       import cv2\n"
              "       def detect(image_path: str) -> list[dict]:\n"
              "           return []")
        print("  3. 再跑一次本脚本 —— 你会看到 0/3 FAIL，然后就知道该往哪使劲了")
        return None
    return detect


def angle_error(truth: float, pred: float) -> float:
    """矩形转 90° 是同一个矩形，所以按 mod 90 比较。"""
    d = abs(truth - pred) % 90.0
    return min(d, 90.0 - d)


def score_one(detect, truth_path: Path) -> bool:
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    image_path = SAMPLES / truth["image"]
    targets = truth["targets"]

    if not image_path.exists():
        print(f"✗ 图片不存在：{image_path}")
        return False

    # 先跑一次热身（排除首次调用的开销），再取 3 次最小值
    try:
        detect(str(image_path))
    except Exception as exc:  # noqa: BLE001
        print(f"✗ detect() 抛异常：{type(exc).__name__}: {exc}")
        return False

    best_ms = float("inf")
    pred = None
    for _ in range(3):
        t0 = time.perf_counter()
        pred = detect(str(image_path))
        best_ms = min(best_ms, (time.perf_counter() - t0) * 1000)

    print(f"\n══ {truth['image']}　（真值 {len(targets)} 个目标，灰度差 {truth['leaf_gray'] - truth['bg_gray']}）")
    if not isinstance(pred, list):
        print(f"✗ detect() 必须返回 list，现在返回了 {type(pred).__name__}")
        return False

    missing_key = next((k for k in REQUIRED_KEYS if pred and k not in pred[0]), None)
    if missing_key:
        print(f"✗ 返回的 dict 缺少字段：{missing_key}（需要 {REQUIRED_KEYS}）")
        return False

    print(f"  检测到 {len(pred)} 个，真值 {len(targets)} 个")

    # 贪心最近邻配对（3 个目标足够）
    remaining = list(range(len(pred)))
    rows = []
    for t in targets:
        if not remaining:
            rows.append((t, None, None, None))
            continue
        j = min(remaining, key=lambda k: (pred[k]["cx"] - t["cx"]) ** 2 + (pred[k]["cy"] - t["cy"]) ** 2)
        remaining.remove(j)
        p = pred[j]
        center = ((p["cx"] - t["cx"]) ** 2 + (p["cy"] - t["cy"]) ** 2) ** 0.5
        rows.append((t, p, center, angle_error(t["angle"], p["angle"])))

    print(f"  {'#':<3}{'真值中心':>16}{'检测中心':>18}{'中心误差':>10}{'角度(真/检/差)':>20}")
    max_center = 0.0
    max_angle = 0.0
    for t, p, center, ang in rows:
        tc = f"({t['cx']:.0f},{t['cy']:.0f})"
        if p is None:
            print(f"  {t['id']:<3}{tc:>16}{'—— 漏检 ——':>18}")
            max_center = float("inf")
            continue
        pc = f"({p['cx']:.0f},{p['cy']:.0f})"
        ac = f"{t['angle']:.0f}/{p['angle']:.0f}/{ang:.1f}"
        max_center = max(max_center, center)
        max_angle = max(max_angle, ang)
        print(f"  {t['id']:<3}{tc:>16}{pc:>18}{center:>8.1f}px{ac:>20}")

    ok_count = len(pred) == len(targets)
    ok_center = max_center <= MAX_CENTER_ERR
    ok_angle = max_angle <= MAX_ANGLE_ERR
    ok_time = best_ms <= MAX_TIME_MS

    def mark(ok: bool, text: str) -> str:
        return f"{'✓' if ok else '✗'} {text}"

    print(f"  {mark(ok_count, f'数量 {len(pred)}/{len(targets)}')}　"
          f"{mark(ok_center, f'最大中心误差 {max_center:.1f}px (≤{MAX_CENTER_ERR:.0f})')}　"
          f"{mark(ok_angle, f'最大角度误差 {max_angle:.1f}° (≤{MAX_ANGLE_ERR:.0f})')}　"
          f"{mark(ok_time, f'耗时 {best_ms:.1f}ms (≤{MAX_TIME_MS:.0f})')}")
    print(f"  ▶ {'PASS' if all([ok_count, ok_center, ok_angle, ok_time]) else 'FAIL'}")
    return all([ok_count, ok_center, ok_angle, ok_time])


def main() -> int:
    ap = argparse.ArgumentParser(description="给 D5 的 detect() 打分")
    ap.add_argument("--image", help="只测某一张图（不传就测 data/samples 下全部）")
    args = ap.parse_args()

    detect = load_detect()
    if detect is None:
        return 2

    if args.image:
        truth_paths = [Path(args.image).with_name(Path(args.image).stem + "_truth.json")]
    else:
        truth_paths = sorted(SAMPLES.glob("*_truth.json"))

    if not truth_paths:
        print(f"✗ {SAMPLES} 下没有 *_truth.json")
        print("  先跑：python tools/low_contrast_demo.py --regen")
        return 2

    results = [score_one(detect, p) for p in truth_paths if p.exists()]

    print("\n" + "=" * 60)
    print(f"{sum(results)}/{len(results)} 张图全部指标达标" +
          ("　→ D5 完成 ✓" if all(results) else "　→ 继续按 D5 的『症状→工具』表调"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
