#!/usr/bin/env python
"""D5 调参窗口：拖滑动条即时看 mask / 目标数 / 覆盖率，按 S 把参数存成 JSON。

前置：src/classic/detect_targets.py 里有 Params 和 find_targets（见文档开头的契约）。

用法：
    python -m src.classic.tune_params
    python -m src.classic.tune_params --image data/samples/low_contrast_leaf_hard.png
按键：ESC/q 退出　S 保存当前参数
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from src.classic.detect_targets import Params, find_targets
except ImportError as exc:
    print("✗ 找不到 Params / find_targets。先按下面的契约重构 src/classic/detect_targets.py：")
    print("    @dataclass(frozen=True)")
    print("    class Params: median_k=5, otsu_delta=0, open_k=0, close_k=0,")
    print("                  min_area=2000.0, use_clahe=False")
    print("    def find_targets(gray, params) -> tuple[list[dict], np.ndarray]: ...")
    print("    def detect(path, params=None) -> list[dict]: ...")
    print(f"\n原始错误：{exc}")
    raise SystemExit(2)

WINDOW = "tune_params"
DELTA_RANGE = 40          # otsu_delta 滑动条范围：-40 ~ +40（OpenCV 滑动条不能为负，所以用偏移）


def nothing(_=None):
    """滑动条回调：这里什么都不做，主循环里统一读值。"""


def _max_panel_width() -> int:
    """窗口宽度上限：超过屏幕宽度时 OpenCV 会把窗口整体缩小，图就变小了。"""
    try:
        from ctypes import windll

        screen_w = windll.user32.GetSystemMetrics(0)
        return max(700, min(1500, screen_w - 100))
    except Exception:
        return 1400


def make_panel(gray_bgr: np.ndarray, mask: np.ndarray, targets: list[dict],
               params: Params, ms: float) -> np.ndarray:
    left = gray_bgr.copy()
    for t in targets:
        box = cv2.boxPoints(((t["cx"], t["cy"]), (t["w"], t["h"]), t["angle"]))
        cv2.drawContours(left, [np.intp(box)], -1, (0, 255, 0), 2)
        cv2.circle(left, (int(t["cx"]), int(t["cy"])), 3, (0, 255, 0), -1)

    right = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    panel = np.hstack([left, right])

    # 两块 800px 拼起来 1600px，笔记本屏幕放不下就会被动缩小 → 主动缩到屏幕宽度内
    max_w = _max_panel_width()
    if panel.shape[1] > max_w:
        scale = max_w / panel.shape[1]
        panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    cov = cv2.countNonZero(mask) / mask.size * 100.0
    bar = np.zeros((40, panel.shape[1], 3), np.uint8)
    text = (f"count={len(targets)}  coverage={cov:5.1f}%  {ms:5.1f}ms   "
            f"k={params.median_k} d={params.otsu_delta:+d} "
            f"open={params.open_k} close={params.close_k} "
            f"minA={params.min_area:.0f} clahe={int(params.use_clahe)}   [S]save [ESC]quit")
    cv2.putText(bar, text, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def main() -> int:
    ap = argparse.ArgumentParser(description="D5 调参窗口")
    ap.add_argument("--image", default=str(ROOT / "data" / "samples" / "low_contrast_leaf.png"))
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"✗ 读不到图片：{args.image}")
        return 2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    # 滑动条：名字、范围、默认值（注意 OpenCV 滑动条只能是整数，且不能为负）
    cv2.createTrackbar("median_k", WINDOW, 5, 9, nothing)      # 会在下面强制取奇数
    cv2.createTrackbar("otsu_delta", WINDOW, DELTA_RANGE, DELTA_RANGE * 2, nothing)  # 0..80 → -40..+40
    cv2.createTrackbar("open_k", WINDOW, 0, 9, nothing)
    cv2.createTrackbar("close_k", WINDOW, 0, 9, nothing)
    cv2.createTrackbar("min_area", WINDOW, 2000, 20000, nothing)  # 上限 20000：真目标约 17500，够拖到"筛掉"的区间
    cv2.createTrackbar("use_clahe", WINDOW, 0, 1, nothing)

    def pos(name: str) -> int:
        return cv2.getTrackbarPos(name, WINDOW)

    print("拖滑动条看左边（检测结果）和右边（mask）怎么变；按 S 保存，ESC 退出。")

    while True:
        params = Params(
            median_k=pos("median_k") | 1,                    # 强制奇数：1,3,5,7,9
            otsu_delta=pos("otsu_delta") - DELTA_RANGE,      # 还原成 -40..+40
            open_k=pos("open_k"),
            close_k=pos("close_k"),
            min_area=float(pos("min_area")),
            use_clahe=bool(pos("use_clahe")),
        )

        t0 = time.perf_counter()
        targets, mask = find_targets(gray, params)      # ← 调的是和 detect 同一个函数
        ms = (time.perf_counter() - t0) * 1000

        cv2.imshow(WINDOW, make_panel(gray_bgr, mask, targets, params, ms))
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            break

        if key in (ord("s"), ord("S")):
            payload = asdict(params)
            out = ROOT / "data" / "samples" / "tuned_params.json"
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n已保存到 {out}")
            print("把这几个数填回 Params 的默认值即可：")
            for k, v in payload.items():
                print(f"    {k} = {v}")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())