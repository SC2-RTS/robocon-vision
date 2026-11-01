#!/usr/bin/env python
"""D6 用：量化「平滑到底把抖动压掉了多少」。

做法：把每帧检测到的目标中心，和 wobble_truth.csv 里的真值中心比，算 RMS 误差。
    真值 = 缓慢平移（真实运动），不含抖动
    误差 = 抖动 + 检测噪声  →  平滑后误差下降，就说明抖动被压掉了

对比三种做法：
    raw      不平滑
    median   RingBuffer(5) 多帧中位数（你 D4 手写的那个容器）
    ema      指数滑动平均（滞后更小，但抗离群差）

用法：
    python tools/score_track.py
    python tools/score_track.py --window 5 --ema 0.4
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classic.detect_targets import Params, find_targets  # noqa: E402

VIDEO = ROOT / "data" / "samples" / "wobble.mp4"
TRUTH = ROOT / "data" / "samples" / "wobble_truth.csv"

try:  # 必须用你 D4 手写的 RingBuffer —— 不是"优先"，是"没有就不跑"
    # 用 _loader 按路径加载：queue.py 里是 `from linked_list import ...` 的平级导入，
    # 直接 `import src.utils.data_structure.queue` 会失败（找不到 linked_list）。
    from src.utils.data_structure._loader import load_local  # noqa: E402

    RingBuffer = load_local("queue.py", "RingBuffer")
except Exception as exc:  # noqa: BLE001
    print("✗ 加载 src/utils/data_structure/queue.py 里的 RingBuffer 失败")
    print(f"  {type(exc).__name__}: {exc}")
    print("  先确认这个文件能单独跑通：python src\\utils\\data_structure\\queue.py")
    raise SystemExit(2)


def ring_to_list(buf) -> list:
    """把 RingBuffer 里现存数据按从旧到新取出来（优先用它自己的 to_list）。"""
    if hasattr(buf, "to_list"):
        return buf.to_list()
    # 退回手工重建：head/size/capacity 是它的公开字段
    return [buf.buffer[(buf.head + i) % buf.capacity] for i in range(buf.size)]


def load_truth(path: Path) -> dict[int, dict[int, tuple[float, float]]]:
    """{frame: {leaf_id: (cx, cy)}}"""
    out: dict[int, dict[int, tuple[float, float]]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.setdefault(int(row["frame"]), {})[int(row["leaf_id"])] = (
                float(row["cx"]),
                float(row["cy"]),
            )
    return out


def match(detections: list[dict], truth_pts: dict[int, tuple[float, float]]) -> dict[int, tuple[float, float]]:
    """按最近中心配对（有真值在手，配对不会错）。"""
    used: set[int] = set()
    pairs: dict[int, tuple[float, float]] = {}
    for leaf_id, (tx, ty) in truth_pts.items():
        best = None
        for j, d in enumerate(detections):
            if j in used:
                continue
            dist = math.hypot(d["cx"] - tx, d["cy"] - ty)
            if best is None or dist < best[1]:
                best = (j, dist)
        if best is not None:
            used.add(best[0])
            d = detections[best[0]]
            pairs[leaf_id] = (d["cx"], d["cy"])
    return pairs


def median_of(values: list[float]) -> float:
    return statistics.median(values)


class MedianTracker:
    """用 RingBuffer(window) 存最近 N 帧，输出中位数。"""

    def __init__(self, window: int):
        self.window = window
        self.buf = RingBuffer(window)

    def push(self, x: float, y: float) -> tuple[float, float]:
        self.buf.write((x, y))
        pts = ring_to_list(self.buf)
        if not pts:
            return x, y
        return median_of([p[0] for p in pts]), median_of([p[1] for p in pts])


class EmaTracker:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.state: tuple[float, float] | None = None

    def push(self, x: float, y: float) -> tuple[float, float]:
        if self.state is None:
            self.state = (x, y)
        else:
            a = self.alpha
            self.state = (a * x + (1 - a) * self.state[0], a * y + (1 - a) * self.state[1])
        return self.state


def main() -> int:
    ap = argparse.ArgumentParser(description="量化平滑对抖动的抑制效果")
    ap.add_argument("--video", default=str(VIDEO))
    ap.add_argument("--truth", default=str(TRUTH))
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--ema", type=float, default=0.4)
    args = ap.parse_args()

    truth = load_truth(Path(args.truth))
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"✗ 打不开视频 {args.video}")
        return 2

    leaf_ids = sorted({lid for pts in truth.values() for lid in pts})
    med = {lid: MedianTracker(args.window) for lid in leaf_ids}
    ema = {lid: EmaTracker(args.ema) for lid in leaf_ids}

    err = {"raw": [], "median": [], "ema": []}
    missing = 0
    frames = 0
    sample_rows: list[str] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx = frames
        frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets, _ = find_targets(gray, Params())
        pairs = match(dets, truth.get(idx, {}))

        for lid in leaf_ids:
            if lid not in pairs:
                missing += 1
                continue
            mx, my = pairs[lid]
            tx, ty = truth[idx][lid]
            smx, smy = med[lid].push(mx, my)
            emx, emy = ema[lid].push(mx, my)

            err["raw"].append(math.hypot(mx - tx, my - ty))
            err["median"].append(math.hypot(smx - tx, smy - ty))
            err["ema"].append(math.hypot(emx - tx, emy - ty))

            if lid == leaf_ids[0] and idx < 8:
                sample_rows.append(
                    f"  帧 {idx:>2}  真值({tx:6.1f},{ty:6.1f})  原始({mx:6.1f},{my:6.1f})"
                    f"  中位数({smx:6.1f},{smy:6.1f})  EMA({emx:6.1f},{emy:6.1f})"
                )

    cap.release()

    def rms(v: list[float]) -> float:
        return math.sqrt(sum(x * x for x in v) / len(v)) if v else 0.0

    print(f"视频 {Path(args.video).name}：{frames} 帧 × {len(leaf_ids)} 个目标"
          f"（漏检 {missing} 次 / 共 {frames * len(leaf_ids)} 次）")
    print(f"中位数窗口 = {args.window} 帧　EMA alpha = {args.ema}　（中位数用的是你 D4 的 RingBuffer）")
    print()
    print("| 方法 | RMS 误差 | 最大误差 | 误差降低 | 滞后 |")
    print("|---|---|---|---|---|")
    base = rms(err["raw"])
    for name, lag in (("raw 不平滑", "0 帧"), (f"median-{args.window}", f"≈{(args.window - 1) / 2:.1f} 帧"),
                      (f"ema-{args.ema}", f"≈{(1 - args.ema) / args.ema:.1f} 帧")):
        key = "raw" if name.startswith("raw") else ("median" if name.startswith("median") else "ema")
        r = rms(err[key])
        reduction = (1 - r / base) * 100 if base else 0
        tag = "—" if key == "raw" else f"{reduction:.0f}%"
        print(f"| {name} | {r:.2f} px | {max(err[key]):.1f} px | {tag} | {lag} |")

    print("\n前 8 帧的一个目标（看它怎么从抖变稳）：")
    print("\n".join(sample_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
