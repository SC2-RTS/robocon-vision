#!/usr/bin/env python
"""测量模型延迟：GPU / CPU 各一次，预热后取均值（附 p50/p95）。

为什么必须这么测（三个容易做错的地方）：
    1. **不预热** → 第一次推理要建 CUDA 上下文、cuDNN 选算法、分配显存，
       能到几百 ms，混进均值会让结果翻倍。必须先跑 warmup 次丢掉。
    2. **只报均值** → 延迟是长尾分布，均值会被少数慢帧拉高。
       所以同时报 std 和 p95（95% 的帧都快于这个值）。
    3. **端到端 vs 纯前向混着说** → `model(img)` 含 BGR→RGB、letterbox、
       HWC→CHW、NMS；纯前向只有网络 forward。两个数差 2~3 倍，必须分开报。

用法：
    python tools/bench_latency.py
    python tools/bench_latency.py --weights a.pt b.pt --iters 50 --imgsz 640
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def bench(fn, warmup: int, iters: int) -> tuple[float, float, float, float]:
    """返回 (mean, std, p50, p95)，单位 ms。"""
    for _ in range(warmup):            # 关键：预热，结果全部丢弃
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return st.mean(ts), st.pstdev(ts), ts[len(ts) // 2], ts[int(len(ts) * 0.95) - 1]


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO 延迟基准（GPU/CPU）")
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--image", default=str(ROOT / "data" / "samples" / "low_contrast_leaf.png"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"读不到图片：{args.image}")
    h, w = img.shape[:2]
    print(f"输入 {Path(args.image).name}  {w}x{h}  imgsz={args.imgsz}")
    print(f"预热 {args.warmup} 次（结果丢弃）　测量 {args.iters} 次")
    print(f"torch {torch.__version__}  CPU 线程数 {torch.get_num_threads()}")
    if torch.cuda.is_available():
        print(f"GPU {torch.cuda.get_device_name(0)}")
    print()

    header = (f"{'模型':<14}{'设备':<7}{'端到端 mean±std':>18}{'p50':>7}{'p95':>7}"
              f"{'pre':>6}{'inf':>6}{'post':>6}{'纯forward':>11}{'首次(不预热)':>14}")
    print(header)
    print("-" * len(header))

    for wp in args.weights:
        if not Path(wp).exists():
            print(f"{Path(wp).name:<14}（文件不存在，跳过）")
            continue
        for dev in ("cuda:0", "cpu"):
            if dev.startswith("cuda") and not torch.cuda.is_available():
                continue
            model = YOLO(wp)
            model.to(dev)

            # ⓪ 第一次调用（含 CUDA 上下文/cuDNN 选算法/显存分配）—— 这就是不能直接用它的原因
            t0 = time.perf_counter()
            first = model(img, conf=0.25, verbose=False)[0]
            first_ms = (time.perf_counter() - t0) * 1000
            # letterbox 之后的真实网络输入尺寸（800x600 送 640 会变成 640x480，不是 640x640）
            lh, lw = getattr(model.predictor, "imgsz", (args.imgsz, args.imgsz))

            # ① 端到端：model(img) 里包含预处理 + 推理 + NMS
            speed_log: list[dict] = []

            def run_end2end() -> None:
                r = model(img, conf=0.25, verbose=False)[0]
                speed_log.append(dict(r.speed))

            mean, std, p50, p95 = bench(run_end2end, args.warmup, args.iters)
            pre = st.mean(s.get("preprocess", 0) for s in speed_log)
            inf = st.mean(s.get("inference", 0) for s in speed_log)
            post = st.mean(s.get("postprocess", 0) for s in speed_log)

            # ② 纯前向：跳过所有前后处理，只测网络（用真实 letterbox 尺寸，保证可比）
            x = torch.zeros(1, 3, lh, lw, device=dev)
            net = model.model

            def run_forward() -> None:
                with torch.no_grad():
                    net(x)

            f_mean = bench(run_forward, args.warmup, args.iters)[0]

            print(f"{Path(wp).stem:<14}{dev:<7}"
                  f"{f'{mean:6.2f} ± {std:5.2f}':>18}{p50:>7.2f}{p95:>7.2f}"
                  f"{pre:>6.2f}{inf:>6.2f}{post:>6.2f}{f_mean:>11.2f}"
                  f"{first_ms:>13.0f}")
    print("\n列说明：pre/inf/post 来自 ultralytics 的 r.speed（端到端的分段）；"
          "纯forward 是 net(tensor) 的时间")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
