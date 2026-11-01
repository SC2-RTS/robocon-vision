#!/usr/bin/env python
"""生成 D5 参数实验记录的表格和对比图（docs/opencv_params.md 的证据来源）。

它做三件事：
    1. 逐项改一个参数，打印一张 markdown 表格（参数 → 指标）
    2. 存三张对比图到 docs/images/
    3. 分段计时，打印各阶段耗时占比

用法：
    python tools/params_report.py              # 打印表格 + 生成图片
    python tools/params_report.py > params.md  # 把表格存下来，粘进文档

结论（"所以该选哪个值"）要你自己写 —— 这个脚本只负责给数字和证据。
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
IMAGES = ROOT / "docs" / "images"
IMAGE = SAMPLES / "low_contrast_leaf.png"

import sys

sys.path.insert(0, str(ROOT))
from src.classic.detect_targets import Params, find_targets  # noqa: E402


def coverage(mask: np.ndarray) -> float:
    return cv2.countNonZero(mask) / mask.size * 100.0


def probe(gray: np.ndarray, **kw) -> tuple[int, float]:
    params = Params(**kw)
    targets, mask = find_targets(gray, params)
    return len(targets), coverage(mask)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out = out.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def two_up(a: np.ndarray, la: str, b: np.ndarray, lb: str) -> np.ndarray:
    return np.hstack([label(a, la), label(b, lb)])


def draw_targets(gray_bgr: np.ndarray, targets: list[dict]) -> np.ndarray:
    out = gray_bgr.copy()
    for t in targets:
        box = cv2.boxPoints(((t["cx"], t["cy"]), (t["w"], t["h"]), t["angle"]))
        cv2.drawContours(out, [np.intp(box)], -1, (0, 255, 0), 2)
        cv2.circle(out, (int(t["cx"]), int(t["cy"])), 3, (0, 255, 0), -1)
    return out


def make_evidence(gray: np.ndarray) -> list[Path]:
    IMAGES.mkdir(parents=True, exist_ok=True)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    saved: list[Path] = []

    # ① 默认参数：原图+框 vs mask
    targets, mask = find_targets(gray, Params())
    p = IMAGES / "params-default.png"
    cv2.imwrite(str(p), two_up(draw_targets(gray_bgr, targets), "default: gray + rects",
                              mask, f"mask: targets={len(targets)} coverage={coverage(mask):.1f}%"))
    saved.append(p)

    # ② otsu_delta=0 vs -30：阈值跌破背景亮度的后果
    _, mask0 = find_targets(gray, Params(otsu_delta=0))
    _, maskbad = find_targets(gray, Params(otsu_delta=-30))
    p = IMAGES / "params-delta-neg30.png"
    cv2.imwrite(str(p), two_up(mask0, f"otsu_delta=0  coverage={coverage(mask0):.1f}%",
                              maskbad, f"otsu_delta=-30  coverage={coverage(maskbad):.1f}%"))
    saved.append(p)

    # ③ 加杂点：不过滤 vs 面积过滤
    clutter = gray.copy()
    for x, y, r in [(60, 80, 6), (700, 60, 5), (120, 520, 7), (760, 300, 5), (400, 560, 6), (40, 300, 5)]:
        cv2.circle(clutter, (x, y), r, 140, -1)
    t0, m0 = find_targets(clutter, Params(min_area=0.0))
    t1, m1 = find_targets(clutter, Params(min_area=2000.0))
    n0, n1 = len(t0), len(t1)
    p = IMAGES / "params-clutter.png"
    cv2.imwrite(str(p), two_up(m0, f"+6 blobs, min_area=0  targets={n0} coverage={coverage(m0):.1f}%",
                              m1, f"+6 blobs, min_area=2000  targets={n1} coverage={coverage(m1):.1f}%"))
    saved.append(p)
    return saved


def md_table(gray: np.ndarray) -> str:
    rows: list[tuple[str, str, str, str]] = []

    def add(param: str, value, note: str = "") -> None:
        n, cov = probe(gray, **{param: value})
        rows.append((param, str(value), f"{n} / {cov:.2f}%", note))

    for v in (0, 20, 30):
        add("otsu_delta", v)
    for v in (-15, -30):
        add("otsu_delta", v)
    for v in (2000, 17500.0, 18000.0, 26000.0, 27000.0):
        add("min_area", v)
    for v in (1, 5, 9):
        add("median_k", v)
    for v in (9, 25):
        add("open_k", v)
    for v in (15, 31):
        add("close_k", v)
    add("use_clahe", True)

    out = ["| 参数 | 取值 | 目标数 / 前景覆盖率 |", "|---|---|---|"]
    for param, value, metric, _note in rows:
        out.append(f"| `{param}` | {value} | {metric} |")
    return "\n".join(out)


def md_timing(gray: np.ndarray, repeat: int = 30) -> str:
    """分段计时：整条链路 + 几个主要阶段（阶段单独测，用于看谁最贵）。"""
    rows: list[tuple[str, float]] = []

    t0 = time.perf_counter()
    for _ in range(repeat):
        find_targets(gray, Params())
    rows.append(("整条 find_targets（含找轮廓、质心、最小外接矩形）", (time.perf_counter() - t0) / repeat * 1000))

    t0 = time.perf_counter()
    for _ in range(repeat):
        med = cv2.medianBlur(gray, 5)
    rows.append(("cv2.medianBlur(k=5)", (time.perf_counter() - t0) / repeat * 1000))

    t0 = time.perf_counter()
    for _ in range(repeat):
        _, m = cv2.threshold(med, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rows.append(("cv2.threshold(Otsu)", (time.perf_counter() - t0) / repeat * 1000))

    t0 = time.perf_counter()
    for _ in range(repeat):
        cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows.append(("cv2.findContours", (time.perf_counter() - t0) / repeat * 1000))

    # 关键对比：读一张 PNG + 转灰度，可能比上面全部处理加起来还贵
    t0 = time.perf_counter()
    for _ in range(repeat):
        img = cv2.imread(str(IMAGE))
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows.append(("cv2.imread + cvtColor（读图 + 转灰度）", (time.perf_counter() - t0) / repeat * 1000))

    total = rows[0][1]
    out = ["| 阶段 | 耗时 (ms) | 占比 |", "|---|---|---|"]
    for name, ms in rows:
        share = ms / total * 100 if total else 0
        out.append(f"| {name} | {ms:.2f} | {share:.0f}% |")
    return "\n".join(out)


def main() -> int:
    gray = cv2.imread(str(IMAGE), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"✗ 读不到 {IMAGE}")
        return 1

    print("## 参数与指标（同一张图，逐项只改一个参数）\n")
    print(md_table(gray))
    print()
    print("## 分段耗时（各 30 次平均）\n")
    print(md_timing(gray))

    for p in make_evidence(gray):
        print(f"\n已生成图片：{p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
