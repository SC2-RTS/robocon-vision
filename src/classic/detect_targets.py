"""D5 · 目标检测：低对比度暗图 + 椒盐噪声 → 目标列表。

对外接口（tools/score_detect.py 按这个打分）：
    detect(image_path: str) -> list[dict]
    每个 dict: {"id", "cx", "cy", "w", "h", "angle", "area"}

角度约定（重要，别随手改）：
    - 长边永远是 w，短边永远是 h
    - angle 单位是度，范围 [-90, 90)，含义是「长边相对于水平轴的夹角」
    - 矩形整体转 90° 还是它自己，所以这个表示唯一到 ±90°
"""
from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass

@dataclass(frozen=True)
class Params:                      # 所有"你能调的"参数集中在这里
    median_k: int = 5              # 中值核（会强制取奇数）
    otsu_delta: int = 0            # 在 Otsu 算出的阈值上 ±（0 表示完全用 Otsu）
    open_k: int = 0                # 开运算核，0=不做
    close_k: int = 0               # 闭运算核，0=不做
    # 面积下限：碎点/反光 ≤200，最小目标 ≈17500。
    # ⚠️ 别贴着 17500 设！实测设成 17500 时，视频里最小目标的面积在 17498~17510
    # 之间波动（噪声 + mp4 压缩），会有 17/150 帧被筛掉。留 8 倍余量最稳。
    min_area: float = 2000.0
    use_clahe: bool = False

def find_targets(gray: np.ndarray, params: Params) -> tuple[list[dict], np.ndarray]:
    """输入灰度图 + 参数 → (目标列表, 二值掩码)。掩码是给调参窗口看的。"""
    # ① 去椒盐：medianBlur 的核必须是奇数；保险起见这里强制取奇数（k=1 等于不滤波）
    k_median = max(1, int(params.median_k) | 1)
    denoised = cv2.medianBlur(gray, k_median)

    # ② 可选：CLAHE 先把暗图的局部对比度拉开
    if params.use_clahe:
        denoised = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)

    # ③ Otsu：直方图上背景（≈26）和叶片（≈78）是两个峰，让它自己找谷底，不用猜阈值
    otsu_thr, mask = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # otsu_delta：在c上做 ± 微调（调参窗口的那个滑动条靠它）
    if params.otsu_delta:
        manual = int(np.clip(otsu_thr + params.otsu_delta, 0, 255))
        _, mask = cv2.threshold(denoised, manual, 255, cv2.THRESH_BINARY)

    # ④ 可选：开运算去碎点 / 闭运算补洞（核 >=3 才有意义，也强制取奇数）
    if params.open_k >= 3:
        k = int(params.open_k) | 1
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
    if params.close_k >= 3:
        k = int(params.close_k) | 1
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results: list[dict] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < params.min_area:
            continue                                   # 修复①：面积过滤，碎点在这里被扔掉

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue                                   # 修复②：退化轮廓（一条线）会让除法崩掉

        # 修复③：中心用 moments 质心，宽高/角度用 minAreaRect —— 两者本来就不是同一个量：
        #   · 质心 = 轮廓的"面积中心"，叶柄面积占比 <1%，几乎不影响它（实测误差 1.4px）
        #   · 最小外接矩形必须包住整个轮廓（含叶柄），它的中心会被叶柄拉偏 20px（实测）
        # 所以别图省事把中心也交给 minAreaRect —— 我试过，中心误差 1.4px → 20.3px。
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        _, _, w, h, angle = _canonical_rect(cv2.minAreaRect(cnt))

        # 修复④：归一化之后立刻写断言，别让「看起来对」骗过自己
        assert -90.0 <= angle < 90.0, f"角度越界: {angle}"
        assert w >= h, f"长边应记为 w: w={w}, h={h}"

        results.append(
            {
                "id": 0,                               # 下面按面积排序后统一编号
                "cx": round(float(cx), 2),
                "cy": round(float(cy), 2),
                "w": round(float(w), 2),
                "h": round(float(h), 2),
                "angle": round(float(angle), 2),
                "area": float(area),                   # 注意：轮廓像素面积，不是 w*h
            }
        )

    # 修复⑤之三：留个稳定的顺序（大的在前），id 可复现，调试时好对照
    results.sort(key=lambda d: d["area"], reverse=True)
    for i, item in enumerate(results, 1):
        item["id"] = i
    return results, mask


def detect(image_path: str, params: Params | None = None) -> list[dict]:
    """对外接口不变（评分器只认它）：只返回目标列表，不返回掩码。"""
    img = cv2.imread(image_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    targets, _mask = find_targets(gray, params or Params())
    return targets


def _canonical_rect(rect) -> tuple[float, float, float, float, float]:
    """把 cv2.minAreaRect 的返回值归一化。

    OpenCV 的角度约定随版本变（4.5.1 之后是 (0,90]，更早是 [-90,0)），
    而且 (w, h) 谁长谁短也不固定 —— 所以别背约定，写完归一化 + 断言。
    """
    (cx, cy), (w, h), ang = rect
    if w < h:                              # 让长边当 w
        w, h = h, w
        ang += 90.0                        # 交换边长等价于把矩形转 90°
    ang = ((ang + 90.0) % 180.0) - 90.0    # 归到 [-90, 90)
    return cx, cy, w, h, ang


if __name__ == "__main__":
    # 单张/多张手测入口（原来被注释掉了，调试时很需要）：
    #     python -m src.classic.detect_targets
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for name in ("low_contrast_leaf.png", "low_contrast_leaf_hard.png"):
        path = root / "data" / "samples" / name
        print(f"\n=== {name} ===")
        found = detect(str(path))
        print(f"检测到 {len(found)} 个目标")
        print(json.dumps(found, ensure_ascii=False, indent=2))
