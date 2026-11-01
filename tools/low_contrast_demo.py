#!/usr/bin/env python
"""造一张「暗 + 低对比度 + 椒盐噪声」的测试图，并按给定流程跑一遍做量化。

为什么自己造而不是找图：
    1. 三个特征可以精确控制（背景灰度、物体/背景差、噪声比例），可复现
    2. 同一场景能生成一份「无噪声」的干净版，用来对比去噪效果
    3. 能顺手量出每一步到底改变了什么，而不是凭眼睛说「好像好点了」

用法：
    python tools/low_contrast_demo.py             # 生成图片 + 跑流程 + 出对比图
    python tools/low_contrast_demo.py --regen     # 强制重新生成图片
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "samples"
IMG_PNG = OUT_DIR / "low_contrast_leaf.png"
IMG_CLEAN = OUT_DIR / "low_contrast_leaf_clean.png"
MONTAGE = ROOT / "docs" / "images" / "low-contrast-pipeline.png"

BG = 26          # 背景灰度（很暗）
SP_AMOUNT = 0.02  # 椒盐噪声比例：1% 盐 + 1% 椒

# 两档对比度：
#   mild —— 灰度差 52，勉强能过 Canny(50,150)，你的参数直接能跑出轮廓
#   hard —— 灰度差 22，Canny 一个边都找不到，必须自己调阈值（教学用）
VARIANTS = {
    "mild": {"leaf": 78, "file": "low_contrast_leaf.png"},
    "hard": {"leaf": 48, "file": "low_contrast_leaf_hard.png"},
}

# 三片叶子的真值（造图和出真值都用它，保证一致）
LEAVES = [
    ((225, 275), (140, 58), 25),
    ((470, 195), (115, 48), -18),
    ((565, 455), (150, 55), 12),
]


# --------------------------------------------------------------------------- #
# 1. 造图
# --------------------------------------------------------------------------- #
def make_scene(w: int = 800, h: int = 600, seed: int = 7, leaf: int = 78) -> np.ndarray:
    """暗背景 + 三片低对比度叶子，外加轻微渐变与暗角。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # 背景：暗灰 + 缓慢渐变（避免纯色导致直方图只有一个峰）
    grad = (xx / w - 0.5) * 6 + (yy / h - 0.5) * 6
    base = BG + grad

    # 暗角：四周再暗一点，模拟镜头
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    base = base * (1.0 - 0.18 * np.clip(r, 0, 1) ** 2)

    canvas = base.astype(np.uint8)

    # 三片叶子（椭圆 + 叶柄），亮度只比背景高一点点
    leaves = LEAVES
    for (cx, cy), (ax, ay), ang in leaves:
        layer = canvas.copy()
        cv2.ellipse(layer, (cx, cy), (ax, ay), ang, 0, 360, leaf, -1)
        # 叶脉：比叶片再亮 6，几乎看不见 —— 低对比度下正是「细节丢失」的典型
        rad = np.deg2rad(ang)
        p1 = (int(cx - np.cos(rad) * ax * 0.9), int(cy - np.sin(rad) * ax * 0.9))
        p2 = (int(cx + np.cos(rad) * ax * 0.9), int(cy + np.sin(rad) * ax * 0.9))
        cv2.line(layer, p1, p2, leaf + 6, 2)
        # 叶柄
        cv2.line(layer, (cx, cy), (int(cx + np.cos(rad) * (ax + 40)),
                                   int(cy + np.sin(rad) * (ax + 40))), leaf - 4, 3)
        canvas = np.maximum(canvas, layer)

    rng = np.random.default_rng(seed)
    canvas = np.clip(canvas.astype(np.int16) + rng.integers(-4, 5, canvas.shape), 0, 255).astype(np.uint8)
    return canvas


def add_salt_pepper(gray: np.ndarray, amount: float = SP_AMOUNT, seed: int = 7) -> np.ndarray:
    """按比例随机撒纯白（盐）和纯黑（椒）。"""
    rng = np.random.default_rng(seed + 1)
    out = gray.copy()
    total = gray.size
    n_salt = int(total * amount / 2)
    n_pepper = int(total * amount / 2)
    ys = rng.integers(0, gray.shape[0], n_salt + n_pepper)
    xs = rng.integers(0, gray.shape[1], n_salt + n_pepper)
    out[ys[:n_salt], xs[:n_salt]] = 255
    out[ys[n_salt:], xs[n_salt:]] = 0
    return out


def write_truth(png: Path, leaf_gray: int) -> Path:
    """把三片叶子的真值写成 JSON，供 tools/score_detect.py 打分用。"""
    import json

    data = {
        "image": png.name,
        "size": [800, 600],
        "bg_gray": BG,
        "leaf_gray": leaf_gray,
        "targets": [
            {
                "id": i + 1,
                "cx": float(cx),
                "cy": float(cy),
                "w": float(ax * 2),
                "h": float(ay * 2),
                "angle": float(ang),
                "area": float(np.pi * ax * ay),
            }
            for i, ((cx, cy), (ax, ay), ang) in enumerate(LEAVES)
        ],
    }
    path = png.parent / (png.stem + "_truth.json")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 2. 你的流程（逻辑一行没改，只是把中间结果留下来）
# --------------------------------------------------------------------------- #
def run_pipeline(img_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    denoised = cv2.medianBlur(gray, 3)                      # 1. 去椒盐
    denoised2 = cv2.GaussianBlur(denoised, (3, 3), 0)       # 2. 轻高斯

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised2)                       # 3. CLAHE

    blur2 = cv2.GaussianBlur(enhanced, (5, 5), 0)           # 4. 再高斯
    edges = cv2.Canny(blur2, 50, 150)                       # 5. Canny

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)  # 6. 闭运算连断边

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 对照组 1：不做 CLAHE，直接 Canny，看看会不会漏
    edges_no_clahe = cv2.Canny(cv2.GaussianBlur(denoised2, (5, 5), 0), 50, 150)

    # 对照组 2：阈值降到 (15, 45)，验证「找不到边」到底是阈值问题还是图没信息
    edges_low = cv2.Canny(blur2, 15, 45)
    mask_low = cv2.morphologyEx(edges_low, cv2.MORPH_CLOSE, kernel)
    contours_low, _ = cv2.findContours(mask_low, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return {
        "gray": gray,
        "denoised": denoised2,
        "enhanced": enhanced,
        "edges": edges,
        "edges_no_clahe": edges_no_clahe,
        "edges_low": edges_low,
        "mask": mask,
        "contours": contours,
        "contours_low": contours_low,
    }


# --------------------------------------------------------------------------- #
# 3. 出图 + 打印数字
# --------------------------------------------------------------------------- #
def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_montage(img: np.ndarray, res: dict) -> np.ndarray:
    tiles = [
        (cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img, "input (dark + low contrast + S&P)"),
        (res["denoised"], "median(3) + gauss(3)"),
        (res["enhanced"], "CLAHE(clip=2.0, tile=8)"),
        (res["edges"], "Canny(50,150)"),
        (res["mask"], "close 3x3"),
        (None, "contours found"),
    ]
    overlay = (cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy())
    cv2.drawContours(overlay, res["contours"], -1, (0, 255, 0), 2)
    tiles[-1] = (overlay, "contours found (RETR_EXTERNAL)")

    cells = []
    for image, text in tiles:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cells.append(label(cv2.resize(image, (400, 300), interpolation=cv2.INTER_AREA), text))

    top = np.hstack(cells[:3])
    bottom = np.hstack(cells[3:])
    return np.vstack([top, bottom])


def main() -> int:
    ap = argparse.ArgumentParser(description="生成低对比度暗图并跑流程")
    ap.add_argument("--regen", action="store_true", help="强制重新生成图片")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MONTAGE.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for name, cfg in VARIANTS.items():
        png = OUT_DIR / cfg["file"]
        clean_path = OUT_DIR / (png.stem + "_clean.png")

        if args.regen or not png.exists():
            clean = make_scene(leaf=cfg["leaf"])
            noisy = add_salt_pepper(clean)
            cv2.imwrite(str(clean_path), clean)
            cv2.imwrite(str(png), noisy)
            cv2.imwrite(str(png.with_suffix(".jpg")), noisy, [cv2.IMWRITE_JPEG_QUALITY, 88])
            write_truth(png, cfg["leaf"])

        gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        res = run_pipeline(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        areas = sorted((cv2.contourArea(c) for c in res["contours"]), reverse=True)
        results.append({"name": name, "cfg": cfg, "png": png, "img": gray, "res": res, "areas": areas})

    print("生成的文件：")
    for r in results:
        cfg = r["cfg"]
        print(f"  {r['png'].relative_to(ROOT)}   背景 {BG} → 叶片 {cfg['leaf']}（差 {cfg['leaf'] - BG}）")
        print(f"  {r['png'].with_suffix('.jpg').relative_to(ROOT)}   JPEG 版（多一层压缩噪点）")
        print(f"  {(OUT_DIR / (r['png'].stem + '_clean.png')).relative_to(ROOT)}   同场景无噪声版")

    print("\n—— 输入图特征 ——")
    for r in results:
        img = r["img"]
        n_white = int((img >= 250).sum())
        n_black = int((img <= 5).sum())
        print(f"[{r['name']}] 均值 {img.mean():5.1f}　标准差 {img.std():5.1f}　"
              f"最暗 {img.min()}　最亮 {img.max()}　"
              f"纯白 {n_white / img.size:.2%} + 纯黑 {n_black / img.size:.2%}（椒盐）")

    print("\n—— 各阶段效果对比 ——")
    head = f"{'档位':<6}{'去噪后std':>10}{'CLAHE后std':>11}{'边缘(有CLAHE)':>14}{'边缘(无CLAHE)':>14}{'轮廓数':>8}{'>800':>7}{'<=50':>7}"
    print(head)
    print("-" * len(head))
    for r in results:
        res = r["res"]
        edges = int((res["edges"] > 0).sum())
        edges0 = int((res["edges_no_clahe"] > 0).sum())
        big = sum(1 for a in r["areas"] if a > 800)
        small = sum(1 for a in r["areas"] if a <= 50)
        print(f"{r['name']:<6}{res['denoised'].std():>10.1f}{res['enhanced'].std():>11.1f}"
              f"{edges:>14}{edges0:>14}{len(r['areas']):>8}{big:>7}{small:>7}")

    print("\n—— 轮廓面积（前 8 个）——")
    for r in results:
        print(f"[{r['name']}] {[int(a) for a in r['areas'][:8]]}")

    print("\n—— 如果把 Canny 阈值从 (50,150) 降到 (15,45) ——")
    for r in results:
        res = r["res"]
        low_areas = sorted((cv2.contourArea(c) for c in res["contours_low"]), reverse=True)
        print(f"[{r['name']}] 边缘点 {int((res['edges_low'] > 0).sum())}　"
              f"轮廓 {len(low_areas)} 个　面积前 5: {[int(a) for a in low_areas[:5]]}")

    mild = next(r for r in results if r["name"] == "mild")
    cv2.imwrite(str(MONTAGE), build_montage(mild["img"], mild["res"]))
    print(f"\n已保存流程对比图：{MONTAGE.relative_to(ROOT)}")
    print("\n提示：mild 版用你现在的参数就能出轮廓；hard 版必须自己调 Canny 阈值 / 换更大中值核。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
