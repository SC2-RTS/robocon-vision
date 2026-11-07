"""第 7 天：手写 [1,5,8400] 的解码 + NMS，并用 [1,300,6] 当"参考答案"自检。

这个文件是**基线版本**：跑通它，再改 3 个参数观察变化。
不要从零重写，先看懂这 5 步，然后动手改（改法见文件末尾的「三个练习」）。

整个后处理分 5 步：
    ① 预处理：letterbox 到 640×640 → BGR→RGB → HWC→CHW → /255
    ② 推理：sess.run（onnxruntime 就这一句）
    ③ 解码：[1,5,8400] 转置成 (8400,5) → 拆 cx/cy/w/h/score → 置信度过滤 → 还原到原图坐标
    ④ NMS：按 conf 降序，和"已保留的框"比 IoU，超过阈值就丢
    ⑤ 自检：和 [1,300,6] 的结果配对算 IoU（>0.9 才算解码正确）

用法：
    python experiments/yolo/decode_nms.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "runs" / "detect" / "yolo26n-leaves" / "weights"
ONNX_RAW = W / "yolo26_1x5x8400.onnx"     # 要自己解码 + NMS 的
ONNX_E2E = W / "yolo26_1x300x6.onnx"      # 参考答案（不用解码）
IMG = ROOT / "datasets" / "leaves" / "images" / "val" / "frame_0120.jpg"

CONF = 0.5      # ← 练习 1：改这个
IOU = 0.5       # ← 练习 2：改这个
PAD = 114       # letterbox 填充值，YOLO 训练时就用的 114，别改成 0
SIZE = 640


# ---------------------------------------------------------------- ① 预处理
def letterbox(img: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """等比缩放 + 居中灰边填充。返回 (画布, 缩放比, x 偏移, y 偏移)。"""
    h, w = img.shape[:2]
    r = min(SIZE / h, SIZE / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = np.full((SIZE, SIZE, 3), PAD, np.uint8)
    dx, dy = (SIZE - nw) // 2, (SIZE - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh))
    return canvas, r, dx, dy


def to_input(canvas: np.ndarray) -> np.ndarray:
    """BGR→RGB、HWC→CHW、/255、加 batch 维。四件事一个都不能少。"""
    return canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0


# ---------------------------------------------------------------- ③ 解码
def decode_raw(out: np.ndarray, r: float, dx: int, dy: int, conf_thr: float):
    """把 [1,5,8400] 解成原图坐标下的 xyxy + 分数。"""
    p = out[0].T                      # (8400, 5)：每行一个候选  ← 转置是必须的
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    score = p[:, 4]                   # 已经是 0~1，不要再 sigmoid

    keep = score > conf_thr           # 先过滤！否则后面 NMS 要算 8400² 次
    cx, cy, w, h, score = cx[keep], cy[keep], w[keep], h[keep], score[keep]

    # 中心点+宽高 → 左上右下，同时从画布坐标还原到原图坐标
    x1 = (cx - w / 2 - dx) / r
    y1 = (cy - h / 2 - dy) / r
    x2 = (cx + w / 2 - dx) / r
    y2 = (cy + h / 2 - dy) / r
    return np.stack([x1, y1, x2, y2], axis=1), score


# ---------------------------------------------------------------- ④ NMS
def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """按分数降序，只和"已保留的框"比 IoU。"""
    order = np.argsort(-scores)       # ① 降序
    keep: list[int] = []
    for i in order:
        if all(iou(boxes[i], boxes[k]) <= iou_thr for k in keep):   # ② 只比已保留的
            keep.append(i)
    return keep


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="[1,5,8400] 解码 + NMS，并用 [1,300,6] 自检")
    ap.add_argument("--conf", type=float, default=CONF, help="置信度阈值（默认 0.5）")
    ap.add_argument("--iou", type=float, default=IOU, help="NMS 的 IoU 阈值（默认 0.5）")
    ap.add_argument("--no-nms", action="store_true", help="跳过 NMS，看有多少重复框")
    ap.add_argument("--image", default=str(IMG))
    args = ap.parse_args()
    conf, iou_thr = args.conf, args.iou

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"读不到图：{args.image}")

    sess_raw = ort.InferenceSession(str(ONNX_RAW), providers=["CPUExecutionProvider"])
    sess_e2e = ort.InferenceSession(str(ONNX_E2E), providers=["CPUExecutionProvider"])
    name = sess_raw.get_inputs()[0].name

    # ① 预处理（计时）
    t0 = time.perf_counter()
    canvas, r, dx, dy = letterbox(img)
    x = to_input(canvas)
    t_pre = (time.perf_counter() - t0) * 1000

    # ② 推理（计时，先预热一次）
    sess_raw.run(None, {name: x})
    t0 = time.perf_counter()
    out_raw = sess_raw.run(None, {name: x})[0]
    t_inf = (time.perf_counter() - t0) * 1000

    # ③④ 解码 + NMS（计时）
    t0 = time.perf_counter()
    boxes, scores = decode_raw(out_raw, r, dx, dy, conf)
    n_cand = len(boxes)
    if args.no_nms:
        keep = list(range(len(boxes)))
    else:
        keep = nms(boxes, scores, iou_thr)
    boxes, scores = boxes[keep], scores[keep]
    t_post = (time.perf_counter() - t0) * 1000

    tag = "（跳过 NMS）" if args.no_nms else f"NMS(iou={iou_thr})"
    print(f"图像 {Path(args.image).name}  {img.shape[1]}x{img.shape[0]}  →  缩放 {r:.3f}  偏移 ({dx},{dy})")
    print(f"\n[1,5,8400]  conf>{conf} 后剩 {n_cand} 个候选，{tag} 后剩 {len(boxes)} 个：")
    for i, (b, s) in enumerate(zip(boxes, scores)):
        print(f"  [{i}] conf={s:.3f}  ({b[0]:.0f},{b[1]:.0f})-({b[2]:.0f},{b[3]:.0f})")

    # ⑤ 参考答案：端到端模型不用解码
    out_e2e = sess_e2e.run(None, {name: x})[0][0]
    e2e = out_e2e[out_e2e[:, 4] > conf]
    ref = np.stack([(e2e[:, 0] - dx) / r, (e2e[:, 1] - dy) / r,
                    (e2e[:, 2] - dx) / r, (e2e[:, 3] - dy) / r], axis=1)
    print(f"\n[1,300,6] 参考答案：{len(ref)} 个框")
    for i, (b, s) in enumerate(zip(ref, e2e[:, 4])):
        print(f"  [{i}] conf={s:.3f}  ({b[0]:.0f},{b[1]:.0f})-({b[2]:.0f},{b[3]:.0f})")

    print("\n自检：我解码出的每个框，和参考答案里最像的框的 IoU")
    worst = 1.0
    for i, b in enumerate(boxes):
        best = max(iou(b, q) for q in ref) if len(ref) else 0.0
        worst = min(worst, best)
        flag = "OK" if best > 0.9 else "低 ← 解码有问题"
        print(f"  我的框[{i}] -> 最匹配 IoU = {best:.3f}  {flag}")
    print(f"\n最差 IoU = {worst:.3f}   （判据：> 0.9 才算解码正确）")

    print(f"\n耗时：预处理 {t_pre:.2f} ms | 推理 {t_inf:.2f} ms | 解码+NMS {t_post:.2f} ms"
          f" | 合计 {t_pre + t_inf + t_post:.2f} ms")
    return 0 if worst > 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())

# =====================================================================
# 三个练习（改完重跑，对照下面的"实测答案"，把现象记进 docs/daily/第7天.md）
#
# 先看一个事实（实测，8400 个候选的分数分布）：
#   最高分 0.9745，>0.5 有 30 个，>0.9 有 20 个，>0.99 有 0 个
#
#   练习 1：CONF = 0.95   → 候选 14 个，NMS 后 3 个（正常）
#           CONF = 0.99   → 候选  0 个，NMS 后 0 个  ← 一个都检不出来！
#           原因：模型最高分只有 0.9745，阈值高于它 = 全部过滤掉。
#           ★ 这是新手最常犯的错："我要很准，就把阈值设到 0.99"。
#
#   练习 2：CONF=0.95 时，把 IOU 改成 0.99 → 6 个框
#           原因：重复框之间的 IoU ≈ 0.97 < 0.99，所以没被抑制。
#           ★ NMS 阈值设得太高 = 等于没做 NMS。
#           （反过来 IOU 设太小，会把两个真实目标误当成重复删掉）
#
#   练习 3：把 nms() 里 `if all(...)` 改成 `if True:`（等于跳过 NMS）
#           → 30 个框（conf>0.5 的候选全部保留）
#
# 另一个实测事实：IOU 在 0.3 ~ 0.95 之间，结果都是 3 个框。
#   说明这个场景很干净：重复框 IoU≈0.97，不同叶子 IoU≈0
#   中间那一大段阈值都是"安全区" —— 阈值不敏感是好事，不是坏事。
#
# 做完这三个，你就理解 NMS 在干什么了：
#   它不是在"找框"，是在"删重复的框"；
#   conf 决定"哪些候选有资格"，iou 决定"多像才算重复"。
# =====================================================================
