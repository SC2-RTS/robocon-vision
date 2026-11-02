#!/usr/bin/env python
"""capture.py 的验收脚本：把 docs/d6_spec.md 里对 capture 的判据变成断言。

为什么要这个脚本（而不是手敲几条命令）：
    1. 手测只会测"快乐路径"（step=1），bug 全藏在你没拨的配置里
    2. 每次改完重新跑，10 秒出结果，不用回忆上次测了啥

用法：
    python tools/check_capture.py
退出码：0 = 全过；1 = 有失败
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classic.capture import FrameSource  # noqa: E402

VIDEO = ROOT / "data" / "samples" / "wobble.mp4"
IMAGE = ROOT / "data" / "samples" / "low_contrast_leaf.png"
VIDEO_FRAMES = 150
VIDEO_FPS = 30.0
T_TOL = 0.002  # 时间戳容差（秒）

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f" {'✓' if ok else '✗'} {name:<40} {detail}")
    if not ok:
        FAILED.append(name)


def cfg(kind: str, path, step: int = 1) -> dict:
    return {"source": {"kind": kind, "path": str(path), "step": step}}


def expect_raise(name: str, fn) -> None:
    try:
        fn()
    except (FileNotFoundError, IOError, ValueError, OSError) as exc:
        check(name, True, type(exc).__name__)
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"抛了 {type(exc).__name__}（类型不对）")
    else:
        check(name, False, "没抛错 ← 会静默跑 0 帧")


def main() -> int:
    print("—— ① image 源 ——")
    with FrameSource(cfg("image", IMAGE)) as s:
        f1 = s.read()
        f2 = s.read()
        fps_img = s.fps
    check("只产出 1 帧（第 2 次 read 是 None）", f1 is not None and f2 is None,
          f"第1次 index={f1.index if f1 else None}，第2次={f2}")
    check("帧是 BGR 三通道", f1 is not None and f1.image.ndim == 3 and f1.image.shape[2] == 3,
          f"shape={f1.image.shape if f1 else None}")
    check("image 源 fps = 0", fps_img == 0.0, f"{fps_img}")

    print("\n—— ② video 源，step=1 ——")
    with FrameSource(cfg("video", VIDEO)) as s:
        fps = s.fps
        frames = [s.read() for _ in range(5)]
    check("fps ≈ 30", abs(fps - VIDEO_FPS) < 0.5, f"{fps}")
    check("index 连续 0..4", [f.index for f in frames] == [0, 1, 2, 3, 4],
          f"{[f.index for f in frames]}")
    ok_t = all(abs(f.t - f.index / VIDEO_FPS) < T_TOL for f in frames)
    check("t == index / fps（视频时间轴）", ok_t, f"{[round(f.t, 3) for f in frames]}")

    print("\n—— ③ video 源，step=3（关键：这里才暴露 bug）——")
    with FrameSource(cfg("video", VIDEO, 3)) as s:
        fs3 = [s.read() for _ in range(3)]
    check("index = 0, 3, 6", [f.index for f in fs3] == [0, 3, 6], f"{[f.index for f in fs3]}")
    ok_t = all(abs(f.t - f.index / VIDEO_FPS) < T_TOL for f in fs3)
    check("t == index / fps", ok_t, f"{[round(f.t, 3) for f in fs3]}")

    ref = cv2.VideoCapture(str(VIDEO))
    same = True
    for f in fs3:
        ref.set(cv2.CAP_PROP_POS_FRAMES, f.index)
        ok, r = ref.read()
        same = same and ok and np.array_equal(f.image, r)
    ref.release()
    check("画面 = 该 index 的真实帧（逐像素）", same)

    print("\n—— ④ 边界 ——")
    with FrameSource(cfg("video", VIDEO)) as s:
        n = 0
        while s.read() is not None:
            n += 1
    check(f"读完后返回 None，共 {VIDEO_FRAMES} 帧", n == VIDEO_FRAMES, f"实际 {n}")

    with FrameSource(cfg("video", VIDEO, 1000)) as s:
        big = s.read()
    check("step 大于总帧数 → 返回 None，不崩", big is None, f"{big}")

    expect_raise("图片路径错 → 抛错", lambda: FrameSource(cfg("image", ROOT / "nope.png")))
    expect_raise("视频路径错 → 抛错", lambda: FrameSource(cfg("video", ROOT / "nope.mp4")))
    expect_raise("未知 kind → 抛 ValueError", lambda: FrameSource(cfg("rtsp", "x")))

    print("\n—— ⑤ camera 源（没有摄像头就算跳过）——")
    try:
        with FrameSource(cfg("camera", 0)) as s:
            f = s.read()
        check("camera 能读到帧", f is not None, f"index={f.index if f else None}, t={f.t if f else None}")
        check("camera 的 t 是墙钟（≥0 且单调）", f is not None and f.t >= 0.0, f"{f.t if f else None}")
    except (IOError, OSError) as exc:
        print(f" - 跳过：本机没有可用摄像头（{type(exc).__name__}）")
    except Exception as exc:  # noqa: BLE001
        check("camera 分支不应抛非 IO 错误", False, f"{type(exc).__name__}: {exc}")

    print()
    if FAILED:
        print(f"✗ {len(FAILED)} 项失败：")
        for name in FAILED:
            print("   -", name)
        return 1
    print("✓ 全部通过，capture.py 符合规格")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
