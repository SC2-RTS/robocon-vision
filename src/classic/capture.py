from dataclasses import dataclass
import time

import cv2
import numpy as np


@dataclass
class Frame:                      # ① 数据袋子，不需要方法
    image: np.ndarray             # BGR，原始分辨率
    index: int                    # 帧号（含被 step 跳过的帧）
    t: float                      # 时间戳（秒）


class FrameSource:
    def __init__(self, cfg: dict) -> None:
        src = cfg["source"]
        self._kind = src["kind"]
        self._step = max(1, int(src.get("step", 1)))
        self._cap: cv2.VideoCapture | None = None
        self._image: np.ndarray | None = None
        self._next_index = 0
        self._t0: float | None = None

        # TODO 你填：按 kind 初始化资源，失败要"抛错"而不是返回 None
        #   image  → self._image = cv2.imread(path)；注意 imread 失败只返回 None、不抛异常
        #   video  → self._cap = cv2.VideoCapture(path)；再查 isOpened()
        #   camera → self._cap = cv2.VideoCapture(int(path))；同上
        #   其它   → raise ValueError(f"未知的 source.kind: {kind}")

        if self._kind == 'image':
            path = src["path"]
            self._image = cv2.imread(path)
            if self._image is None:
                raise FileNotFoundError("文件不存在")
            
        elif self._kind == 'video':
            path = src["path"]
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise IOError("无法打开视频文件")
            
        elif self._kind == 'camera':
            self._cap = cv2.VideoCapture(int(src["path"]))
            if not self._cap.isOpened():
                raise IOError("无法打开摄像头")
        else:
            raise ValueError(f"未知的 source.kind: {self._kind}")

        self._t0 = time.perf_counter()

    def __enter__(self) -> "FrameSource":      # ③ with 协议
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._cap is not None:
            self._cap.release()                # 无论正常/异常都会走到这里

        return False                           # 不吞异常

    def read(self) -> Frame | None:
        # TODO 你填：
        #   image 源：第一次返回 Frame(self._image, 0, 0.0)；第二次起返回 None
        #   video/camera：先空读 (step-1) 次，第 step 次才产出
        #   读不到 → 返回 None（"放完了"和"摄像头断了"都是 None）
        
        if self._kind == 'image':
            if self._next_index == 0:
                self._next_index += 1
                return Frame(self._image, 0, 0.0)
            else:
                return None

        elif self._kind in ['video', 'camera']:
            if self._cap is None:
                return None
            
            ret, frame = self._cap.read()          # 先读 1 帧（就是 index 这一帧）
            if not ret:
                return None
            index = self._next_index
            self._next_index += self._step

            # 获取时间戳
            if self._kind == "video":
                t = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            else:                                   # camera
                t = time.perf_counter() - self._t0  # 用 perf_counter，别用 time.time()

            for _ in range(self._step - 1):        # 读完再跳过剩下的 step-1 帧
                if not self._cap.grab():
                    return None
            
            return Frame(frame, index, t)

    @property                                   # ② 用起来像字段
    def fps(self) -> float:
        if self._cap is None:
            return 0.0
        
        return float(self._cap.get(cv2.CAP_PROP_FPS))
    