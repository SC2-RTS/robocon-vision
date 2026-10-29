"""D4 三个真实小应用：中位数平滑 / TopK 筛选 / 丢帧策略。

运行：
    python src\\utils\\data_structure\\apps.py
    python -m src.utils.data_structure.apps

三个都对应计划里的原话：
    1. 用 RingBuffer(maxlen=5) 存最近 5 帧检测框，取中位数输出（消除抖动）
    2. 用 MinHeap 从 20 个候选框里取置信度 Top3
    3. 用 deque(maxlen=1) 演示"只保留最新帧"的丢帧策略
"""
from __future__ import annotations

import random
import statistics
from collections import deque
from dataclasses import dataclass

try:  # 作为包的一部分被 import
    from ._loader import load_local
except ImportError:  # 直接运行本文件
    from _loader import load_local

# 复用你自己写的实现（不重复造轮子）
RingBuffer = load_local("queue.py", "RingBuffer")
MinHeap = load_local("heap.py", "MinHeap")


# --------------------------------------------------------------------------- #
# 公共小工具
# --------------------------------------------------------------------------- #
@dataclass
class Box:
    """一个检测框：中心点 + 宽高 + 置信度。"""

    cx: float
    cy: float
    w: float
    h: float
    conf: float

    def __lt__(self, other: "Box") -> bool:
        """MinHeap 靠它比较大小 —— 只按置信度比。"""
        return self.conf < other.conf


def ring_snapshot(ring) -> list:
    """按写入顺序取出 RingBuffer 里现存的数据（不消费）。优先用它自己的 to_list()。"""
    if hasattr(ring, "to_list"):
        return ring.to_list()
    return [ring.buffer[(ring.head + i) % ring.capacity] for i in range(ring.size)]


# --------------------------------------------------------------------------- #
# 应用 1：多帧中位数平滑
# --------------------------------------------------------------------------- #
def median_smooth(history: list[Box]) -> Box:
    """对最近 N 帧的每个字段取中位数。

    为什么用中位数而不是平均数：中位数对离群值不敏感。
    某一帧跳出一个离谱的框，平均数会被直接拽走，中位数几乎不动。
    """
    return Box(
        statistics.median(b.cx for b in history),
        statistics.median(b.cy for b in history),
        statistics.median(b.w for b in history),
        statistics.median(b.h for b in history),
        statistics.median(b.conf for b in history),
    )


def demo_median_smooth() -> None:
    random.seed(7)
    ring = RingBuffer(5)  # 只保留最近 5 帧
    truth = (320.0, 240.0)

    print("【应用 1】RingBuffer(5) + 中位数平滑：抑制检测框抖动")
    print(f"{'帧':>4} {'原始中心':>17} {'原始误差':>9} {'平滑后中心':>17} {'平滑误差':>9}")

    for frame in range(9):
        noise = 45.0 if frame == 6 else 6.0  # 第 6 帧故意塞一个离群值
        raw = Box(
            truth[0] + random.uniform(-noise, noise),
            truth[1] + random.uniform(-noise, noise),
            80.0,
            120.0,
            0.90 + random.uniform(-0.03, 0.03),
        )
        ring.write(raw)
        smooth = median_smooth(ring_snapshot(ring))
        err_raw = ((raw.cx - truth[0]) ** 2 + (raw.cy - truth[1]) ** 2) ** 0.5
        err_smooth = ((smooth.cx - truth[0]) ** 2 + (smooth.cy - truth[1]) ** 2) ** 0.5

        print(
            f"{frame:>4} {f'({raw.cx:5.1f},{raw.cy:5.1f})':>17} {err_raw:>7.1f}px "
            f"{f'({smooth.cx:5.1f},{smooth.cy:5.1f})':>17} {err_smooth:>7.1f}px"
        )

    print("→ 第 6 帧那个离谱的框被中位数吃掉了；队列里始终只有 5 帧，内存不随帧数增长。")
    print(f"→ 队列容量 {ring.capacity}，写入 9 帧后仍只保存 {len(ring)} 帧，队列值: {ring.to_list()}\n")


# --------------------------------------------------------------------------- #
# 应用 2：MinHeap 取 TopK
# --------------------------------------------------------------------------- #
def topk_by_confidence(boxes: list[Box], k: int) -> list[Box]:
    """O(n log k)：维持一个大小 ≤ k 的最小堆，堆里就是最大的 k 个。

    关键点：这里要"留下大的"，所以用**最小堆** —— 堆顶是当前最小的那个，
    一旦超过 k 个就把堆顶踢掉，剩下的自然就是最大的 k 个。
    """
    heap = MinHeap()
    for box in boxes:
        heap.push(box)
        if heap.size() > k:
            heap.pop()  # 踢掉当前置信度最低的

    picked = []
    while heap.size() > 0:
        picked.append(heap.pop())
    picked.reverse()  # 最小堆弹出来是升序，反转成从大到小
    return picked


def demo_topk() -> None:
    random.seed(11)
    boxes = [
        Box(random.uniform(0, 640), random.uniform(0, 480), 40.0, 40.0, random.uniform(0.20, 0.95))
        for _ in range(20)
    ]
    picked = topk_by_confidence(boxes, 3)
    brute = sorted(boxes, key=lambda b: b.conf, reverse=True)[:3]

    assert [round(b.conf, 6) for b in picked] == [round(b.conf, 6) for b in brute], "堆法结果与暴力排序不一致"

    print("【应用 2】MinHeap 取 Top3（20 个候选框，堆大小只占 3）")
    print(f"{'序号':>4} {'置信度':>7} {'中心':>15} 选中")
    for i, box in enumerate(boxes, 1):
        mark = "★" if box in picked else ""
        print(f"{i:>4} {box.conf:>7.3f} {f'({box.cx:.0f},{box.cy:.0f})':>15}  {mark}")

    print(f"\nTop3       → {[f'{b.conf:.3f}' for b in picked]}")
    print(f"暴力排序对照 → {[f'{b.conf:.3f}' for b in brute]}（一致 ✓）")
    print("→ 20 个候选只维护 3 个元素的堆：O(n log k)，而不是把 20 个全排序 O(n log n)。")
    print("→ 比赛里 n 可能是几十上百个候选，k 通常只要 1~3 个目标，差距会很明显。\n")


# --------------------------------------------------------------------------- #
# 应用 3：丢帧策略
# --------------------------------------------------------------------------- #
def _run_stream(bounded: bool, ticks: int = 8, per_tick: int = 3):
    """模拟：相机每个周期产 per_tick 帧，推理每个周期只处理 1 帧。"""
    slot: deque = deque(maxlen=1 if bounded else None)
    fid = 0
    rows = []
    for tick in range(1, ticks + 1):
        for _ in range(per_tick):
            slot.append(fid)  # maxlen=1 时旧帧被自动挤掉
            fid += 1
        # 注意用 popleft：无界队列时它取的是**最旧**的帧（真正的排队），
        # 有界 maxlen=1 时队列里只有一个元素，popleft 就是取最新的那个。
        got = slot.popleft()
        rows.append((tick, got, fid - 1 - got, len(slot)))  # (周期, 处理帧, 帧龄, 剩余积压)
    return rows, fid


def demo_drop_frame() -> None:
    ticks, per_tick = 8, 3
    rows_b, total = _run_stream(True, ticks, per_tick)
    rows_u, _ = _run_stream(False, ticks, per_tick)

    print("【应用 3】deque(maxlen=1) 只保留最新帧：丢帧 vs 排队")
    print(f"相机 {ticks} 个周期共产生 {total} 帧，推理只处理得动 {ticks} 帧\n")

    print("① 有界 deque(maxlen=1)：旧帧被自动挤掉")
    print(f"{'推理周期':>8} {'处理帧':>7} {'帧龄':>6} {'积压':>6}")
    for tick, got, age, backlog in rows_b:
        print(f"{tick:>8} {got:>7} {age:>6} {backlog:>6}")
    print(f"→ 丢掉 {total - ticks} 帧，但每次处理的都是最新画面，帧龄恒为 0，内存恒定\n")

    print("② 无界 deque()：全部排队，处理的是越来越老的帧")
    print(f"{'推理周期':>8} {'处理帧':>7} {'帧龄':>6} {'积压':>6}")
    for tick, got, age, backlog in rows_u:
        print(f"{tick:>8} {got:>7} {age:>6} {backlog:>6}")
    print("→ 积压越来越大，延迟没有上界：第 8 个周期处理的还是第 7 帧，比现实晚了 16 帧")
    print("→ 机械臂按 16 帧前的位置去抓，目标早跑了。所以实时视觉必须丢帧，不能排队\n")


if __name__ == "__main__":
    demo_median_smooth()
    demo_topk()
    demo_drop_frame()
