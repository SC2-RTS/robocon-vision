"""D4 容器自测：把每个容器的关键操作各跑一遍，并打印时间复杂度。

运行：
    python src\\utils\\data_structure\\selftest.py
    python -m src.utils.data_structure.selftest
"""
from __future__ import annotations

try:  # 作为包的一部分被 import
    from ._loader import load_local
except ImportError:  # 直接运行本文件
    from _loader import load_local

LinkedListStack = load_local("stack.py", "LinkedListStack")
LinkedListQueue, LinkedListDeque, RingBuffer = load_local(
    "queue.py", "LinkedListQueue", "LinkedListDeque", "RingBuffer"
)
MinHeap = load_local("heap.py", "MinHeap")

# 容器 / 操作 → 复杂度 → 为什么
COMPLEXITY = [
    ("Stack.push / pop", "O(1)", "头插、头删，不搬移任何数据"),
    ("Stack.peek", "O(1)", "直接看栈顶指针"),
    ("Queue.push（入队）", "O(1)", "尾插，靠 _tail 指针"),
    ("Queue.pop（出队）", "O(1)", "头删；对比 list.pop(0) 是 O(n)"),
    ("Deque 两端 push / pop", "O(1)", "双向链表两端都有指针"),
    ("Deque.peek_first / last", "O(1)", "两侧都能直接访问"),
    ("任意容器的 to_list", "O(n)", "必须从头走一遍"),
    ("RingBuffer.write / read", "O(1)", "取模回绕，数据永不搬移"),
    ("RingBuffer 满时覆盖", "O(1)", "丢最旧一条，读指针前移一格"),
    ("MinHeap.push / pop", "O(log n)", "上浮 / 下沉，最多走树高"),
    ("MinHeap.peek", "O(1)", "堆顶就是最小值"),
    ("MinHeap 建堆（逐个 push）", "O(n log n)", "你现在的做法；加个 build() 自底向下沉可降到 O(n)"),
    ("从 n 个候选取 TopK", "O(n log k)", "堆只维持 k 个元素，k 小时接近 O(n)"),
]

FAILED: list[str] = []


def expect(name: str, got, want) -> None:
    ok = got == want
    print(f" {'✓' if ok else '✗'} {name:<32} {got}")
    if not ok:
        FAILED.append(f"{name}: 期望 {want}，实际 {got}")


def print_complexity() -> None:
    print("—— 时间复杂度 ——")
    for op, cost, why in COMPLEXITY:
        print(f"  {op:<26} {cost:<10} {why}")
    print()


def main() -> int:
    print_complexity()

    print("—— 行为检查 ——")
    stack = LinkedListStack()
    for v in range(1, 6):
        stack.push(v)
    expect("Stack push 1..5 → to_list", stack.to_list(), [1, 2, 3, 4, 5])
    expect("Stack peek", stack.peek(), 5)
    expect("Stack pop", stack.pop(), 5)
    expect("Stack size", stack.size(), 4)

    queue = LinkedListQueue()
    for v in (1, 2, 3):
        queue.push(v)
    expect("Queue 先入先出", queue.pop(), 1)
    expect("Queue 剩余", queue.to_list(), [2, 3])

    deque_ = LinkedListDeque()
    deque_.push_last(2)
    deque_.push_first(1)
    deque_.push_last(3)
    expect("Deque 头尾各入队 → 顺序", deque_.to_list(), [1, 2, 3])
    expect("Deque pop_first", deque_.pop_first(), 1)
    expect("Deque pop_last", deque_.pop_last(), 3)
    expect("Deque peek_first", deque_.peek_first(), 2)

    ring = RingBuffer(3)
    for i in range(1, 6):
        ring.write(i)
    expect("RingBuffer 容量3写5个 → 依次读", [ring.read() for _ in range(3)], [3, 4, 5])
    expect("RingBuffer 读空后 len", len(ring), 0)

    heap = MinHeap()
    for v in (5, 3, 8, 1, 9, 2):
        heap.push(v)
    expect("MinHeap peek", heap.peek(), 1)
    expect("MinHeap 逐个 pop 应升序", [heap.pop() for _ in range(6)], [1, 2, 3, 5, 8, 9])
    expect("MinHeap 弹空后 size", heap.size(), 0)

    print()
    if FAILED:
        print(f"✗ {len(FAILED)} 项失败：")
        for item in FAILED:
            print("   -", item)
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
