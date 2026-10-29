class MinHeap:
    def __init__(self):
        self.heap = []  # 底层用列表模拟完全二叉树

    def peek(self) -> int:
        if not self.heap:
            return None

        return self.heap[0]

    def size(self) -> int:
        return len(self.heap)

    def push(self, val: int):
        self.heap.append(val)
        self._sift_up(self.size() - 1)

    def _sift_up(self, index: int):
        while index > 0:
            parents = (index - 1) // 2

            if self.heap[index] < self.heap[parents]:
                self.heap[index], self.heap[parents] = self.heap[parents], self.heap[index]
                index = parents
            else:
                break

    def pop(self) -> int:
        if not self.heap:
            raise IndexError('堆为空')
        if self.size() == 1:
            return self.heap.pop()

        value = self.peek()
        self.heap[0] = self.heap.pop()
        self._sift_down(0)

        return value

    def _sift_down(self, index: int):
        n = self.size()
        while True:
            smallest = index
            left_child = 2 * index + 1
            right_child = 2 * index + 2

            if left_child < n and self.heap[left_child] < self.heap[smallest]:
                smallest = left_child
            if right_child < n and self.heap[right_child] < self.heap[smallest]:
                smallest = right_child
            if smallest == index:
                break

            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            index = smallest

if __name__ == "__main__":
    h = MinHeap()
    
    # 1. 乱序插入
    for x in [5, 2, 8, 1, 9, 3]:
        h.push(x)
        print(f"插入 {x} 后，底层数组状态: {h.heap}")

    print("-" * 30)
    
    # 2. 依次弹出堆顶
    print("开始依次弹出最小值:")
    while h.size() > 0:
        print(f"弹出: {h.pop()}, 剩余底层数组: {h.heap}")
