class MinHeap:
    def __init__(self):
        # 底层用一个列表（数组）来存储数据
        self.heap = []

    def push(self, val: int) -> None:
        """插入元素：放到数组末尾，然后上浮"""
        self.heap.append(val)
        # 从最后一个元素（新加入的元素）开始上浮
        self._sift_up(len(self.heap) - 1)

    def pop(self) -> int:
        """弹出堆顶元素（最小值）：将最后一个元素移到堆顶，然后下沉"""
        if not self.heap:
            raise IndexError("pop from an empty heap")
        
        # 如果只有一个元素，直接弹出
        if len(self.heap) == 1:
            return self.heap.pop()

        # 1. 记录堆顶元素（最小值）
        min_val = self.heap[0]
        
        # 2. 把最后一个元素移到堆顶，并删除最后一个元素
        self.heap[0] = self.heap.pop()
        
        # 3. 从堆顶（索引0）开始下沉
        self._sift_down(0)
        
        return min_val

    def peek(self) -> int:
        """查看堆顶元素但不弹出"""
        if not self.heap:
            raise IndexError("peek from an empty heap")
        return self.heap[0]

    def _sift_up(self, index: int) -> None:
        """上浮：与父节点比较，如果比父节点小，就交换，直到到达堆顶或满足堆性质"""
        parent_idx = (index - 1) // 2
        
        # 只要还没到根节点，且当前节点比父节点小，就交换
        while index > 0 and self.heap[index] < self.heap[parent_idx]:
            self.heap[index], self.heap[parent_idx] = self.heap[parent_idx], self.heap[index]
            
            # 更新当前节点索引为父节点索引，准备下一轮比较
            index = parent_idx
            parent_idx = (index - 1) // 2

    def _sift_down(self, index: int) -> None:
        """下沉：与左右子节点中较小的那个比较，如果比子节点大，就交换，直到叶子节点或满足堆性质"""
        n = len(self.heap)
        
        while True:
            # 假设当前节点是最小的
            smallest = index
            left_child = 2 * index + 1
            right_child = 2 * index + 2

            # 如果左子节点存在，且左子节点比当前最小节点还小
            if left_child < n and self.heap[left_child] < self.heap[smallest]:
                smallest = left_child

            # 如果右子节点存在，且右子节点比当前最小节点还小
            if right_child < n and self.heap[right_child] < self.heap[smallest]:
                smallest = right_child

            # 如果当前节点已经是最小的了，说明满足堆性质，停止下沉
            if smallest == index:
                break

            # 否则，将当前节点与最小的子节点交换
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            
            # 更新当前节点索引，继续下一轮下沉
            index = smallest

    def __len__(self) -> int:
        return len(self.heap)

    def __repr__(self) -> str:
        return f"MinHeap({self.heap})"

if __name__ == "__main__":
    min_heap = MinHeap()
    
    # 插入数据 (故意打乱顺序)
    data = [10, 5, 20, 2, 8, 15]
    for x in data:
        min_heap.push(x)
        print(f"插入 {x} 后，堆状态: {min_heap}")

    print("-" * 30)
    
    # 依次弹出堆顶元素（应该从小到大输出）
    print("依次弹出最小值:")
    while len(min_heap) > 0:
        print(min_heap.pop(), end=" ")
    print()
