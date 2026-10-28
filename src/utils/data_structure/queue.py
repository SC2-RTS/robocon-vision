from linked_list import ListNode, DoublyListNode
from typing import Any, List

class LinkedListQueue:
    """基于链表实现的队列"""

    def __init__(self):
        """构造方法"""
        self._head: ListNode | None = None
        self._tail: ListNode | None = None
        self._size: int = 0

    def size(self) -> int:
        """获取队列的长度"""
        return self._size

    def is_empty(self) -> bool:
        """判断队列是否为空"""
        return self._size == 0

    def push(self, value: int):
        """入队"""
        node = ListNode(value)

        if self._head is None:
            self._head = node
            self._tail = node
        else:
            self._tail.next = node
            self._tail = node

        self._size += 1

    def pop(self) -> int:
        """出队"""
        num = self.peek()
        self._head = self._head.next
        self._size -= 1

        return num

    def peek(self) -> int:
        """访问队头元素"""
        if self.is_empty():
            raise IndexError('队列为空')

        return self._head.value

    def to_list(self) -> list[int]:
        """转化为列表用于打印"""
        aerr = []
        temp = self._head
        while temp:
            aerr.append(temp.value)
            temp = temp.next

        return aerr

class LinkedListDeque:
    """基于双向链表实现的双向队列"""

    def __init__(self):
        """构造方法"""
        self._head: DoublyListNode | None = None
        self._tail: DoublyListNode | None = None
        self._size: int = 0

    def size(self) -> int:
        """获取队列的长度"""
        return self._size

    def is_empty(self) -> bool:
        """判断队列是否为空"""
        return self._size == 0

    def push(self, num: int, is_front: bool):
        """入队"""
        node = DoublyListNode(num)

        if self.is_empty():
            self._head = self._tail = node
        elif is_front:
            self._head.prev = node
            node.next = self._head
            self._head = node
        else:
            self._tail.next = node
            node.prev = self._tail
            self._tail = node

        self._size += 1
            
    def push_first(self, num: int):
        """从队头入队"""
        self.push(num, True)

    def push_last(self, num: int):
        """从队尾入队"""
        self.push(num, False)

    def pop(self, is_front: bool) -> int:
        """出队"""
        if self.is_empty():
            raise IndexError('双向队列为空')
        elif self._size == 1:
            num = self._head.value
            self._head = self._tail = None
        elif is_front:
            num = self._head.value
            fnext = self._head.next
            fnext.prev = None
            self._head.next = None
            self._head = fnext
        else:
            num = self._tail.value
            tprev = self._tail.prev
            tprev.next = None
            self._tail.prev = None
            self._tail = tprev

        self._size -= 1
        return num

    def pop_first(self) -> int:
        """从队头出队"""
        return self.pop(True)

    def pop_last(self) -> int:
        """从队尾出队"""
        return self.pop(False)

    def peek_first(self) -> int:
        """访问队头元素"""
        if self.is_empty():
            raise IndexError('双向队列为空')
        
        return self._head.value

    def peek_last(self) -> int:
        """访问队尾元素"""
        if self.is_empty():
            raise IndexError('双向队列为空')
        
        return self._tail.value

    def to_list(self) -> list[int]:
        """转化为列表用于打印"""
        aerr = []
        temp = self._head
        while temp:
            aerr.append(temp.value)
            temp = temp.next

        return aerr

class RingBuffer:
    def __init__(self, capacity: int):
        """
        初始化环形缓冲区
        :param capacity: 缓冲区的固定容量
        """
        if capacity <= 0:
            raise ValueError("缓冲区容量必须大于 0")
        
        self.capacity = capacity
        self.buffer = [None] * capacity  # 底层定长数组
        self.head = 0                    # 读指针：指向最旧的数据
        self.tail = 0                    # 写指针：指向下一个要写入的位置
        self.size = 0                    # 当前缓冲区内的数据量

    def write(self, data: Any) -> None:
        """
        写入数据（如果满了，自动覆盖最旧的数据）
        """
        if self.size == self.capacity:
            # 缓冲区已满，覆盖最旧的数据，读指针必须跟着往前移动一位
            self.head = (self.head + 1) % self.capacity
        else:
            # 缓冲区未满，数据量增加
            self.size += 1
            
        # 写入数据，并移动写指针
        self.buffer[self.tail] = data
        self.tail = (self.tail + 1) % self.capacity

    def read(self) -> Any:
        """
        读取并移除最旧的数据（先进先出 FIFO）
        """
        if self.size == 0:
            return None  # 或者抛出异常，这里选择返回 None
        
        # 取出数据
        data = self.buffer[self.head]
        self.buffer[self.head] = None  # 清除引用，帮助垃圾回收
        # 移动读指针
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        
        return data

    def to_list(self) -> List[Any]:
        """
        按从旧到新的顺序，返回当前缓冲区内的所有有效数据
        """
        # 从 head 开始，依次取 size 个元素
        return [self.buffer[(self.head + i) % self.capacity] for i in range(self.size)]

    def __len__(self) -> int:
        """支持 len(rb) 语法"""
        return self.size

    def __repr__(self) -> str:
        """方便在控制台打印查看"""
        return f"RingBuffer(capacity={self.capacity}, data={self.to_list()})"

def test_queue():
    queue = LinkedListQueue()
    for i in range(10):
        queue.push(i)

    print(queue.to_list())
    print(queue.pop())
    print(queue.to_list())

def test_deque():
    deque = LinkedListDeque()
    for i in range(10):
        deque.push_last(i)

    print(deque.to_list())
    print(deque.pop_last())
    print(deque.to_list())

    # 测试
    rb = RingBuffer(3)
    rb.write('A')
    rb.write('B')
    rb.write('C')
    print(rb.read())  # 输出 A
    rb.write('D')     # 此时缓冲区里有 B, C，再写 D，刚好填满
    rb.write('E')     # 缓冲区已满，覆盖最旧的 B。现在里面有 C, D, E
    print(rb.read())  # 输出 C
    print(rb.read())  # 输出 D

def test_ring_buffer():
    # 1. 初始化一个容量为 3 的环形缓冲区
    rb = RingBuffer(3)

    # 2. 写入 3 个数据（刚好填满）
    rb.write("A")
    rb.write("B")
    rb.write("C")
    print("填满后:", rb.to_list())  # 输出: ['A', 'B', 'C']

    # 3. 写入第 4 个数据（触发覆盖，最旧的 'A' 被挤掉）
    rb.write("D")
    print("覆盖后:", rb.to_list())  # 输出: ['B', 'C', 'D']

    # 4. 读取一个数据（取出最旧的 'B'）
    val = rb.read()
    print(f"读出的数据: {val}")     # 输出: B
    print("读取后:", rb.to_list())  # 输出: ['C', 'D']

    # 5. 再次写入两个数据（填满剩下的空间）
    rb.write("E")
    rb.write("F")
    print("再次写入后:", rb.to_list())

    print("底层数组实际状态:", rb.buffer) 

if __name__ == "__main__":
    # test_queue()
    # test_deque()
    test_ring_buffer()
