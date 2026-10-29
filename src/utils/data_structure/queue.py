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
        if capacity <= 0:
            raise ValueError("容量必须大于 0")
        
        self.capacity = capacity
        self.buffer = [None] * capacity  # 定长数组
        self.head = 0                    # 读指针：永远指向最旧的数据
        self.tail = 0                    # 写指针：永远指向下一个要写入的空位
        self.size = 0                    # 当前有效数据量

    def write(self, data: Any) -> None:
        """写入数据（满则覆盖最旧的）"""
        if self.size == self.capacity:
            self.head = (self.head + 1) % self.capacity
        else:
            self.size += 1

        self.buffer[self.tail] = data
        self.tail = (self.tail + 1) % self.capacity

    def read(self) -> Any:
        """读取并移除最旧的数据"""
        if self.size == 0:
            return None

        value = self.buffer[self.head]
        self.buffer[self.head] = None
        self.head = (self.head + 1) % self.capacity
        self.size -= 1

        return value

    def to_list(self) -> List[Any]:
        """按从旧到新的顺序，返回当前缓冲区内的所有有效数据"""
        return [self.buffer[(self.head + i) % self.capacity] for i in range(self.size)]

    def __len__(self):
        return self.size

    def __repr__(self):
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

def test_ring_buffer():
    rb = RingBuffer(3)

    # 1. 测试定长与正常写入
    rb.write("A")
    rb.write("B")
    print("1. 正常写入后:", rb.to_list())  # 输出: ['A', 'B']

    # 2. 测试写满
    rb.write("C")
    print("2. 写满后:", rb.to_list())      # 输出: ['A', 'B', 'C']

    # 3. 测试覆盖（满了再写，A被挤掉）
    rb.write("D")
    print("3. 覆盖后:", rb.to_list())      # 输出: ['B', 'C', 'D']

    # 4. 测试读取与后续写入
    rb.read()  # 读走 B
    rb.write("E")  # 覆盖 C 的位置，逻辑上追加在 D 后面
    print("4. 读取并写入后:", rb.to_list()) # 输出: ['C', 'D', 'E']

if __name__ == "__main__":
    # test_queue()
    # test_deque()
    test_ring_buffer()
