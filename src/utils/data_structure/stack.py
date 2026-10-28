try:  # 作为包的一部分被 import
    from .linked_list import ListNode
except ImportError:  # 直接运行本文件
    from linked_list import ListNode

class LinkedListStack:
    """基于链表实现的栈"""

    def __init__(self):
        """构造方法"""
        self._peek: ListNode | None = None
        self._size: int = 0

    def size(self) -> int:
        """获取栈的长度"""
        return self._size

    def is_empty(self) -> bool:
        """判断栈是否为空"""
        return self._size == 0

    def push(self, value: int):
        """入栈"""
        node = ListNode(value)
        node.next = self._peek
        self._peek = node
        self._size += 1

    def pop(self) -> int:
        """出栈"""
        num = self.peek()
        self._peek = self._peek.next
        self._size -= 1

        return num

    def peek(self) -> int:
        """访问栈顶元素"""
        if self.is_empty():
            raise IndexError("栈为空")

        return self._peek.value

    def to_list(self) -> list[int]:
        """转化为列表用于打印"""
        arr = []
        node = self._peek
        while node:
            arr.append(node.value)
            node = node.next

        arr.reverse()
        return arr

if __name__ == "__main__":
    stack = LinkedListStack()
    for i in range(10):
        stack.push(i)

    print(stack.to_list())
    print(stack.pop())
    print(stack.to_list())
    