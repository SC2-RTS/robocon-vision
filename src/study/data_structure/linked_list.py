class ListNode:
    def __init__(self, value: int):
        self.value = value
        self.next: ListNode | None = None

class DoublyListNode:
    def __init__(self, value: int):
        self.value = value
        self.prev: DoublyListNode | None = None
        self.next: DoublyListNode | None = None
        