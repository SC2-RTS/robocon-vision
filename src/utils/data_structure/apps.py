from queue import RingBuffer
from heap import MinHeap
import random, statistics
random.seed(42)

def test1():
    ring_buffer = RingBuffer(5)
    temperature = 25
    nums_error = []
    for i in range(9):
        if i == 5:
            decimal = 31.0
        else:
            decimal = round(random.uniform(-0.5, 0.5), 2) + temperature

        nums_error.append(round(decimal - temperature, 2))
        ring_buffer.write(decimal)

    nums = ring_buffer.to_list()
    median = statistics.median(nums)
    print(f"Numbers: {nums}")
    print(f"Median: {median}")
    print(f'Median_error: {round(median - temperature, 2)}')
    print(f"Errors: {nums_error}")
    print()

def test2():
    random_score = [random.randint(0, 100) for _ in range(10)]
    min_heap = MinHeap()
    for score in random_score:
        min_heap.push(score)

    heap_list = min_heap.heap
    heap_list_reverse = sorted(heap_list, reverse=True)
    print(f"Random Scores: {random_score}")
    print(f"Heap List: {heap_list}")
    print(f"Heap List Reverse: {heap_list_reverse}")

if __name__ == "__main__":
    test1()
    test2()
