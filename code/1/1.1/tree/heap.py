
class Heap:

    def __init__(self, mode='max'):
        self.mode = mode # 顶堆模式，max为大顶堆；min为小顶堆
        self.heap = []

    def _left(self, i):
        return 2 * i + 1

    def _right(self, i):
        return 2 * i + 2

    def _parent(self, i):
        return (i - 1) // 2  # 向下整除
    
    def _size(self):
        return len(self.heap)
    
    def _swap(self, i, p):
        tmp = self.heap[p]
        self.heap[p] = self.heap[i]
        self.heap[i] = tmp
    
    def _sift_up(self, i):
        while True:
            p = self._parent(i) # 获取节点 i 的父节点
            # 当“越过根节点”或“节点无须修复”时，结束堆化
            if p < 0 or self.heap[i] <= self.heap[p]:
                break
            self._swap(i, p) # 交换两节点
            i = p # 循环向上堆化

    def push(self, key):
        self.heap.append(key)
        self._sift_up(self._size() - 1) # 从底至顶堆化

    def _sift_down(self, i):
        while True:
            # 判断节点 i, l, r 中值最大的节点，记为 max_
            l, r, max_ = self._left(i), self._right(i), i
            if l < self._size() and self.heap[l] > self.heap[max_]:
                max_ = l
            if r < self._size() and self.heap[r] > self.heap[max_]:
                max_ = r
            # 若节点 i 最大或索引 l, r 越界，则无须继续堆化，跳出
            if max_ == i:
                break
            self._swap(i, max_)
            i = max_

    def pop(self):
        if self._size() == 0:
            # 判空处理
            raise IndexError('heap is empty.')
        # 1. 交换根节点与最右叶节点（交换首元素与尾元素）
        self._swap(0, self._size() - 1)
        # 2. 删除节点
        key = self.heap.pop()
        # 3. 从顶至底堆化
        self._sift_down(0)
        return key

    def peek(self) -> int:
        if self._size() <= 0:
            raise IndexError('heap is empty.')
        return self.heap[0]
    
    def top_k(self, nums, k):
        # 将数组的前 k 个元素入堆
        for i in range(k):
            self.push(nums[i])
        # 从第 k+1 个元素开始，保持堆的长度为 k
        for i in range(k, len(nums)):
            print(nums[i], '-', self.peek())
            # 若当前元素大于堆顶元素，则将堆顶元素出堆、当前元素入堆
            if nums[i] < self.peek():
                self.pop()
                self.push(nums[i])
    
# heap = Heap()
# heap.push(4)
# heap.push(9)
# heap.push(1)
# heap.push(3)
# heap.push(10)
# print(heap.peek())
# print(heap.max_heap)
# heap.pop()
# print(heap.peek())
# print(heap.max_heap)
# heap.pop()
# print(heap.peek())
# print(heap.max_heap)
# heap.pop()
# print(heap.peek())
# print(heap.max_heap)
# heap.pop()
# print(heap.peek())
# print(heap.max_heap)
# heap.pop()
# print(heap.peek())
# print(heap.max_heap)

nums = [1, 7, 6, 3, 2]
top = Heap()
top.top_k(nums, 3)
print(top.heap)
