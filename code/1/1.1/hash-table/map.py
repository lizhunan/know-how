
class Pair:
    """
        键值对
    """

    def __init__(self, key, value):
        self.key = key
        self.value = value

class ArrayHashMap:
    """
        基于数组的哈希表
        无法避免哈希冲突
    """

    def __init__(self):

        # 长度为100的hash table
        self.buckets = [None] * 100
    
    def _hash_func(self, key):
        return key % len(self.buckets)

    def get(self, key):
        pair = self.buckets[self._hash_func(key)]
        if pair is None:
            return None
        return pair.value

    def put(self, key, value):
        pair = Pair(key, value)
        self.buckets[self._hash_func(key)] = pair

    def remove(self, key):
        self.buckets[self._hash_func(key)] = None

    def entry_set(self):
        result = []
        for pair in self.buckets:
            if pair is not None:
                result.append(pair)
        return result

    def key_set(self):
        result = []
        for pair in self.buckets:
            if pair is not None:
                result.append(pair.key)
        return result

    def value_set(self):
        result = []
        for pair in self.buckets:
            if pair is not None:
                result.append(pair.value)
        return result

    def print(self):
        for pair in self.buckets:
            if pair is not None:
                print(pair.key, "->", pair.value)

class LinkedHashMap:
    """
        链式地址哈希表
    """
    
    def __init__(self, load_thres=0.75, extend_ratio=2):
        
        self.size = 0 # 哈希表使用量
        self.capacity = 10 # 哈希表容量
        self.load_thres = load_thres # 扩容阈值
        self.extend_ratio = extend_ratio # 扩容倍数
        self.buckets = [[] for _ in range(self.capacity)] # hash table桶结构

    def _hash_func(self, key):
        return key % self.capacity

    def _load_factor(self):
        return self.size / self.capacity
    
    def _extend(self):  

        # 暂存旧哈希表
        temp = self.buckets

        # 扩容后的哈希表
        self.capacity *= self.extend_ratio
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0

        # 哈希表搬运
        for t in temp:
            for pair in t:
                self.put(pair.key, pair.value)
    
    def put(self, key, value):

        # 扩容
        if self._load_factor() > self.load_thres:
            self._extend()
        bucket = self.buckets[self._hash_func(key)]

        # 若key存在，则更新
        for pair in bucket:
            if pair.key == key:
                pair.value = value
                return

        # 若key不存在，则新增
        pair = Pair(key, value)
        bucket.append(pair)
        self.size += 1

    def remove(self, key):

        bucket = self.buckets[self._hash_func(key)]
        for pair in bucket:
            if pair.key == key:
                bucket.remove(pair)
                self.size -= 1
                break

    def get(self, key):
        bucket = self.buckets[self._hash_func(key)]
        for pair in bucket:
            if pair.key == key:
                return pair.value
        return None
    
    def print(self):
        for bucket in self.buckets:
            print('', end='|')
            for pair in bucket:
                if pair is not None:
                    print('(', pair.key, pair.value,')', end='->')
            print('\n')
        print('----------------------')

# map = ArrayHashMap()
# map.put(0, 'hash')
# map.put(1, 'hello')
# map.put(50, 'world')
# map.print()
# print('0,100 哈希冲突')
# map.put(100, '!')
# map.print()
# print('1,101 哈希冲突')
# map.put(101, 'end')
# map.print()

# linked_map = LinkedHashMap()

# for i in range(0, 24):
#     linked_map.put(i, str(i))
# linked_map.print()
# linked_map.remove(11)
# linked_map.print()