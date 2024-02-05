import queue

class BinTreeNode:
    """
        二叉树节点类
    """
    
    def __init__(self, value: int):
        self.value = value # 节点值
        self.left = None   # 左子节点引用
        self.right = None  # 右子节点引用

class BinTree:
    """
        二叉树
    """

    def __init__(self, root):
        self.root = root
        self.result = []

    def level_order(self, root):
        if root is None:
            return
        q = queue.Queue() 
        q.put(root)
        while not q.empty():
            node = q.get()
            self.result.append(node.value)
            if node.left is not None:
                q.put(node.left)
            if node.right is not None:
                q.put(node.right)

    def pre_order(self, node):
        if node is None:
            return
        self.result.append(node.value)
        self.in_order(node.left)
        self.in_order(node.right)
        

    def in_order(self, node):
        if node is None:
            return
        self.in_order(node.left)
        self.result.append(node.value)
        self.in_order(node.right)

    def post_order(self, node):
        if node is None:
            return
        self.post_order(node.left)
        self.post_order(node.right)
        self.result.append(node.value)

    def search(self, key):
        cur = self.root
        while cur is not None:
            if key < cur.value:
                cur = cur.left
            elif key > cur.value:
                cur = cur.right
            else:
                return cur
        return None

    def insert(self, key):
        if self.root is None:
            self.root = BinTreeNode(key)
            return
        cur = self.root
        pre = None
        while cur is not None:
            pre = cur
            if key < cur.value:
                cur = cur.left
            elif key > cur.value:
                cur = cur.right
            else:
                return
        node = BinTreeNode(key)
        if pre.value > key:
            pre.left = node
        else:
            pre.right = node
    
    def remove(self, key):
        if self.root is None:
            return
        cur = self.root
        pre = None
        while cur is not None:
            if cur.value == key:
                break
            pre = cur
            if key < cur.value:
                cur = cur.left
            elif key > cur.value:
                cur = cur.right
        if cur is None:
            return
        if (cur.right is None) or (cur.left is None):
            # 删除节点的度为0或1
            child = cur.left or cur.right
            if cur != self.root:
                if pre.left == cur:
                    pre.left = child
                else:
                    pre.right = child
            else:
                # 若删除根节点，则重新制定根节点
                self.root = child
        elif (cur.right is not None) and (cur.left is not None):
            # 删除节点的度为2
            tmp = cur.right # 获取待删除节点的下一个节点
            while tmp.left is not None: # 找到右子树的最小节点
                tmp = tmp.left
            self.remove(tmp.value) # 递归删除节点
            cur.value = tmp.value

class AVLTreeNode:
    """
        AVL 树节点类
    """    

    def __init__(self, value: int):
        self.value = value # 节点值
        self.height = 0    # 节点高度
        self.left = None   # 左子节点引用
        self.right = None  # 右子节点引用

class AVLTree:
    """
        AVL 树
    """

    def __init__(self, root):
        self.root = root

    def height(self, node):
        if node is not None:
            return node.height
        return -1

    def _update_height(self, node):
        node.height = max([self.height(node.left), self.height(node.right)]) + 1
    
    def _balance_factor(self, node):
        if node is None:
            return 0
        # 节点平衡因子 = 左子树高度 - 右子树高度
        return self.height(node.left) - self.height(node.right)

##### Binary Tree 1
root1 = BinTreeNode(1)
node2 = BinTreeNode(2)
node3 = BinTreeNode(3)
node4 = BinTreeNode(4)
node5 = BinTreeNode(5)
node6 = BinTreeNode(6)
node7 = BinTreeNode(7)
root1.left = node2
root1.right = node3
node2.left = node4
node2.right = node5
node3.left = node6
node3.right = node7
##### Binary Tree 1

##### Binary Tree 2
root2 = BinTreeNode(8)
node2 = BinTreeNode(4)
node3 = BinTreeNode(12)
node4 = BinTreeNode(2)
node5 = BinTreeNode(6)
node6 = BinTreeNode(11)
node7 = BinTreeNode(14)
node8 = BinTreeNode(1)
node9 = BinTreeNode(13)
root2.left = node2
root2.right = node3
node2.left = node4
node2.right = node5
node3.left = node6
node3.right = node7
node4.left = node8
node4.right = None
node7.left = node9
##### Binary Tree 2

# 二叉树遍历测试
tree = BinTree(root1)
tree.level_order(root1)
print(tree.result)
tree = BinTree(root1)
tree.in_order(root1)
print(tree.result)
print(tree.search(3))

# BST 测试
tree = BinTree(root2)
tree.insert(16)
tree.insert(10)
tree.remove(1)
tree.remove(6)
tree.remove(4)
tree.remove(12)
tree.remove(8)
tree.remove(16)
tree.remove(1)
tree.remove(10)
tree.remove(2)
tree.remove(13)
tree.remove(14)
tree.remove(11)
tree.in_order(tree.root)
print(tree.result)