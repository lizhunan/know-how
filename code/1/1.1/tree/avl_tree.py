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
        self.result = []

    def _height(self, node):
        if node is not None:
            return node.height
        return -1

    def _update_height(self, node):
        if node is None:
            return 0
        node.height = max([self._height(node.left), self._height(node.right)]) + 1
    
    def _balance_factor(self, node):
        if node is None:
            return 0
        # 节点平衡因子 = 左子树高度 - 右子树高度
        return self._height(node.left) - self._height(node.right)

    def _right_rotate(self, node):
        child = node.left
        grand_child = child.right
        # 以child为原点，将node向右旋转
        child.right = node
        node.left = grand_child
        # 更新节点高度
        self._update_height(node)
        self._update_height(child)
        return child

    def _left_rotate(self, node):
        child = node.right
        grand_child = child.left
        # 以child为原点，将node向右旋转
        child.left = node
        node.right = grand_child
        # 更新节点高度
        self._update_height(node)
        self._update_height(child)
        return child
    
    def _rotate(self, node):
        balance_factor = self._balance_factor(node)
        if balance_factor > 1:
            # 左偏树
            if self._balance_factor(node.left) >= 0:
                # 右旋
                return self._right_rotate(node)
            else:
                # 先左旋后右旋
                node.left = self._left_rotate(node.left)
                return self._right_rotate(node)
        elif balance_factor < -1:
            # 右偏树
            if self._balance_factor(node.right) <= 0:
                # 左旋
                return self._left_rotate(node)
            else:
                # 先右旋后左旋
                node.right = self._right_rotate(node.right)
            return self._left_rotate(node)

        # 平衡树，无须旋转，直接返回
        return node

    def _insert_helper(self, node, key):
        if node is None:
            return AVLTreeNode(key)
        # 1. 查找插入位置并插入节点
        if key < node.value:
            node.left = self._insert_helper(node.left, key)
        elif key > node.value:
            node.right = self._insert_helper(node.right, key)
        else:
            # 重复节点不插入，直接返回
            return node
        # 更新节点高度
        self._update_height(node)
        # 2. 执行旋转操作，使该子树重新恢复平衡
        return self._rotate(node)

    def insert(self, key):
        self.root = self._insert_helper(self.root, key)

    def _remove_helper(self, node, key):
        # 1. 查找节点并删除
        if node is None:
            return None
        # 1. 查找节点并删除
        if key < node.value:
            node.left = self._remove_helper(node.left, key)
        elif key > node.value:
            node.right = self._remove_helper(node.right, key)
        else:
            if node.left is None or node.right is None:
                child = node.left or node.right
                # 子节点数量 = 0 ，直接删除 node 并返回
                if child is None:
                    return None
                # 子节点数量 = 1 ，直接删除 node
                else:
                    node = child
            else:
                # 子节点数量 = 2 ，则将中序遍历的下个节点删除，并用该节点替换当前节点
                temp = node.right
                while temp.left is not None:
                    temp = temp.left
                node.right = self._remove_helper(node.right, temp.value)
                node.value = temp.value
        # 更新节点高度
        self._update_height(node)
        # 2. 执行旋转操作，使该子树重新恢复平衡
        return self._rotate(node)

    def remove(self, key):
        self.root = self._remove_helper(self.root, key)

    def in_order(self, node):
        if node is None:
            return
        self.in_order(node.left)
        self.result.append(node.value)
        self.in_order(node.right)

tree = AVLTree(None)
tree.insert(1)
tree.insert(3)
tree.insert(2)
tree.insert(10)
tree.insert(6)
tree.insert(9)
tree.remove(2)
tree.remove(3)
tree.remove(6)
tree.remove(10)
tree.remove(9)
tree.remove(1)
tree.in_order(tree.root)
print(tree.result)