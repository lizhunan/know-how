# 链表

- 编辑：李竹楠
- 日期：2024/02/06

## 1. 排序和查找

该部分在[排序](./sort.md)和[查找](./searching.md)部分有详细说明。

## 2. 链表双指针

在数组中，可以使用对撞指针、快慢指针和分离双指针。但是，在单链表中，因为遍历节点只能顺着 `next` 指针向后进行，所以对于单链表而言，一般使用**快慢指针**和**分离双指针**。其中，快慢指针分为**起点不一致的快慢指针**和**步长不一致的快慢指针**。

### 2.1 起点不一致的快慢指针

**起点不一致的快慢指针**：指的是两个指针从同一侧开始遍历链表，但是两个指针的起点不一样。 快指针 `fast` 比慢指针 `slow` 先走 `n` 步，直到快指针移动到链表尾端时为止。

#### 2.1.1 起点不一致的快慢指针求解步骤

1. 使用两个指针 `slow`、`fast`。`slow`、`fast` 都指向链表的头节点，即：`slow = head`，`fast = head`。
2. 先将快指针向右移动 `n` 步。然后再同时向右移动快、慢指针。
3. 等到快指针移动到链表尾部（即 `fast == None`）时跳出循环体。

``` python
slow = head
fast = head
while n:
    fast = fast.next
    n -= 1
while fast:
    fast = fast.next
    slow = slow.next
```

#### 2.1.2 起点不一致的快慢指针适用范围

起点不一致的快慢指针主要用于找到链表中倒数第 k 个节点、删除链表倒数第 N 个节点等。

#### 2.1.3 例题

##### 2.1.3.1 [删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/description/)

给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

示例 1：

> 输入：head = [1,2,3,4,5], n = 2
> 输出：[1,2,3,5]

示例 2：

> 输入：head = [1], n = 1
> 输出：[]

示例 3：

> 输入：head = [1,2], n = 1
> 输出：[1]

思路：~~常规思路是遍历一遍链表，求出链表长度，再遍历一遍到对应位置，删除该位置上的节点~~。如果用一次遍历实现的话，可以使用**快慢指针**。让快指针先走 `n` 步，然后快慢指针、慢指针再同时走，每次一步，这样**等快指针遍历到链表尾部的时候，慢指针就刚好遍历到了倒数第 n 个节点位置**。将该位置上的节点删除即可。

需要注意的是要删除的节点可能包含了头节点。我们可以考虑在遍历之前，新建一个头节点，让其指向原来的头节点。这样，最终如果删除的是头节点，则删除原头节点即可。返回结果的时候，可以直接返回新建头节点的下一位节点（小技巧：用于应对处理第一个元素的方式，可以在前面加一个头）。

``` python
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        newHead = ListNode(-1, head)
        fast = head
        slow = newHead
        while n:
            fast = fast.next
            n -= 1
        while fast:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return newHead.next
```

- 时间复杂度：$O(n)$。
- 空间复杂度：$O(1)$。

### 2.2 步长不一致的快慢指针

**步长不一致的快慢指针**：指的是两个指针从同一侧开始遍历链表，两个指针的起点一样，但是步长不一致。例如，慢指针 `slow` 每次走 1 步，快指针 `fast` 每次走 2 步。直到快指针移动到链表尾端时为止。

#### 2.2.1 步长不一致的快慢指针求解步骤

1. 使用两个指针 `slow`、`fast`。`slow`、`fast` 都指向链表的头节点。
2. 在循环体中将快、慢指针同时向右移动，但是快、慢指针的移动步长不一致。比如将慢指针每次移动 1 步，即 `slow = slow.next`。快指针每次移动 2 步，即 `fast = fast.next.next`。
3. 等到快指针移动到链表尾部（即 `fast == None`）时跳出循环体。

``` python
fast = head
slow = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
```

#### 2.2.2 步长不一致的快慢指针适用范围

步长不一致的快慢指针适合寻找链表的中点、判断和检测链表是否有环、找到两个链表的交点等问题。

#### 2.2.3 例题

##### 2.2.3.1 [链表的中间结点](https://leetcode.cn/problems/middle-of-the-linked-list/description/)

给你单链表的头结点 `head` ，请你找出并返回链表的中间结点。如果有两个中间结点，则返回第二个中间结点。

示例 1：

> 输入：head = [1,2,3,4,5]
> 输出：[3,4,5]
> 解释：链表只有一个中间结点，值为 3 。

示例 2：

> 输入：head = [1,2,3,4,5,6]
> 输出：[4,5,6]
> 解释：该链表有两个中间结点，值分别为 3 和 4 ，返回第二个结点。

思路：第一个思路是，先遍历一遍链表，统计一下节点个数为 n，再遍历到 n / 2 的位置，返回中间节点。第二个思路是使用快慢指针，使用步长不一致的快慢指针进行一次遍历找到链表的中间节点。

``` python
class Solution(object):
    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        fast = head
        slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        return slow
```

两种思路的复杂度都一样：

- 时间复杂度：$O(n)$。
- 空间复杂度：$O(1)$。

##### 2.2.3.2 [环形链表（判断是否有环）](https://leetcode.cn/problems/linked-list-cycle/description/)

给你一个链表的头节点 `head` ，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：`pos` 不作为参数进行传递 。仅仅是为了标识链表的实际情况。

如果链表中存在环 ，则返回 true 。 否则，返回 false 。

示例1：

> 输入：head = [3,2,0,-4], pos = 1
> 输出：true
> 解释：链表中有一个环，其尾部连接到第二个节点。

示例2：

> 输入：head = [1,2], pos = 0
> 输出：true
> 解释：链表中有一个环，其尾部连接到第一个节点。

示例3：

> 输入：head = [1], pos = -1
> 输出：false
> 解释：链表中没有环。

思路：第一个思路是使用哈希表，如果一次遍历下来有重复节点，则是有环。第二个思路是使用快慢指针（Floyd 判圈算法）这种方法类似于在操场跑道跑步。两个人从同一位置同时出发，如果跑道有环（环形跑道），那么快的一方总能追上慢的一方。

``` python

```

### 2.3 分离双指针

**分离双指针**：两个指针分别属于不同的链表，两个指针分别在两个链表中移动

#### 2.3.1 分离双指针求解步骤

1. 使用两个指针 `left_1`、`left_2`。`left_1` 指向第一个链表头节点，即：`left_1 = list1`，`left_2` 指向第二个链表头节点，即：`left_2 = list2`。
2. 当满足一定条件时，两个指针同时右移，即 `left_1 = left_1.next`、`left_2 = left_2.next`。
3. 当满足另外一定条件时，将 `left_1` 指针右移，即 `left_1 = left_1.next`。
4. 当满足其他一定条件时，将 `left_2` 指针右移，即 `left_2 = left_2.next`。
5. 当其中一个链表遍历完时或者满足其他特殊条件时跳出循环体。

``` python
left_1 = list1
left_2 = list2
while left_1 and left_2:
    if condition 1:
        left_1 = left_1.next
        left_2 = left_2.next
    elif condition 2:
        left_1 = left_1.next
    elif condition 3:
        left_2 = left_2.next
```

#### 2.3.2 分离双指针适用范围

分离双指针一般用于有序链表合并等问题。

#### 2.3.3 例题

##### 2.3.3.1 [合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/description/)

将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

示例 1：

> 输入：l1 = [1,2,4], l2 = [1,3,4]
> 输出：[1,1,2,3,4,4]

示例 2：

> 输入：l1 = [], l2 = []
> 输出：[]

示例 3：

> 输入：l1 = [], l2 = [0]
> 输出：[0]

``` python
class Solution(object):
    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        l1 = list1
        l2 = list2
        node = ListNode(-1)
        ret = node
        if l1 is None or l2 is None:
            return l1 or l2 
        while l1 and l2:
            if l1.val <= l2.val:
                node.next = l1
                l1 = l1.next
            else:
                node.next = l2
                l2 = l2.next
            node = node.next
        node.next = l1 if l1 is not None else l2
        return ret.next
```

- 时间复杂度：$O(n)$。
- 空间复杂度：$O(n)$。