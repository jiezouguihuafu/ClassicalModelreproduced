from typing import List
import re


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x=0, next=None):
        self.val = x
        self.next = next


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# TODO：206. 反转链表
class ReverseList:
    def reverseList(self, head: ListNode):
        return self.way1(head)

    # 方法一：迭代
    def way1(self,head: ListNode) -> ListNode:
        pre = None
        curr = head
        while curr:
            tmp = curr.next
            curr.next = pre
            pre = curr
            curr = tmp
        return pre

    # 方法二：递归
    def way2(self,head: ListNode) -> ListNode:
        if head is None or head.next == None:
            return head

        p = self.way2(head.next)
        head.next.next = head
        head.next = None
        return p


# TODO：141. 环形链表
class HasCycle:
    def hasCycle(self, head: ListNode) -> bool:
        return self.way1(head)

    # 方法一：快慢指针
    def way1(self,head: ListNode):
        if not head or not head.next:
            return False

        left = head
        right = head.next
        while right and right.next:
            if left == right:
                return True
            left = left.next
            right = right.next.next
        return False

    # 方法二：哈希表
    def way2(self,head: ListNode):
        s = set()
        while head:
            if head in s:
                return True
            s.add(head)
            head = head.next
        return False



class LeetCode(object):
    def __init__(self):
        pass


    # TODO：704. 二分查找
    def binary_search(self, nums: List[int], target: int):
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right -= 1
            else:
                left += 1
        return -1

    # TODO：92. 反转链表指定区域
    def reverseBetween(self, head: ListNode, m: int, n: int):
        # 方法一:递归
        def way1(head: ListNode, m, n):
            if not head:
                return None

            left, right = head, head
            stop = False

            def helper(right, m, n):
                nonlocal left, stop
                if n == 1:
                    return
                right = right.next
                if m > 1:
                    left = left.next
                helper(right, m - 1, n - 1)

                if left == right or left == right.next:
                    stop = True
                if not stop:
                    left.val, right.val = right.val, left.val
                    left = left.next

            helper(right, m, n)
            return head

        # 方法二：迭代
        def way2(head: ListNode, m, n):
            if not head: return None

            pre, curr = None, head
            while m > 1:
                pre = curr
                curr = curr.next
                m -= 1
                n -= 1
            con, tail = pre, curr
            while n:
                tmp = curr.next
                curr.next = pre
                pre = curr
                curr = tmp
                n -= 1
            if con:
                con.next = pre
            else:
                head = pre
            tail.next = curr
            return head

        # 方法三

        return way1(head, m, n)


    # TODO：142. 环形链表的入口
    def detectCycle(self, head: ListNode) -> ListNode:
        if not (head and head.next): return None
        slow = head
        fast = head
        while fast:
            slow = slow.next
            if fast.next:
                fast = fast.next.next
            else:
                return None
            if slow == fast:
                fast = head
                while slow != fast:
                    fast = fast.next
                    slow = slow.next
                return fast
        return None

    # TODO：21.合并两个有序链表
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        # 方法一：递归
        def way1(l1: ListNode, l2: ListNode):
            if not l1:
                return l2
            elif not l2:
                return l1
            elif l1.val <= l2.val:
                l1.next = way1(l1.next, l2)
                return l1
            else:
                l2.next = way1(l1, l2.next)
                return l2

        # 方法二：迭代
        def way2(l1: ListNode, l2: ListNode):
            pre = ListNode(-1)
            curr = pre
            while l1 and l2:
                if l1.val < l2.val:
                    curr.next = l1
                    tmp = l1.next
                    l1.next = l2
                    l1 = tmp
                else:
                    curr.next = l2
                    tmp = l2.next
                    l2.next = l1
                    l2 = tmp
                curr = curr.next
            curr.next = l1 if l1 else l2
            return pre.next

        return way2(l1, l2)

    # TODO：144.二叉树的前序遍历
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        # 方法一：递归
        def way1(root: TreeNode):
            return [root.val] + way1(root.left) + way1(root.right) if root else []

        # 方法二：迭代
        def way2(root: TreeNode) -> List[int]:
            if not root: return []
            stack = [root]
            element = []
            while stack:
                node = stack.pop()
                # print(node)
                if isinstance(node, int):
                    element.append(node)
                else:
                    if node.right:
                        stack.append(node.right)
                    if node.left:
                        stack.append(node.left)
                    stack.append(node.val)

            return element

        return way2(root)

    # TODO：94.二叉树的中序遍历
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        # 方法一：递归
        def way1(root: TreeNode):
            return way1(root.left) + [root.val] + way1(root.right) if root else []

        # 方法二：迭代
        def way2(root: TreeNode):
            if not root: return []
            stack = [root]
            element = []
            while stack:
                node = stack.pop()
                if isinstance(node, int):
                    element.append(node)
                else:
                    if node.right:
                        stack.append(node.right)
                    stack.append(node.val)
                    if node.left:
                        stack.append(node.left)
            return element

        return way2(root)

    # TODO：145. 二叉树的后序遍历
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        # 方法一：递归
        def way1(root: TreeNode):
            return way1(root.left) + way1(root.right) + [root.val] if root else []

        # 方法二：迭代
        def way2(root: TreeNode):
            if not root: return []
            stack = [root]
            element = []
            while stack:
                node = stack.pop()
                if isinstance(node, int):
                    element.append(node)
                else:
                    stack.append(node.val)
                    if node.right:
                        stack.append(node.right)
                    if node.left:
                        stack.append(node.left)
            return element

        return way2(root)

    # TODO：102. 二叉树的层次遍历
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        self.d = {}
        if not root: return []
        self.levelOrder_helper(root, 1)
        return list(self.d.values())

    def levelOrder_helper(self, node, layer):
        if node:
            self.d[layer] = self.d.get(layer, []) + [node.val]
            if node.left:
                self.levelOrder_helper(node.left, layer + 1)
            if node.right:
                self.levelOrder_helper(node.right, layer + 1)

    # ==================================

    # TODO：107.二叉树的层次遍历2
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        return self.levelOrder(root)[::-1]

    # TODO：103.二叉树的锯齿形层次遍历
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        self.d = {}
        if not root: return []
        self.zigzagLevelOrder_helper(root, 1)
        return list(self.d.values())

    def zigzagLevelOrder_helper(self, node, layer):
        if node:
            if layer % 2 == 1:
                self.d[layer] = self.d.get(layer, []) + [node.val]
            else:
                self.d[layer] = [node.val] + self.d.get(layer, [])
            if node.left:
                self.zigzagLevelOrder_helper(node.left, layer + 1)
            if node.right:
                self.zigzagLevelOrder_helper(node.right, layer + 1)

    # ==================================

    # TODO：104. 二叉树的最大深度
    def maxDepth(self, root: TreeNode) -> int:
        # 方法一：迭代
        def way1(root: TreeNode):
            if not root: return 0
            stack = [(root, 0)]
            max_deep = 0
            while stack:
                node, deep = stack.pop()
                deep = deep + 1
                max_deep = max(max_deep, deep)
                if node.left:
                    stack.append((node.left, deep))
                if node.right:
                    stack.append((node.right, deep))
            return max_deep

        # 方法一：递归
        return self.maxDepth_helper(root, 0)

    def maxDepth_helper(self, node, deep):
        if not node:
            return deep
        return max(self.maxDepth_helper(node.left, deep + 1), self.maxDepth_helper(node.right, deep + 1))

    # ==================================

    # TODO：230.二叉搜索树中第K小的元素
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        # 方法一：迭代
        def way1(root: TreeNode, k: int):
            element = []
            stack = [root]
            while stack:
                node = stack.pop()
                if isinstance(node, int):
                    element.append(node)
                else:
                    if node.right:
                        stack.append(node.right)
                    if node.left:
                        stack.append(node.left)
                    stack.append(node.val)
            element.sort()
            return element[k - 1]

        # 方法二：递归
        def way2(root: TreeNode, k: int):
            return self.inorderTraversal(root)

        return way2(root, k)[k - 1]

    # TODO：二叉树最大路径和，单向
    def maxPathSumUnidirection(self, root: TreeNode) -> int:
        # 只能父节点到子节点，路径可以不含根节点
        self.max_value = 0
        self.maxPathSumUnidirection_helper(root, 0)
        return self.max_value

    def maxPathSumUnidirection_helper(self, node, before_sum):
        if not node:
            self.max_value = max(self.max_value, before_sum)
        else:
            if before_sum < 0:
                before_sum = node.val
            else:
                before_sum += node.val

            self.max_value = max(self.max_value, before_sum)
            self.maxPathSumUnidirection_helper(node.left, before_sum)
            self.maxPathSumUnidirection_helper(node.right, before_sum)

    # =====================================

    # TODO：124.二叉树最大路径和，双向
    def maxPathSumBidirection(self, root: TreeNode):
        self.max_sum = float("-inf")
        return max(self.max_sum, self.maxPathSumBidirection_helper(root))

    def maxPathSumBidirection_helper(self, node):
        if isinstance(node, int):
            return node
        else:
            left = self.maxPathSumBidirection_helper(node.left) if node.left else 0
            right = self.maxPathSumBidirection_helper(node.right) if node.right else 0
            self.max_sum = max(self.max_sum, left + right + node.val, left + node.val, right + node.val)
            return max(node.val, left + node.val, right + node.val)
    # ===============================================


# TODO 翻转一个子链表，并返回新的头与尾
def reverse_evety_listnode(head: ListNode, tail: ListNode, terminal:ListNode):
    cur = head
    pre = None
    while cur != terminal:
        # 留下联系方式
        next = cur.next
        # 修改指针
        cur.next = pre
        # 继续往下走
        pre = cur
        cur = next
    # 反转后的新的头尾节点返回出去
    return tail, head

# TODO:快速排序
def quick_sort(arr):
    if not arr: return arr
    value = arr[0]
    left = []
    right = []
    for i in arr[1:]:
        if i < value:
            left.append(i)
        else:
            right.append(i)
    return quick_sort(left) + [value] + quick_sort(right)


# TODO：剑指 Offer 40. 最小的 k个数
class GetLeastNumbers(object):
    def __init__(self):
        self.small_k = []

    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        self.way1(arr, k)
        return self.small_k

    # 方法1：快速排序的思想
    def way1(self, arr: List[int], k: int):
        if k <= 0: return
        if arr:
            value = arr[0]
            left = []
            right = []
            for i in arr[1:]:
                if i < value:
                    left.append(i)
                else:
                    right.append(i)
            left_lenght = len(left)
            if left_lenght > k:
                self.way1(left, k)
            elif left_lenght == k:
                self.small_k.extend(left)
            elif left_lenght == k - 1:
                self.small_k.extend(left.append(value))
            elif left_lenght < k - 1:
                self.small_k.extend(left.append(value))
                self.way1(right, k - 1 - left_lenght)

    # 方法二：最小堆
    def way2(self, arr: List[int], k: int):
        import heapq
        if not arr: return []
        small_k = [-i for i in arr[:k]]
        heapq.heapify(small_k)
        for i in arr[k:]:
            if i < -small_k[0]:
                heapq.heappop(small_k)
                heapq.heappush(small_k, -i)
        ans = [-i for i in small_k]
        return ans


# TODO 19.删除链表的第N个节点
class RemoveNthFromEnd(object):
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        pre = ListNode(0, head)
        left = pre
        right = head
        for _ in range(n):
            right = right.next
        while right:
            left = left.next
            right = right.next
        left.next = left.next.next
        return pre.next


# TODO 2.链表求和
class AddTwoNumbers:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        node = ListNode()
        pre = node
        multi = 0
        while l1 or l2:
            if not l1:
                val_l1 = 0
            else:
                val_l1 = l1.val
                l1 = l1.next
            if not l2:
                val_l2 = 0
            else:
                val_l2 = l2.val
                l2 = l2.next
            multi, res = divmod(val_l1 + val_l2 + multi, 10)
            pre.next = ListNode(res)
            pre = pre.next
        if multi != 0:
            pre.next = ListNode(multi)
            pre = pre.next
        return node.next


# TODO 236. 二叉树的最近公共祖先
class LowestCommonAncestor:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        self.ancestor = None
        self.helper(root, p.val, q.val)
        return self.ancestor

    def helper(self, node, p, q):
        if not node:
            return []
        left = self.helper(node.left, p, q)
        right = self.helper(node, p, q)

        node_value_list = left + [node.val] + right
        if p in node_value_list and q in node_value_list and not self.ancestor:
            self.ancestor = node
        return node_value_list


# TODO 剑指 Offer 48. 最长不含重复字符的子字符串
class LengthOfLongestSubstring:
    def lengthOfLongestSubstring(self, s: str) -> int:
        s_len = len(s)
        if s_len <= 1: return s_len

        max_len = 0
        left = 0
        right = 0
        while right < s_len - 1:
            if s[right + 1] not in s[left:right + 1]:
                max_len = max(right - left + 2, max_len)
            else:
                max_len = max(right - left + 1, max_len)
                left += s[left:right + 1].index(s[right + 1]) + 1
            right += 1
        return max_len


# TODO 160. 相交链表 -- 两个链表的第一个公共节点
class GetIntersectionNode:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        preA = headA
        preB = headB
        while preA != preB:
            preA = preA.next if preA else headB
            preB = preB.next if preB else headA
        return preA


# TODO 889. 根据前序和后序遍历构造二叉树
class ConstructFromPrePost:
    def constructFromPrePost(self, pre: List[int], post: List[int]) -> TreeNode:
        if not pre: return None
        root = TreeNode(pre[0])
        if len(pre) == 1: return root

        L = post.index(pre[1])
        root.left = self.constructFromPrePost(pre[1:L + 1], post[:L])
        root.right = self.constructFromPrePost(pre[L + 1:], post[L:-1])
        return root


# TODO 199. 二叉树的右视图
class RightSideView:
    def rightSideView(self, root: TreeNode) -> List[int]:
        self.layer_node = {}
        self.helper(root, 0)
        return list(self.layer_node.values())

    def helper(self, node, layer):
        if node:
            self.layer_node[layer] = node.val
            self.helper(node.left, layer + 1)
            self.helper(node.right, layer + 1)


# TODO 53. 最大连续子数组累加和
class maxSubArray:
    def maxSubArray(self, nums: List[int]) -> int:
        if len(nums) == 1: return nums[0]
        max_value = nums[0]
        curr_value = nums[0]
        for i in nums[1:]:
            if curr_value <= 0:
                max_value = max(max_value, curr_value)
                curr_value = i
            else:
                curr_value += i
            max_value = max(max_value, curr_value)
        return max_value


# TODO 15. 三数之和
class ThreeSum:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        ans = []
        for first in range(n):
            if first > 0 and nums[first] == nums[first - 1]:
                continue
            third = n - 1

            for second in range(first + 1, n):
                if second > first + 1 and nums[second] == nums[second - 1]:
                    continue
                while second < third and nums[first] + nums[second] + nums[third] > 0:
                    third -= 1

                if second == third:
                    break
                if nums[first] + nums[second] + nums[third] == 0:
                    ans.append([nums[first], nums[second], nums[third]])
        return ans


# TODO 1143. 最长公共子序列
class LongestCommonSubsequence:
    # 迭代方法
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m = len(text1) + 1
        n = len(text2) + 1
        dp = [[0] * n] * m
        for i in range(1, m):
            for j in range(1, n):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[-1][-1]

    # 递归方法
    def way2(self, text1: str, text2: str) -> int:
        m = len(text1) - 1
        n = len(text2) - 1

        def dp(i, j):
            if i == -1 or j == -1:
                return 0
            if text1[i] == text2[j]:
                return dp(i - 1, j - 1) + 1
            else:
                return max(dp(i - 1, j), dp(i, j - 1))

        return dp(m, n)


# TODO 674. 最长连续递增序列
class FindLengthOfLCIS:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        if len(nums)<=1:return len(nums)
        before_max_len = 0
        nums=[float("-inf")] + nums
        curr_max_len = 0
        for i in range(1,len(nums)):
            if nums[i-1]<nums[i]:
                curr_max_len+=1
            else:
                curr_max_len=1
            before_max_len=max(before_max_len,curr_max_len)
        return before_max_len


# TODO 69. x 的平方根
class MySqrt:
    def mySqrt(self, x: int) -> int:
        left = 0
        right = x
        ans = 0
        while left<=right:
            mid = (left+right)//2
            if mid*mid<=x:
                ans=mid
                left=mid+1
            else:
                right=mid-1
        return ans


# TODO 148. 排序链表
class SortList:
    def sortList(self, head: ListNode) -> ListNode:
        if not head: return head
        n = 0
        pre = head
        while pre:
            pre = pre.next
            n += 1

        while n:
            pre = head
            for _ in range(n - 1):
                print(head)
                if pre.val > pre.next.val:
                    pre.val, pre.next.val = pre.next.val, pre.val
                pre = pre.next
            n -= 1
        return head


# TODO 227.基本计算器-表达式求值
class Calculate:
    def calculate(self, s: str) -> int:
        s = s.replace(" ", "")

        a = re.sub(r"([+-])", r" \1 ", s).split()
        b = []

        for i in a:
            if "*" in i or "/" in i:
                b.append(self.multi(i))
            else:
                b.append(i)

        return int(self.add(b))

    def add(self, a):
        b = []

        index = 0
        while index < len(a):
            if a[index].isdigit():
                b.append(float(a[index]))
                index += 1
            elif a[index] == "+":
                b.append(float(a[index + 1]))
                index += 2
            elif a[index] == "-":
                b.append(-float(a[index + 1]))
                index += 2

        n = 0
        for i in b:
            n = n + i
            print(n)
        return n

    def multi(self, a):

        a = re.sub(r"([*/])", r" \1 ", a).split()
        b = []

        index = 0
        while index < len(a):
            if a[index].isdigit():
                b.append(float(a[index]))
                index += 1
            elif a[index] == "*":
                b.append(float(a[index + 1]))
                index += 2
            elif a[index] == "/":
                b.append(1 / float(a[index + 1]))
                index += 2

        n = 1
        for i in b:
            n = int(n * i)
        return str(int(n))


# TODO 54. 螺旋矩阵
class SpiralOrder:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix or not matrix[0]:
            return []

        left = 0
        top = 0
        right = len(matrix[0]) - 1
        bottom = len(matrix) - 1
        order = []

        while left <= right and top <= bottom:
            for column in range(left, right + 1):
                order.append(matrix[top][column])
            for row in range(top + 1, bottom + 1):
                order.append(matrix[row][right])
            if left < right and top < bottom:
                for column in range(right - 1, left - 1, -1):
                    order.append(matrix[bottom][column])
                for row in range(bottom - 1, top, -1):
                    order.append(matrix[row][left])
            left += 1
            right -= 1
            top += 1
            bottom -= 1
        return order


# TODO 226. 翻转二叉树
class InvertTree:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root:
            root.left, root.right = root.right, root.left
            root.left = self.invertTree(root.left)
            root.right = self.invertTree(root.right)
        return root


# TODO 23. 合并K个升序链表
class MergeKLists:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        ele = []
        for one_list_node in lists:
            while one_list_node:
                ele.append(one_list_node.val)
                one_list_node = one_list_node.next

        ele.sort()
        new_node = ListNode()
        pre = new_node
        for v in ele:
            node = ListNode(v)
            pre.next = node
            pre = pre.next
        return new_node.next

# TODO 200. 岛屿数量
class NumIslands:
    def numIslands(self, grid: List[List[str]]) -> int:
        H = len(grid) + 2
        W = len(grid[0]) + 2

        new_grid = []
        new_grid.append(["0"]*W)
        for line in grid:
            new_grid.append(["0"]+line+["0"])
        new_grid.append(["0"]*W)

        def helper(h,w):
            if 0<=h < H-1 and 0<=w < W-1 and new_grid[h][w]=="1":
                new_grid[h][w]="0"

                helper(h+1,w)
                helper(h-1,w)
                helper(h,w+1)
                helper(h,w-1)

        n = 0
        for row in range(1,H-1):
            for col in range(1,W-1):
                if new_grid[row][col]=="1":
                    helper(row,col)
                    n+=1
        return n

# TODO 11. 盛最多水的容器
class MaxArea:
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height)-1
        area = 0
        while left<right:
            area=max(area,min(height[left],height[right])*(right-left))
            if height[left]<=height[right]:
                left+=1
            else:
                right-=1
        return area


# TODO 328. 奇偶链表
class OddEvenList:
    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head:return head

        evenHead = head.next
        odd = head
        even = evenHead
        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next=odd.next
            even=even.next
        odd.next=evenHead
        return head

# TODO 56. 合并区间
class Merge:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x:(x[0],x[1]))
        res = []
        for ele in intervals:
            if not res:
                res.append(ele)
            else:
                if res[-1][-1]>=ele[0]:
                    res[-1][-1]=max(ele[1],res[-1][-1])
                else:
                    res.append(ele)
        return res

# TODO 234. 回文链表
class IsPalindrome:
    def isPalindrome(self, head: ListNode) -> bool:
        if head is None:
            return True

        # 找到前半部分链表的尾节点并反转后半部分链表
        first_half_end = self.end_of_first_half(head)
        second_half_start = self.reverse_list(first_half_end.next)

        # 判断是否回文
        result = True
        first_position = head
        second_position = second_half_start
        while result and second_position is not None:
            if first_position.val != second_position.val:
                result = False
            first_position = first_position.next
            second_position = second_position.next

        return result

    def end_of_first_half(self, head):
        fast = head
        slow = head
        while fast.next is not None and fast.next.next is not None:
            fast = fast.next.next
            slow = slow.next
        return slow

    def reverse_list(self, head):
        previous = None
        current = head
        while current is not None:
            next_node = current.next
            current.next = previous
            previous = current
            current = next_node
        return previous

# TODO 46. 全排列
class Permute:
    def permute(self, nums):
        def backtrack(first):
            if first == n:
                res.append(nums[:])
            else:
                for i in range(first,n):
                    nums[first],nums[i] = nums[i], nums[first]
                    backtrack(first+1)
                    nums[first], nums[i] = nums[i], nums[first]

        n = len(nums)
        res = []
        backtrack(0)
        return res

# TODO 121.买卖股票的最佳时机
class MaxProfit:
    def maxProfit(self, prices: List[int]) -> int:
        minprice = float("inf")
        maxprofit = 0
        for price in prices:
            maxprofit = max(price - minprice, maxprofit)
            minprice = min(price, minprice)
        return maxprofit


# TODO 409. 最长回文串的长度大小
class LongestPalindrome:
    def longestPalindrome(self, s: str) -> int:
        middle=1
        d={}
        for i in s:
            d[i]=d.get(i,0)+1
        l = 0
        for i in d:
            n = d[i]
            multi,res = divmod(n,2)
            l+=multi*2
            if res and middle:
                l+=1
                middle=0

        return l


# TODO 33. 搜索旋转排序数组
class RotateSearch:
    def search(self, nums: List[int], target: int) -> int:
        if not nums:return -1

        l,r = 0,len(nums)-1
        while l<=r:
            mid = (l+r)//2
            if nums[mid]==target:
                return mid

            if nums[0]<=nums[mid]:
                if nums[0]<=target<nums[mid]:
                    r=mid-1
                else:
                    l=mid+1
            else:
                if nums[mid]<target<=nums[-1]:
                    l=mid+1
                else:
                    r=mid-1
        return -1


# TODO 81. 搜索旋转排序数组 II
class RotateSearch2:
    def search(self, nums: List[int], target: int) -> bool:
        if not nums:return False

        l,r = 0,len(nums)-1

        while l<=r:
            mid = (l+r)//2
            if nums[mid]==target:
                return True
            if nums[l]==nums[mid]:
                l+=1
                continue
            if nums[l]<nums[mid]:
                if nums[l]<=target<nums[mid]:
                    r=mid-1
                else:
                    l=mid+1
            else:
                if nums[mid]<target<=nums[r]:
                    l=mid+1
                else:
                    r=mid-1
        return False


# TODO 257. 二叉树的所有路径
class BinaryTreePaths:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        if not root: return []
        self.res = []
        self.helper([], root)
        print(self.res)

        return ["->".join(i) for i in self.res]

    def helper(self, before: list, node):
        if not node.left and not node.right:
            self.res.append(before + [str(node.val)])

        if node.left:
            self.helper(before + [str(node.val)], node.left)
        if node.right:
            self.helper(before + [str(node.val)], node.right)

# TODO 83. 删除排序链表中的重复元素
class DeleteDuplicates:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head:return head
        pre = head
        while pre.next and pre:
            if pre.val==pre.next.val:
                pre.next=pre.next.next
            else:
                pre=pre.next
        return head


# TODO 22. 括号生成
class GenerateParenthesis:
    def generateParenthesis(self, n: int) -> List[str]:
        if n==0:
            return [""]
        ans = []
        for c in range(n):
            for left in self.generateParenthesis(c):
                for right in self.generateParenthesis(n-1-c):
                    ans.append("({}){}".format(left,right))
        return ans


# TODO 7. 整数反转
class Reverse:
    def reverse(self, x: int) -> int:
        if x<0:
            return -self.helper(-x)
        else:
            return self.helper(x)

    def helper(self,x):
        output = 0
        while True:
            multi,res = divmod(x,10)
            output=output*10+res
            x//=10
            if multi==0:
                break
        return output if 0<=output<2**31 else 0


# TODO 剑指 Offer 22. 链表中倒数第k个节点
class GetKthFromEnd:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        left = head
        right = head
        for _ in range(k):
            right=right.next
        while right:
            left=left.next
            right=right.next
        return left


# TODO 169. 多数元素
class MajorityElement:
    def majorityElement(self, nums):
        count = 0
        candidate=None
        for value in nums:
            if count==0:
                candidate=value
            if value==candidate:
                count+=1
            else:
                count-=1
        return candidate


# TODO 剑指 Offer 54. 二叉搜索树的第k大节点
class KthLargest:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        self.l = []
        self.helper(root)
        return self.l[-k]


    def helper(self, node:TreeNode):
        if node.left:
            self.helper(node.left)
        self.l.append(node.val)
        if node.right:
            self.helper(node.right)


# TODO 48. 旋转图像
class Rotate:
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for i in range(n):
            for j in range(i,n):
                matrix[j][i],matrix[i][j] = matrix[i][j],matrix[j][i]

        for i in range(n):
            matrix[i].reverse()
        return None


# TODO 41. 缺失的第一个正数
class FirstMissingPositive:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            if nums[i]<=0:
                nums[i]=n+1

        for i in range(n):
            curr = abs(nums[i])
            if curr<=n:
                nums[curr-1]=-abs(nums[curr-1])

        for i in range(n):
            if nums[i]>0:
                return i+1
        return n+1



# TODO 64. 最小路径和
class MinPathSum:
    # 方法一：递归
    def way1(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]: return 0
        self.grid = grid
        self.length = len(grid[0])
        self.width = len(grid)
        return self.helper(0, 0)

    def helper(self, l, w):
        if l < self.length - 1 and w < self.width - 1:
            return min(self.helper(l, w + 1), self.helper(l + 1, w)) + self.grid[w][l]
        elif l == self.length - 1 and w < self.width - 1:
            return self.helper(l, w + 1) + self.grid[w][l]
        elif l < self.length - 1 and w == self.width - 1:
            return self.helper(l + 1, w) + self.grid[w][l]
        else:
            return self.grid[w][l]

    # 方法二：循环
    def way2(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]: return 0

        rows, columns = len(grid), len(grid[0])

        for i in range(1,columns):
            grid[0][i]+=grid[0][i-1]
        for i in range(1,rows):
            grid[i][0]+=grid[i-1][0]

        for i in range(1,rows):
            for j in range(1,columns):
                grid[i][j]+=min(grid[i-1][j],grid[i][j-1])

        return grid[-1][-1]


class LongestValidParentheses:
    def longestValidParentheses(self, s: str) -> int:
        maxans = 0
        stack = []
        stack.append(-1)

        for index,val in enumerate(s):
            if val=="(":
                stack.append(index)
            else:
                stack.pop()
                if not stack:
                    stack.append(index)
                else:
                    maxans=max(maxans,index-stack[-1])
        return maxans


# TODO 110. 平衡二叉树
class isBalancedTree:
    def isBalanced(self, root: TreeNode) -> bool:
        self.balance = True
        self.helper(root)
        return self.balance

    def helper(self,node):
        if not node:return 0
        left = self.helper(node.left)
        right = self.helper(node.rigth)
        if abs(left-right)>1:
            self.balance=False
        return max(left,right)+1


# TODO 112. 路径总和
class HasPathSum:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root:return False
        self.total = sum
        self.exist = False
        self.helper(root,0)
        return self.exist

    def helper(self,node,val):
        if not node.left and not node.right:
            curr = node.val + val
            if curr==self.total:
                self.exist=True
        else:
            if node.left:
                self.helper(node.left, val+node.val)
            if node.right:
                self.helper(node.right,node.val+val)


# TODO 113. 路径总和 II
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        if not root:return []
        self.total = sum
        self.exist = []
        self.helper(root,[])
        return self.exist

    def helper(self,node,val):
        if not node.left and not node.right:
            curr = val + [node.val]
            if sum(curr)==self.total:
                self.exist.append(curr)
        else:
            if node.left:
                self.helper(node.left, val+[node.val])
            if node.right:
                self.helper(node.right,val+[node.val])


# TODO 14. 最长公共前缀
class LongestCommonPrefix:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:return ""
        l = min([len(i) for i in strs])
        a = ""
        for i in range(l):
            if len(set([s[i] for s in strs]))==1:
                a+=strs[0][i]
            else:
                break
        return a


# TODO 617. 合并二叉树
class MergeTrees:
    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
        if t1 and t2:
            t1.val+=t2.val
            t1.left=self.mergeTrees(t1.left,t2.left)
            t1.right=self.mergeTrees(t1.right,t2.right)
            return t1
        elif not t1 and t2:
            return t2
        elif t1 and not t2:
            return t1
        else:
            return None


# TODO 162. 寻找峰值
class FindPeakElement:
    def findPeakElement(self, nums: List[int]) -> int:
        left ,right = 0,len(nums)-1
        while left<right:
            mid = (left+right)//2
            if nums[mid]>nums[mid+1]:
                right=mid
            else:
                left=mid+1
        return left


# TODO 329.矩阵中的最长递增路径
class LongestIncreasingPath:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix or not matrix[0]: return 0
        self.matrix = matrix
        self.rows, self.column = len(matrix), len(matrix[0])
        max_len = 0
        for r in range(self.rows):
            for c in range(self.column):
                curr = self.helper(r, c, matrix[r][c], 1)
                print(curr)
                max_len = max(max_len, curr)

        return max_len

    def helper(self, row, col, value, n):
        # left
        if col > 0 and self.matrix[row][col - 1] > value:
            left = self.helper(row, col - 1, self.matrix[row][col - 1], n + 1)
        else:
            left = n
        # right
        if col < self.column - 1 and self.matrix[row][col + 1] > value:
            right = self.helper(row, col + 1, self.matrix[row][col + 1], n + 1)
        else:
            right = n

        # up
        if row > 0 and self.matrix[row - 1][col] > value:
            up = self.helper(row - 1, col, self.matrix[row - 1][col], n + 1)
        else:
            up = n

        # down
        if row < self.rows - 1 and self.matrix[row + 1][col] > value:
            down = self.helper(row + 1, col, self.matrix[row + 1][col], n + 1)
        else:
            down = n

        return max([left, right, up, down])




