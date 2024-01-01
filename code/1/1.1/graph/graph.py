from collections import deque

class Graph:

    def __init__(self, map):
        self.map = map

    def BFS(self, start=0):
        # 使用队列
        queue = deque([start])
        visited = set([start])

        while queue:
            node = queue.popleft()
            for neighbor, has_edge in enumerate(self.map[node]):
                if has_edge and neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
            print(f'{node}: {queue}')

    def DFS(self, start=0):
        # 使用栈
        stack = deque([start])
        visited = set([start])

        while stack:
            node = stack.pop()
            for neighbor, has_edge in reversed(list(enumerate(self.map[node]))):
                if has_edge and neighbor not in visited:
                    stack.append(neighbor) 
                    visited.add(neighbor)
            print(f'{node}: {stack}')

    def dijkstra(self, start=0):
        ##
        ## 对应3.1节的dijkstra的算法图解，算法图解上节点用A, B, C, D, E表示节点[0..4]
        ##
        num_nodes = len(self.map)
        # 对应于算法图解中的集合S
        # 这是一个重要的的最短路径数组，不断更新该数组直到所有的点都被访问到
        s_set = [float('inf')] * num_nodes
        # 对应第一步中A->A=0，对集合进行初始化
        s_set[start] = 0
        # 访问过的点，每一次访问之后就要将访问过的点加入到该数组中
        # 这样做是为了避免重复访问
        visited = [False] * num_nodes

        for _ in range(num_nodes):
            
            min_distance = float('inf')
            min_node = -1

            # 选择未访问节点中距离最小的节点
            for i in range(num_nodes):
                if not visited[i] and s_set[i] < min_distance:
                    min_distance = s_set[i]
                    min_node = i
            
            visited[min_node] = True
            print(f'-------------find min_node: {min_node}')
        
            # 更新相邻节点的距离信息
            # 对应更新算法图解中的集合U
            for neighbor, weight in enumerate(self.map[min_node]):
                new_distance = float('inf')
                if not visited[neighbor] and weight > 0:
                    new_distance = s_set[min_node] + weight
                    if new_distance < s_set[neighbor]:
                        s_set[neighbor] = new_distance
                print(f'0-->{neighbor}={s_set[neighbor]}')

        print(s_set)

    def floyd_warshall(self, start=0):
        pass

    def A_star(self):
        pass


# map1与graph.md文档中2.1节中的../../../pics/90.gif图一致
# 注意，该图是无向图，箭头只是表示下一个节点法线方向，而不是有向图
map1 = [[0, 1, 1, 1, 1, 0, 0, 0],
       [1, 0, 0, 0, 0, 1, 1, 0],
       [1, 0, 0, 0, 0, 0, 0, 1],
       [1, 0, 0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 0]]

# map2与graph.md文档中3.1节中的../../../pics/93.png图一致
INF= float('inf')
map2 = [[0, 4, INF, 2, INF],
       [4, 0, 4, 1, INF],
       [INF, 4, 0, 1, 3],
       [2, 1, 1, 0, 7],
       [INF, INF, 3, 7, 0]]

### 图的遍历
# iteration = Graph(map=map1)
# iteration.DFS(start=0)

### 最短路径
shortest_path = Graph(map=map2)
# shortest_path.dijkstra(start=0)
shortest_path.floyd_warshall(start=0)