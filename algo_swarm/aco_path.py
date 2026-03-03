import numpy as np
import time
import heapq
import pandas as pd

MOVES = [(-1,0),(1,0),(0,-1),(0,1)]

class ACO_Greedy_Path:
    def __init__(self, grid, alpha=1.0, beta=2.0):
        self.grid = grid
        self.size = grid.shape[0]
        self.start = (0, 0)
        self.goal = (self.size-1, self.size-1)
        self.alpha = alpha
        self.beta = beta
        self.pheromone = np.ones_like(grid, dtype=float)

    def heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def neighbors(self, pos, visited):
        for dx, dy in MOVES:
            nx, ny = pos[0]+dx, pos[1]+dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                if self.grid[nx,ny] == 0 and (nx,ny) not in visited:
                    yield (nx, ny)

    def _reconstruct(self, parent, curr):
        path = []
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        return path[::-1]

    def search(self):
        start_time = time.time()
        pq = []
        visited = set()
        # ✅ FIX: dùng parent-pointer dict thay vì lưu path trong heap
        parent = {self.start: None}
        nodes_expanded = 0

        h0 = self.heuristic(self.start, self.goal)
        heapq.heappush(pq, (h0, self.start))  # ✅ chỉ lưu (score, node)

        while pq:
            _, curr = heapq.heappop(pq)
            if curr in visited:
                continue
            visited.add(curr)
            nodes_expanded += 1

            if curr == self.goal:
                path = self._reconstruct(parent, curr)
                return True, len(path), nodes_expanded, time.time()-start_time

            for n in self.neighbors(curr, visited):
                tau = self.pheromone[n] ** self.alpha
                eta = (1 / (self.heuristic(n, self.goal) + 1)) ** self.beta
                score = -(tau * eta)
                if n not in parent:          # chỉ push nếu chưa gặp
                    parent[n] = curr
                    heapq.heappush(pq, (score, n))

        return False, None, nodes_expanded, time.time()-start_time


def generate_grid(size=30, obstacle_prob=0.15, seed=0):
    np.random.seed(seed)
    grid = (np.random.rand(size, size) < obstacle_prob).astype(int)
    grid[0,0] = 0
    grid[size-1,size-1] = 0
    return grid

# =========================
# BENCHMARK
# =========================
RUNS = 20
records = []

for r in range(RUNS):
    grid = generate_grid(seed=r)
    algo = ACO_Greedy_Path(grid)
    success, path_len, nodes, t = algo.search()
    records.append({
        "Run": r + 1,
        "Success": success,
        "Path_Length": path_len,
        "Nodes_Expanded": nodes,
        "Time_sec": t
    })

df = pd.DataFrame(records)

# Tính thống kê trước khi format
success_rate  = df['Success'].mean() * 100
mean_path     = df['Path_Length'].mean()
std_path      = df['Path_Length'].std()
mean_time     = df['Time_sec'].mean()
mean_nodes    = df['Nodes_Expanded'].mean()
df["Time_sec"] = df["Time_sec"].map(lambda x: f"{x:.4e}")