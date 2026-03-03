import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
import heapq

class AStar:
    """Giải quyết bài toán Discrete Optimization: Shortest Path."""
    def __init__(self, grid_size=20, obstacle_prob=0.15):
        self.size = grid_size
        self.grid = np.zeros((grid_size, grid_size))
        self.grid[np.random.rand(grid_size, grid_size) < obstacle_prob] = 1
        self.start = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)
        self.grid[self.start] = 0
        self.grid[self.goal] = 0

    def heuristic(self, a, b):
        """Khoảng cách Manhattan - Admissible heuristic cho lưới 2D"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def search(self):
        # Priority Queue lưu: (f_score, tọa_độ)
        pq = [(self.heuristic(self.start, self.goal), self.start)]
        
        g_score = {self.start: 0}
        parent = {self.start: None}
        nodes_expanded = 0
        start_time = time.time()
        
        while pq:
            current_f, curr = heapq.heappop(pq)
            nodes_expanded += 1
            
            if curr == self.goal:
                duration = time.time() - start_time
                return True, self._reconstruct(parent, curr), nodes_expanded, duration
            
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor = (curr[0] + dx, curr[1] + dy)
                if (0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size and 
                    self.grid[neighbor] == 0):
                    
                    tentative_g = g_score[curr] + 1
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + self.heuristic(neighbor, self.goal)
                        parent[neighbor] = curr
                        heapq.heappush(pq, (f_score, neighbor))
                        
        return False, [], nodes_expanded, time.time() - start_time

    def _reconstruct(self, parent, curr):
        path = []
        while curr is not None:
            path.append(curr)
            curr = parent.get(curr)
        return path[::-1]

    def save_visual_result(self, path, nodes_expanded, filename="astar_discrete_path.png"):
        plt.figure(figsize=(9, 9))
        plt.imshow(self.grid, cmap='binary')
        
        if path:
            py, px = zip(*path)
            plt.plot(px, py, color='#2980b9', linewidth=3, label='A* Optimal Path')
            plt.scatter(self.start[1], self.start[0], color='#2ecc71', s=200, label='Start')
            plt.scatter(self.goal[1], self.goal[0], color='#e74c3c', s=200, label='Goal')            
            plt.text(self.size / 2, -1.0, f"Total Path Length: {len(path)}", 
                     fontsize=15, color='#2980b9', fontweight='bold', ha='center')
        
        plt.title(f"A* Search: Discrete Shortest Path\nNodes Expanded: {nodes_expanded}", pad=25)
        plt.legend()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()


def run_astar_statistics(n_runs=20): 
    sizes = [10, 20, 30, 40, 50, 60]
    all_results = []
    
    print(f"Đang chạy thử nghiệm A* (Mỗi size {n_runs} lần)...")
    for s in sizes:
        for _ in range(n_runs):
            solver = AStar(grid_size=s, obstacle_prob=0.15)
            success, path, nodes, duration = solver.search()
            if success:
                all_results.append({
                    'Grid_Size': s, 'Time_sec': duration,
                    'Nodes_Expanded': nodes, 'Path_Length': len(path)
                })    

    df = pd.DataFrame(all_results)

    summary = df.groupby('Grid_Size').agg({
        'Time_sec': ['mean', 'median', 'std', 'min', 'max'],
        'Nodes_Expanded': ['mean', 'median', 'std'],
        'Path_Length': ['mean']
    })
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns.values]
    summary.to_csv("astar_statistical_summary.csv", encoding='utf-8-sig')   
    print("Đã tạo file: astar_statistical_summary.csv và astar_performance_analysis.png")


if __name__ == "__main__":
    demo = AStar(20, 0.15)
    ok, p, n, _ = demo.search()
    if ok: demo.save_visual_result(p, n)
    run_astar_statistics(n_runs=20)