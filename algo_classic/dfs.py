import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

class DFS:
    """Giải quyết bài toán Discrete Optimization: Shortest Path."""
    def __init__(self, grid_size=20, obstacle_prob=0.15):
        self.size = grid_size
        self.grid = np.zeros((grid_size, grid_size))
        # Khởi tạo vật cản ngẫu nhiên
        self.grid[np.random.rand(grid_size, grid_size) < obstacle_prob] = 1
        self.start = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)
        # Đảm bảo điểm bắt đầu và kết thúc không bị chặn
        self.grid[self.start] = 0
        self.grid[self.goal] = 0

    def search(self):
        """Thực hiện thuật toán DFS sử dụng Stack (LIFO)"""
        stack = [self.start]
        visited = {self.start: None}
        nodes_expanded = 0
        start_time = time.time()
        
        while stack:
            curr = stack.pop() # Lấy phần tử cuối cùng (Duyệt sâu)
            nodes_expanded += 1
            
            if curr == self.goal:
                duration = time.time() - start_time
                return True, self._reconstruct(visited, curr), nodes_expanded, duration
            
            # Kiểm tra các ô lân cận (4 hướng)
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor = (curr[0] + dx, curr[1] + dy)
                if (0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size and 
                    self.grid[neighbor] == 0 and neighbor not in visited):
                    visited[neighbor] = curr
                    stack.append(neighbor)
                    
        return False, [], nodes_expanded, time.time() - start_time

    def _reconstruct(self, visited, curr):
        path = []
        while curr is not None:
            path.append(curr)
            curr = visited[curr]
        return path[::-1]

    def save_visual_result(self, path, nodes_expanded, filename="dfs_discrete_path.png"):
        plt.figure(figsize=(9, 9))
        plt.imshow(self.grid, cmap='binary')
        
        if path:
            py, px = zip(*path)
            plt.plot(px, py, color='#f39c12', linewidth=3, label='DFS Path (Non-optimal)')
            plt.scatter(self.start[1], self.start[0], color='#2ecc71', s=200, label='Start', zorder=5)
            plt.scatter(self.goal[1], self.goal[0], color='#3498db', s=200, label='Goal', zorder=5)
            plt.text(self.size / 2, -1.0, f"Total Path Length: {len(path)}", 
                     fontsize=15, color='#f39c12', fontweight='bold', ha='center')
        
        plt.title(f"DFS Discrete Search: Shortest Path Problem\nNodes Expanded: {nodes_expanded}", pad=25)
        plt.legend(loc='upper right')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

def run_dfs_statistics(n_runs=20):
    sizes = [10, 20, 30, 40, 50, 60]
    all_results = []
    
    print(f"Đang thực hiện kiểm thử DFS (Mỗi kích thước chạy {n_runs} lần)...")
    
    for s in sizes:
        for _ in range(n_runs):
            solver = DFS(grid_size=s, obstacle_prob=0.15)
            success, path, nodes, duration = solver.search()
            if success:
                all_results.append({
                    'Grid_Size': s,
                    'Time_sec': duration,
                    'Nodes_Expanded': nodes,
                    'Path_Length': len(path)
                })
        print(f"    - Hoàn thành thu thập dữ liệu DFS size {s}x{s}")

    df = pd.DataFrame(all_results)

    summary = df.groupby('Grid_Size').agg({
        'Time_sec': ['mean', 'median', 'std', 'min', 'max'],
        'Nodes_Expanded': ['mean', 'median', 'std'],
        'Path_Length': ['mean', 'max']
    })

    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns.values]
    summary = summary.reset_index()
    summary.to_csv("dfs_statistical_summary.csv", index=False, encoding='utf-8-sig')
    print("\nĐã tạo file thống kê DFS: dfs_statistical_summary.csv")


if __name__ == "__main__":
    demo = DFS(grid_size=20, obstacle_prob=0.15)
    success, path, nodes, _ = demo.search()
    if success:
        demo.save_visual_result(path, nodes)
    run_dfs_statistics(n_runs=20)