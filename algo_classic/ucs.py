import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
import heapq

class UCS:
    """Giải quyết bài toán Discrete Optimization: Shortest Path."""
    def __init__(self, grid_size=20, obstacle_prob=0.15):
        self.size = grid_size
        self.grid = np.zeros((grid_size, grid_size))
        # Khởi tạo vật cản ngẫu nhiên
        self.grid[np.random.rand(grid_size, grid_size) < obstacle_prob] = 1
        self.start = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)
        # Đảm bảo điểm bắt đầu và kết thúc trống
        self.grid[self.start] = 0
        self.grid[self.goal] = 0
        
        # Trong lưới 2D đơn giản, ta giả định mỗi bước đi có cost là 1
        self.step_cost = 1 

    def search(self):
        # Priority Queue lưu: (tổng_cost_hiện_tại, tọa_độ)
        pq = [(0, self.start)]
        visited = {} # Lưu tọa độ: cost_nhỏ_nhất_đã_biết
        parent = {self.start: None}
        nodes_expanded = 0
        start_time = time.time()
        
        while pq:
            current_cost, curr = heapq.heappop(pq)
            
            if curr in visited and visited[curr] <= current_cost:
                continue
                
            visited[curr] = current_cost
            nodes_expanded += 1
            
            if curr == self.goal:
                duration = time.time() - start_time
                return True, self._reconstruct(parent, curr), nodes_expanded, duration, current_cost
            
            # Kiểm tra 4 hướng
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor = (curr[0] + dx, curr[1] + dy)
                if (0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size and 
                    self.grid[neighbor] == 0):
                    
                    new_cost = current_cost + self.step_cost
                    if neighbor not in visited or new_cost < visited[neighbor]:
                        parent[neighbor] = curr
                        heapq.heappush(pq, (new_cost, neighbor))
                        
        return False, [], nodes_expanded, time.time() - start_time, 0

    def _reconstruct(self, parent, curr):
        path = []
        while curr is not None:
            path.append(curr)
            curr = parent.get(curr)
        return path[::-1]

    def save_visual_result(self, path, nodes_expanded, total_cost, filename="ucs_discrete_path.png"):
        plt.figure(figsize=(9, 9))
        plt.imshow(self.grid, cmap='binary')
        
        if path:
            py, px = zip(*path)
            plt.plot(px, py, color='#9b59b6', linewidth=3, label='UCS Path (Optimal Cost)')
            plt.scatter(self.start[1], self.start[0], color='#2ecc71', s=200, label='Start', zorder=5)
            plt.scatter(self.goal[1], self.goal[0], color='#3498db', s=200, label='Goal', zorder=5)
            plt.text(self.size / 2, -1.0, f"Total Cost: {total_cost} | Nodes Expanded: {nodes_expanded}", 
                     fontsize=14, color='#9b59b6', fontweight='bold', ha='center')
        
        plt.title(f"UCS Discrete Search: Shortest Path (Grid {self.size}x{self.size})", pad=25)
        plt.legend(loc='upper right')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

def run_ucs_statistics(n_runs=20):
    sizes = [10, 20, 30, 40, 50, 60]
    all_results = []
    
    print(f"Đang thực hiện kiểm thử UCS (Mỗi size chạy {n_runs} lần)...")
    
    for s in sizes:
        for _ in range(n_runs):
            solver = UCS(grid_size=s, obstacle_prob=0.15)
            success, path, nodes, duration, cost = solver.search()
            if success:
                all_results.append({
                    'Grid_Size': s,
                    'Time_sec': duration,
                    'Nodes_Expanded': nodes,
                    'Path_Length': len(path),
                    'Total_Cost': cost
                })
        print(f"    - Hoàn thành thu thập dữ liệu UCS size {s}x{s}")

    df = pd.DataFrame(all_results)

    summary = df.groupby('Grid_Size').agg({
        'Time_sec': ['mean', 'median', 'std', 'min', 'max'],
        'Nodes_Expanded': ['mean', 'median', 'std'],
        'Total_Cost': ['mean']
    })

    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns.values]
    summary = summary.reset_index()
    summary.to_csv("ucs_statistical_summary.csv", index=False, encoding='utf-8-sig')
    print("\nĐã tạo file thống kê UCS: ucs_statistical_summary.csv")


if __name__ == "__main__":
    demo = UCS(grid_size=20, obstacle_prob=0.15)
    success, path, nodes, _, cost = demo.search()
    if success:
        demo.save_visual_result(path, nodes, cost)
    run_ucs_statistics(n_runs=20)