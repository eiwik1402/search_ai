import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
from collections import deque
from mpl_toolkits.mplot3d import Axes3D

def sphere_function(x):
    """Bài toán liên tục: Tìm cực tiểu hàm Sphere (convex, unimodal)"""
    return np.sum(x**2)

class HillClimbingDiscrete:
    """Giải quyết bài toán Discrete Optimization: Shortest Path"""
    def __init__(self, grid_size=20, obstacle_prob=0.15):
        self.size = grid_size
        self.grid = np.zeros((grid_size, grid_size))
        self.grid[np.random.rand(grid_size, grid_size) < obstacle_prob] = 1
        self.start, self.goal = (0, 0), (grid_size - 1, grid_size - 1)
        self.grid[self.start], self.grid[self.goal] = 0, 0
        
    def value(self, pos):
        """Hàm đánh giá (Heuristic): Giá trị âm của Manhattan Distance để 'leo lên' đỉnh 0"""
        return -(abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1]))

    def run(self):
        current = self.start
        path = [current]
        start_time = time.time()
        nodes_expanded = 0

        while True:
            nodes_expanded += 1
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (current[0] + dx, current[1] + dy)
                if (0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size and 
                    self.grid[nxt] == 0 and nxt not in path):
                    neighbors.append(nxt)

            if not neighbors: 
                break 

            neighbor = max(neighbors, key=lambda n: self.value(n))
            
            if self.value(neighbor) <= self.value(current):
                duration = time.time() - start_time
                return (current == self.goal), path, nodes_expanded, duration
            
            current = neighbor
            path.append(current)

        duration = time.time() - start_time
        return (current == self.goal), path, nodes_expanded, duration

class HillClimbingContinuous:
    """Giải quyết bài toán Continuous Optimization: Sphere Function"""
    def __init__(self, func, bounds=(-5, 5), step_size=0.1, max_iter=250):
        self.func = func
        self.lb, self.ub = bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.history_pos = []

    def value(self, pos):
        """Tìm cực tiểu bằng cách 'leo lên' đỉnh của giá trị âm"""
        return -self.func(pos)

    def run(self):
        current = np.random.uniform(self.lb, self.ub, 2)
        self.history_pos = [current.copy()]
        start_time = time.time()

        for _ in range(self.max_iter): 
            best_neighbor = current.copy()

            for i in range(2):
                for move in [-self.step_size, self.step_size]:
                    nxt = current.copy()
                    nxt[i] = np.clip(nxt[i] + move, self.lb, self.ub)
                    if self.value(nxt) > self.value(best_neighbor):
                        best_neighbor = nxt
            
            if self.value(best_neighbor) <= self.value(current):
                break
            
            current = best_neighbor
            self.history_pos.append(current.copy())
            
        duration = time.time() - start_time
        return current, self.func(current), duration

def execute_and_report():
    all_stats = []

    # BÀI TOÁN RỜI RẠC (DISCRETE) 
    print("    -> Đang xử lý bài toán Rời rạc (Shortest Path)...")
    d_solver = HillClimbingDiscrete(grid_size=20)
    success, path, nodes, d_time = d_solver.run()

    plt.figure(figsize=(8, 8))
    plt.imshow(d_solver.grid, cmap='binary')
    if path:
        py, px = zip(*path)
        plt.plot(px, py, color='#e67e22', linewidth=3, label='HC Path')
        plt.text(15, -3.0, f"Total Path Length: {len(path)}", 
                 fontsize=14, color='red', fontweight='bold', ha='center')
    plt.title(f"Hill Climbing Discrete: Shortest Path\nNodes Expanded: {nodes}")
    plt.legend()
    plt.savefig("hc_discrete_result.png")
    plt.close()

    # BÀI TOÁN LIÊN TỤC (CONTINUOUS) 
    print("    -> Đang xử lý bài toán Liên tục (Sphere Function)...")
    c_solver = HillClimbingContinuous(sphere_function)
    best_p, best_v, c_time = c_solver.run()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    X_s = np.linspace(-5, 5, 50)
    Y_s = np.linspace(-5, 5, 50)
    X_s, Y_s = np.meshgrid(X_s, Y_s)
    Z_s = X_s**2 + Y_s**2
    ax.plot_surface(X_s, Y_s, Z_s, cmap='viridis', alpha=0.3)
    
    h_pos = np.array(c_solver.history_pos)
    h_z = np.array([sphere_function(p) for p in h_pos])
    ax.plot(h_pos[:, 0], h_pos[:, 1], h_z, color='red', marker='o', markersize=4, label='HC Descent Path')
    ax.set_title(f"Hill Climbing Continuous: Sphere 3D\nFinal f(x,y) = {best_v:.4f}")
    plt.savefig("hc_continuous_3d.png")
    plt.close()

    # CHẠY THỐNG KÊ (STATISTICS) 
    print("    -> Đang chạy thử nghiệm diện rộng để lấy dữ liệu thống kê...")
    for _ in range(30): 
        # Thống kê Discrete
        ds = HillClimbingDiscrete(grid_size=20)
        suc, p, n, t = ds.run()
        all_stats.append({'Type': 'Discrete', 'Success': suc, 'Time': t, 'Path_Length': len(p), 'Nodes': n, 'Fitness': 0})
        
        # Thống kê Continuous
        cs = HillClimbingContinuous(sphere_function)
        _, f_val, t_c = cs.run()
        all_stats.append({'Type': 'Continuous', 'Success': True, 'Time': t_c, 'Path_Length': 0, 'Nodes': 0, 'Fitness': f_val})

    df = pd.DataFrame(all_stats)

    summary = df.groupby('Type').agg({
        'Time': ['mean', 'median', 'std', 'min', 'max'],
        'Path_Length': ['mean', 'max'],
        'Nodes': ['mean'],
        'Fitness': ['mean', 'min']
    })

    summary.columns = [f"{c[0]}_{c[1]}" for c in summary.columns]
    summary.to_csv("hill_climbing_statistical_summary.csv", encoding='utf-8-sig')
    
    print("\nHOÀN THÀNH!")
    print("\n--- TÓM TẮT KẾT QUẢ ---")
    print(summary[['Time_mean', 'Time_std', 'Fitness_mean']])

if __name__ == "__main__":
    execute_and_report()