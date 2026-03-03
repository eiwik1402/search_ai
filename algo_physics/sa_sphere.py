import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

def sphere_function(x):
    """Bài toán liên tục: Tìm cực tiểu hàm Sphere """
    return np.sum(x**2)

class SimulatedAnnealingDiscrete:
    """Giải quyết bài toán Discrete Optimization: Traveling Salesman Problem (TSP) """
    def __init__(self, n_cities=20, area_size=100):
        self.n_cities = n_cities
        # Khởi tạo tọa độ các thành phố ngẫu nhiên
        self.cities = np.random.rand(n_cities, 2) * area_size
        self.dist_matrix = self._create_dist_matrix()

    def _create_dist_matrix(self):
        dists = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                dists[i, j] = np.linalg.norm(self.cities[i] - self.cities[j])
        return dists

    def total_distance(self, tour):
        """Hàm năng lượng (Energy): Tổng quãng đường của tour """
        dist = 0
        for i in range(len(tour)):
            dist += self.dist_matrix[tour[i], tour[(i + 1) % len(tour)]]
        return dist

    def run(self, T_start=1000, cooling_rate=0.995, max_iter=2000):
        # Khởi tạo tour ngẫu nhiên
        current_tour = np.random.permutation(self.n_cities)
        current_dist = self.total_distance(current_tour)
        best_tour = current_tour.copy()
        best_dist = current_dist
        
        T = T_start
        history_best = []
        start_time = time.time()

        for _ in range(max_iter):
            # Sinh neighbor bằng cách swap 2 thành phố ngẫu nhiên (2-opt move)
            neighbor = current_tour.copy()
            idx1, idx2 = np.random.choice(self.n_cities, 2, replace=False)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
            
            next_dist = self.total_distance(neighbor)
            delta_e = next_dist - current_dist
            
            # Tiêu chuẩn Metropolis 
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                current_tour, current_dist = neighbor, next_dist
                if current_dist < best_dist:
                    best_dist, best_tour = current_dist, current_tour.copy()
            
            history_best.append(best_dist)
            T *= cooling_rate
            if T < 0.001: break
            
        return best_tour, best_dist, history_best, time.time() - start_time

class SimulatedAnnealingContinuous:
    """Giải quyết bài toán Continuous Optimization: Sphere Function """
    def __init__(self, func, bounds=(-5, 5), T_start=100, cooling_rate=0.99):
        self.func = func
        self.lb, self.ub = bounds
        self.T_start = T_start
        self.cooling_rate = cooling_rate

    def run(self, max_iter=1000):
        current = np.random.uniform(self.lb, self.ub, 2)
        curr_val = self.func(current)
        history_pos = [current.copy()]
        T = self.T_start
        start_time = time.time()

        for _ in range(max_iter):
            neighbor = current + np.random.uniform(-0.5, 0.5, 2)
            neighbor = np.clip(neighbor, self.lb, self.ub)
            next_val = self.func(neighbor)
            
            delta_e = next_val - curr_val
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                current, curr_val = neighbor, next_val
                history_pos.append(current.copy())
            
            T *= self.cooling_rate
            if T < 0.001: break
            
        return history_pos, curr_val, time.time() - start_time

def execute_report():
    print("[*] Đang thực thi Simulated Annealing (SA)...")
    stats = []

    # 1. Xử lý TSP (Discrete)
    print("    -> Giải quyết bài toán TSP...")
    tsp_solver = SimulatedAnnealingDiscrete(n_cities=20)
    best_tour, best_dist, history, d_time = tsp_solver.run()
    
    plt.figure(figsize=(8, 6))
    tour_coords = tsp_solver.cities[np.append(best_tour, best_tour[0])]
    plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'r-o', markersize=8)
    plt.title(f"SA TSP Solution\nTotal Distance: {best_dist:.2f}")
    plt.savefig("sa_tsp_result.png")
    plt.close()

    # 2. Xử lý Sphere (Continuous)
    print("    -> Giải quyết bài toán Sphere 3D...")
    sa_cont = SimulatedAnnealingContinuous(sphere_function)
    history_pos, best_v, c_time = sa_cont.run()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
    ax.plot_surface(X, Y, X**2 + Y**2, cmap='viridis', alpha=0.3)
    h = np.array(history_pos)
    ax.plot(h[:, 0], h[:, 1], h[:, 0]**2 + h[:, 1]**2, color='red', marker='o', markersize=2)
    ax.set_title(f"SA Continuous: Sphere 3D\nBest Value: {best_v:.4f}")
    plt.savefig("sa_continuous_3d.png")
    plt.close()

    # 3. Thống kê (Multiple Runs) 
    print("    -> Đang thu thập dữ liệu thống kê (Mean, Std, Median)...")
    for _ in range(25):
        # Test TSP
        _, d, _, t1 = SimulatedAnnealingDiscrete(20).run()
        stats.append({'Type': 'TSP (Discrete)', 'Fitness': d, 'Time': t1})
        # Test Sphere
        _, v, t2 = SimulatedAnnealingContinuous(sphere_function).run()
        stats.append({'Type': 'Sphere (Continuous)', 'Fitness': v, 'Time': t2})

    df = pd.DataFrame(stats)
    summary = df.groupby('Type').agg(['mean', 'median', 'std', 'min', 'max'])
    summary.to_csv("sa_statistical_summary.csv", encoding='utf-8-sig')
    
    print("\nHOÀN THÀNH!")
    print("[*] Đã lưu: sa_tsp_result.png, sa_continuous_3d.png, sa_statistical_summary.csv")

if __name__ == "__main__":
    execute_report()