import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import math

DIM = 5
NUM_CITIES = 20
TRIALS = 30
T_INIT = 1000.0
T_MIN = 1e-4       
MARKOV_LEN = 100

np.random.seed(42)
TSP_COORDS = np.random.rand(NUM_CITIES, 2) * 100
TSP_DIST_MAT = np.zeros((NUM_CITIES, NUM_CITIES))
for i in range(NUM_CITIES):
    for j in range(NUM_CITIES):
        TSP_DIST_MAT[i][j] = np.linalg.norm(TSP_COORDS[i] - TSP_COORDS[j])

def rastrigin_vectorized(x):
    return 10 * x.shape[1] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)

def calc_tsp_dist_fast(path):
    return TSP_DIST_MAT[path[:-1], path[1:]].sum() + TSP_DIST_MAT[path[-1], path[0]]

def run_sa_rastrigin_fast(alpha):
    bounds = [-5.12, 5.12]
    start_time = time.time()
    
    curr_x = np.random.uniform(bounds[0], bounds[1], (TRIALS, DIM))
    curr_val = rastrigin_vectorized(curr_x)
    best_x = np.copy(curr_x)
    best_val = np.copy(curr_val)
    
    temp = T_INIT
    history_best_val = [np.min(best_val)]
    
    while temp > T_MIN:
        for _ in range(MARKOV_LEN):
            candidate_x = np.clip(curr_x + np.random.uniform(-0.5, 0.5, (TRIALS, DIM)), bounds[0], bounds[1])
            candidate_val = rastrigin_vectorized(candidate_x)
            
            delta = candidate_val - curr_val
            
            accept = (delta < 0) | (np.random.rand(TRIALS) < np.exp(np.clip(-delta / temp, -700, 0)))
            
            curr_x[accept] = candidate_x[accept]
            curr_val[accept] = candidate_val[accept]
  
            improved = curr_val < best_val
            best_x[improved] = curr_x[improved]
            best_val[improved] = curr_val[improved]
            
        history_best_val.append(np.min(best_val))
        temp *= alpha
        
    end_time = time.time()
    avg_time = (end_time - start_time) / TRIALS
    return best_val, avg_time, history_best_val

def run_sa_tsp(alpha):
    curr_path = np.random.permutation(NUM_CITIES)
    curr_dist = calc_tsp_dist_fast(curr_path)
    best_path, best_dist = np.copy(curr_path), curr_dist
    
    temp = T_INIT
    start_time = time.time()
    
    while temp > T_MIN:
        for _ in range(MARKOV_LEN):
            i, j = np.random.randint(0, NUM_CITIES, 2)
            if i > j: i, j = j, i
            
            candidate_path = np.copy(curr_path)
            candidate_path[i:j+1] = candidate_path[i:j+1][::-1]
            candidate_dist = calc_tsp_dist_fast(candidate_path)
            
            delta = candidate_dist - curr_dist
            if delta < 0 or np.random.rand() < math.exp(-delta / temp):
                curr_path, curr_dist = candidate_path, candidate_dist
                if curr_dist < best_dist:
                    best_path, best_dist = np.copy(curr_path), curr_dist
                    
       
        if best_dist < 400.0:
            break
            
        temp *= alpha
        
    return best_dist, best_path, time.time() - start_time

def plot_sa_rastrigin_3d():
    def rast_2d(x, y):
        return 20 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y)
    
    path_x, path_y, path_z = [], [], []
    curr_x, curr_y = 4.0, 4.0
    temp = T_INIT
    while temp > 1e-1:
        path_x.append(curr_x)
        path_y.append(curr_y)
        path_z.append(rast_2d(curr_x, curr_y))
        
        nx = np.clip(curr_x + np.random.uniform(-0.5, 0.5), -5.12, 5.12)
        ny = np.clip(curr_y + np.random.uniform(-0.5, 0.5), -5.12, 5.12)
        
        diff = rast_2d(nx, ny) - rast_2d(curr_x, curr_y)
        if diff < 0 or np.random.rand() < math.exp(-diff / temp):
            curr_x, curr_y = nx, ny
        temp *= 0.90
        
    X = np.linspace(-5.12, 5.12, 100)
    Y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(X, Y)
    Z = rast_2d(X, Y)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6, linewidth=0, antialiased=False)
    ax.plot(path_x, path_y, path_z, color='red', marker='o', markersize=3, linewidth=2, label='SA Trajectory')
    ax.scatter(path_x[0], path_y[0], path_z[0], color='green', s=100, label='Start')
    ax.scatter(path_x[-1], path_y[-1], path_z[-1], color='black', s=100, label='End')
    
    ax.set_title('3D Visualization: SA Path on Rastrigin')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Fitness Value')
    ax.legend()
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=45, azim=45)
    plt.savefig('sa_3d_landscape.png', dpi=300)

def main():
    print("-" * 90)
    print(f"{'Cooling Rate (Alpha)':<20} | {'Best Fitness':<15} | {'Mean Fitness':<15} | {'Std Dev':<15} | {'Avg Time (s)':<15}")
    print("-" * 90)
    
    alphas = [0.90, 0.95, 0.99]
    best_overall_history = None
    
    for alpha in alphas:
        best_vals, avg_t, hist = run_sa_rastrigin_fast(alpha)
        
        if alpha == 0.99: 
            best_overall_history = hist
            
        print(f"{alpha:<20} | {np.min(best_vals):<15.5f} | {np.mean(best_vals):<15.5f} | {np.std(best_vals):<15.5f} | {avg_t:<15.5f}")

    plt.figure(figsize=(8, 5))
    plt.plot(best_overall_history, color='blue', linewidth=2, label='Best So Far')
    plt.yscale('log')
    plt.title('SA Convergence on Rastrigin Function')
    plt.xlabel('Temperature Steps')
    plt.ylabel('Best Fitness (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig('sa_ras_convergence.png', dpi=300)

    print("\n" + "-" * 105)
    print(f"{'Algorithm':<25} | {'Best Dist':<12} | {'Mean Dist':<12} | {'Std Dev':<12} | {'Success Rate':<15} | {'Avg Time (s)':<15}")
    print("-" * 105)
    
    sa_results = []
    sa_times = []
    best_tsp_dist = float('inf')
    best_tsp_perm = None
    
    for _ in range(TRIALS):
        dist, perm, t = run_sa_tsp(0.99)
        sa_results.append(dist)
        sa_times.append(t)
        if dist < best_tsp_dist:
            best_tsp_dist = dist
            best_tsp_perm = perm

    threshold = 400.0
    sa_success = sum(1 for x in sa_results if x < threshold) / TRIALS * 100
    
    print(f"{'Simulated Annealing':<25} | {np.min(sa_results):<12.2f} | {np.mean(sa_results):<12.2f} | {np.std(sa_results):<12.2f} | {f'{sa_success:.0f}%':<15} | {np.mean(sa_times):<15.4f}")

    ordered_coords = np.vstack([TSP_COORDS[best_tsp_perm], TSP_COORDS[best_tsp_perm][0]])
    plt.figure(figsize=(8, 5))
    plt.plot(ordered_coords[:, 0], ordered_coords[:, 1], 'o-', color='red', linewidth=2, label='Optimized Route')
    for i, txt in enumerate(best_tsp_perm):
        plt.annotate(str(txt), (TSP_COORDS[txt, 0], TSP_COORDS[txt, 1]), fontsize=10, weight='bold', xytext=(5, 5), textcoords='offset points')
    plt.title(f'SA Optimized Route (Distance: {best_tsp_dist:.2f})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.savefig('sa_tsp_route.png', dpi=300)
    
    plot_sa_rastrigin_3d()

if __name__ == "__main__":
    main()