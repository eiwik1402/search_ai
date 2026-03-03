import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from algo_classic.bfs import BFS
from algo_classic.dfs import DFS
from algo_classic.ucs import UCS
from algo_classic.gbfs import GBFS
from algo_classic.astar import AStar
import algo_classic.hill_climbing as hill_climbing
from algo_evolution.de import run_de_rastrigin
from algo_evolution.ga import run_ga_rastrigin
from algo_physics.sa_rastrigin import run_sa_rastrigin_fast
from algo_swarm.aco import AntColony, create_cluster_cities
from algo_swarm.pso import PSO
import algo_swarm.abc as abc_module
from algo_swarm.fa import Firefly
from algo_swarm.cs import CuckooSearch
from algo_human_tlbo.tlbo import TLBO

def main_menu():
    print("\n" + "="*60)
    print("      ALGORITHM RESULTS VISUALIZATION MENU")
    print("="*60)
    print("Select algorithm category:")
    print("A. Classic Searching Algorithms")
    print("B. Evolution Algorithms")
    print("C. Physics-Based Algorithms")
    print("D. Swarm Intelligence Algorithms")
    print("E. Human-Based Learning Algorithms")
    print("F. Exit")

def classic_algorithms():
    return {
        1: "Breadth-First Search (BFS)",
        2: "Depth-First Search (DFS)",
        3: "Uniform Cost Search (UCS)",
        4: "Greedy Best-First Search (GBFS)",
        5: "A* Search",
        6: "Hill Climbing"
    }

def evolution_algorithms():
    """Display evolution algorithms"""
    return {
        1: "Genetic Algorithm (GA)",
        2: "Differential Evolution (DE)"
    }

def physics_algorithms():
    return {
        1: "Simulated Annealing (SA)"
    }

def swarm_algorithms():
    return {
        1: "Ant Colony Optimization (ACO)",
        2: "Particle Swarm Optimization (PSO)",
        3: "Artificial Bee Colony (ABC)",
        4: "Firefly Algorithm (FA)",
        5: "Cuckoo Search (CS)"
    }

def human_algorithms():
    return {
        1: "Teaching-Learning-Based Optimization (TLBO)"
    }

def print_submenu(algo_dict):
    for i, name in algo_dict.items():
        print(f"{i}. {name}")
    print("0. Back to main menu")

# ==================== CLASSIC ALGORITHMS ====================
def visualize_classic_algo(choice: int):
    if choice == 1:
        visualize_bfs()
    elif choice == 2:
        visualize_dfs()
    elif choice == 3:
        visualize_ucs()
    elif choice == 4:
        visualize_gbfs()
    elif choice == 5:
        visualize_astar()
    elif choice == 6:
        visualize_hill_climbing()

def visualize_bfs():
    print("\n[BFS] Visualizing Discrete Optimization Result...")
    np.random.seed(42)
    solver = BFS(grid_size=20, obstacle_prob=0.15)
    success, path, nodes_exp, duration = solver.search()
    if success:
        plt.figure(figsize=(9, 8))
        plt.imshow(solver.grid, cmap='binary')
        py, px = zip(*path)
        plt.plot(px, py, color='#e74c3c', linewidth=3, label='BFS Path', marker='o', markersize=2)
        plt.scatter(solver.start[1], solver.start[0], color='#2ecc71', s=300, label='Start', zorder=5, edgecolors='black', linewidth=2)
        plt.scatter(solver.goal[1], solver.goal[0], color='#3498db', s=300, label='Goal', zorder=5, edgecolors='black', linewidth=2)
        plt.title(f"BFS Discrete: Shortest Path\nGrid {solver.size}x{solver.size} | Path: {len(path)} | Nodes: {nodes_exp}", fontsize=11, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    input("\nPress ENTER to continue...")

def visualize_dfs():
    print("\n[DFS] Visualizing Discrete Optimization Result...")
    np.random.seed(42)
    solver = DFS(grid_size=20, obstacle_prob=0.15)
    success, path, nodes_exp, duration = solver.search()
    if success:
        plt.figure(figsize=(9, 8))
        plt.imshow(solver.grid, cmap='binary')
        py, px = zip(*path)
        plt.plot(px, py, color='#f39c12', linewidth=3, label='DFS Path', marker='o', markersize=2)
        plt.scatter(solver.start[1], solver.start[0], color='#2ecc71', s=300, label='Start', zorder=5, edgecolors='black', linewidth=2)
        plt.scatter(solver.goal[1], solver.goal[0], color='#3498db', s=300, label='Goal', zorder=5, edgecolors='black', linewidth=2)
        plt.title(f"DFS Discrete: Shortest Path\nGrid {solver.size}x{solver.size} | Path: {len(path)} | Nodes: {nodes_exp}", fontsize=11, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    input("\nPress ENTER to continue...")

def visualize_ucs():
    print("\n[UCS] Visualizing Discrete Optimization Result...")
    np.random.seed(42)
    solver = UCS(grid_size=20, obstacle_prob=0.15)
    success, path, nodes_exp, duration, total_cost = solver.search()
    if success:
        plt.figure(figsize=(9, 8))
        plt.imshow(solver.grid, cmap='binary')
        py, px = zip(*path)
        plt.plot(px, py, color='#9b59b6', linewidth=3, label='UCS Path (Optimal)', marker='o', markersize=2)
        plt.scatter(solver.start[1], solver.start[0], color='#2ecc71', s=300, label='Start', zorder=5, edgecolors='black', linewidth=2)
        plt.scatter(solver.goal[1], solver.goal[0], color='#3498db', s=300, label='Goal', zorder=5, edgecolors='black', linewidth=2)
        plt.title(f"UCS Discrete: Shortest Path\nGrid {solver.size}x{solver.size} | Cost: {total_cost} | Nodes: {nodes_exp}", fontsize=11, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def visualize_gbfs():
    print("\n[GBFS] Visualizing Discrete Optimization Result...")
    np.random.seed(42)
    solver = GBFS(grid_size=20, obstacle_prob=0.15)
    success, path, nodes_exp, duration = solver.search()
    if success:
        plt.figure(figsize=(9, 8))
        plt.imshow(solver.grid, cmap='binary')
        py, px = zip(*path)
        plt.plot(px, py, color='#27ae60', linewidth=3, label='GBFS Path', marker='o', markersize=2)
        plt.scatter(solver.start[1], solver.start[0], color='#2ecc71', s=300, label='Start', zorder=5, edgecolors='black', linewidth=2)
        plt.scatter(solver.goal[1], solver.goal[0], color='#3498db', s=300, label='Goal', zorder=5, edgecolors='black', linewidth=2)
        plt.title(f"GBFS Discrete: Shortest Path\nGrid {solver.size}x{solver.size} | Path: {len(path)} | Nodes: {nodes_exp}", fontsize=11, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def visualize_astar():
    print("\n[A*] Visualizing Discrete Optimization Result...")
    np.random.seed(42)
    solver = AStar(grid_size=20, obstacle_prob=0.15)
    success, path, nodes_exp, duration = solver.search()
    if success:
        plt.figure(figsize=(9, 8))
        plt.imshow(solver.grid, cmap='binary')
        py, px = zip(*path)
        plt.plot(px, py, color='#2980b9', linewidth=3, label='A* Path (Optimal)', marker='o', markersize=2)
        plt.scatter(solver.start[1], solver.start[0], color='#2ecc71', s=300, label='Start', zorder=5, edgecolors='black', linewidth=2)
        plt.scatter(solver.goal[1], solver.goal[0], color='#e74c3c', s=300, label='Goal', zorder=5, edgecolors='black', linewidth=2)
        plt.title(f"A* Discrete: Shortest Path\nGrid {solver.size}x{solver.size} | Path: {len(path)} | Nodes: {nodes_exp}", fontsize=11, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def visualize_hill_climbing():
    print("\n[Hill Climbing] Visualizing Discrete + Continuous Results...")
    np.random.seed(42)
    d_solver = hill_climbing.HillClimbingDiscrete(grid_size=20)
    success, path, nodes_exp, duration = d_solver.run()
    c_solver = hill_climbing.HillClimbingContinuous(hill_climbing.sphere_function, bounds=(-5, 5), step_size=0.1, max_iter=250)
    best_pos, best_val, c_duration = c_solver.run()
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(121)
    ax.imshow(d_solver.grid, cmap='binary')
    if path:
        py, px = zip(*path)
        ax.plot(px, py, color='#e67e22', linewidth=3, label='HC Path', marker='o', markersize=2)
        ax.scatter(d_solver.start[1], d_solver.start[0], color='#2ecc71', s=300, label='Start', zorder=5, edgecolors='black', linewidth=2)
        ax.scatter(d_solver.goal[1], d_solver.goal[0], color='#3498db', s=300, label='Goal', zorder=5, edgecolors='black', linewidth=2)
    ax.set_title(f"Discrete: Shortest Path\nNodes: {nodes_exp} | Path: {len(path)}", fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax = fig.add_subplot(122, projection='3d')
    X_s = np.linspace(-5, 5, 40)
    Y_s = np.linspace(-5, 5, 40)
    X_s, Y_s = np.meshgrid(X_s, Y_s)
    Z_s = X_s**2 + Y_s**2
    ax.plot_surface(X_s, Y_s, Z_s, cmap='viridis', alpha=0.4, linewidth=0.2)
    h_pos = np.array(c_solver.history_pos)
    h_z = np.array([hill_climbing.sphere_function(p) for p in h_pos])
    ax.plot(h_pos[:, 0], h_pos[:, 1], h_z, color='red', marker='o', markersize=4, label='HC Path', linewidth=2)
    ax.scatter([best_pos[0]], [best_pos[1]], [best_val], color='gold', s=300, marker='*', edgecolors='black', linewidth=2, label='Final', zorder=10)
    ax.set_title(f"Continuous: Sphere 3D\nf(x,y) = {best_val}", fontsize=11, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

# ==================== EVOLUTION ALGORITHMS ====================
def visualize_evolution_algo(choice: int):
    if choice == 1:
        visualize_ga()
    elif choice == 2:
        visualize_de()

def visualize_ga():
    print("\n[GA] Visualizing Continuous Optimization Results...")
    np.random.seed(42)
    best_val, elapsed, history = run_ga_rastrigin(mut_rate=0.05)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 2D Rastrigin surface
    ax = axes[0]
    x = np.linspace(-5.12, 5.12, 100)
    y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(x, y)
    Z = 20 + X**2 - 10*np.cos(2*np.pi*X) + Y**2 - 10*np.cos(2*np.pi*Y)
    contour = ax.contourf(X, Y, Z, levels=15, cmap='viridis')
    ax.scatter([0], [0], color='red', s=200, marker='*', edgecolors='black', linewidth=2, label='Optimum', zorder=10)
    ax.set_xlabel('X1', fontweight='bold')
    ax.set_ylabel('X2', fontweight='bold')
    ax.set_title(f'GA Solutions Distribution\nBest: {best_val}', fontweight='bold', fontsize=11)
    plt.colorbar(contour, ax=ax)
    ax.legend()
    ax = axes[1]
    ax.plot(history, color='green', linewidth=2.5, label='GA Convergence')
    ax.set_xlabel('Generation', fontweight='bold')
    ax.set_ylabel('Best Fitness (Log Scale)', fontweight='bold')
    ax.set_title('GA Convergence Curve', fontweight='bold', fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

def visualize_de():
    print("\n[DE] Visualizing Continuous Optimization Results...")
    np.random.seed(42)
    best_val, elapsed, history = run_de_rastrigin(F=0.7)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    x = np.linspace(-5.12, 5.12, 100)
    y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(x, y)
    Z = 20 + X**2 - 10*np.cos(2*np.pi*X) + Y**2 - 10*np.cos(2*np.pi*Y)
    contour = ax.contourf(X, Y, Z, levels=15, cmap='plasma')
    ax.scatter([0], [0], color='cyan', s=200, marker='*', edgecolors='black', linewidth=2, label='Optimum', zorder=10)
    ax.set_xlabel('X1', fontweight='bold')
    ax.set_ylabel('X2', fontweight='bold')
    ax.set_title(f'DE Solutions Distribution\nBest: {best_val}', fontweight='bold', fontsize=11)
    plt.colorbar(contour, ax=ax)
    ax.legend()
    ax = axes[1]
    ax.plot(history, color='blue', linewidth=2.5, label='DE Convergence')
    ax.set_xlabel('Generation', fontweight='bold')
    ax.set_ylabel('Best Fitness (Log Scale)', fontweight='bold')
    ax.set_title('DE Convergence Curve', fontweight='bold', fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

# ==================== PHYSICS ALGORITHMS ====================
def visualize_physics_algo(choice: int):
    if choice == 1:
        visualize_sa()

def visualize_sa():
    print("\n[SA] Visualizing Continuous Optimization Results + 3D Trajectory...")
    np.random.seed(42)
    best_vals, avg_time, history = run_sa_rastrigin_fast(alpha=0.95)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    x = np.linspace(-5.12, 5.12, 100)
    y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(x, y)
    Z = 20 + X**2 - 10*np.cos(2*np.pi*X) + Y**2 - 10*np.cos(2*np.pi*Y)
    contour = ax1.contourf(X, Y, Z, levels=15, cmap='cool')
    ax1.scatter([0], [0], color='yellow', s=200, marker='*', edgecolors='black', linewidth=2, label='Optimum', zorder=10)
    ax1.set_xlabel('X1', fontweight='bold')
    ax1.set_ylabel('X2', fontweight='bold')
    ax1.set_title(f'SA Solutions Distribution\nBest: {np.min(best_vals)}', fontweight='bold', fontsize=11)
    plt.colorbar(contour, ax=ax1)
    ax1.legend()
    ax2.plot(history, color='blue', linewidth=2.5, label='SA Convergence')
    ax2.set_xlabel('Temperature Step', fontweight='bold')
    ax2.set_ylabel('Best Fitness (Log Scale)', fontweight='bold')
    ax2.set_title('SA Convergence Curve', fontweight='bold', fontsize=11)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# ==================== SWARM ALGORITHMS ====================
def visualize_swarm_algo(choice: int):
    if choice == 1:
        visualize_aco()
    elif choice == 2:
        visualize_pso()
    elif choice == 3:
        visualize_abc()
    elif choice == 4:
        visualize_fa()
    elif choice == 5:
        visualize_cs()

def visualize_aco():
    print("\n[ACO] Visualizing Discrete Optimization + Convergence + Pheromone Matrix...")
    np.random.seed(42)
    coords = create_cluster_cities(n_clusters=5, cities_per_cluster=6, seed=42)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    opt = AntColony(dist, n_ants=30, n_iter=100)
    opt.run()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    path = opt.best_path + [opt.best_path[0]]
    ax.plot(coords[path, 0], coords[path, 1], 'o-', color='red', linewidth=2, markersize=8, label='Tour')
    ax.scatter(coords[:, 0], coords[:, 1], c='blue', s=100, alpha=0.6, zorder=5)
    for i in range(len(coords)):
        ax.annotate(str(i), (coords[i, 0], coords[i, 1]), fontsize=8, ha='center')
    ax.set_title(f'ACO Discrete: TSP\nDistance: {opt.best_length:.2f}', fontweight='bold', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax = axes[1]
    ax.plot(opt.history_best, color='blue', linewidth=2.5)
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Best Distance', fontweight='bold')
    ax.set_title('ACO Convergence', fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax = axes[2]
    im = ax.imshow(opt.history_tau[-1], cmap='YlOrBr', aspect='auto')
    ax.set_xlabel('City', fontweight='bold')
    ax.set_ylabel('City', fontweight='bold')
    ax.set_title('Pheromone Matrix (Final)', fontweight='bold', fontsize=11)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

def visualize_pso():
    print("\n[PSO] Visualizing Continuous Optimization Results...")
    np.random.seed(42)
    opt = PSO(func=lambda x: np.sum(x**2), dim=2, bounds=(-5.12, 5.12),
              n_particles=40, n_iter=100, w=0.9, w_min=0.6, c1=2.0, c2=2.0, discrete=False)
    opt.run()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    if hasattr(opt, 'history_pos') and len(opt.history_pos) > 0:
        X_final = np.vstack(opt.history_pos)
    else:
        X_final = np.array([opt.gbest])
    ax.scatter(X_final[:, 0], X_final[:, 1], s=12, alpha=0.4, c='blue', label='Visited')
    ax.scatter([opt.gbest[0]], [opt.gbest[1]], color='gold', s=200, marker='*', edgecolors='black', linewidth=2, label='Best', zorder=10)
    ax.scatter([0], [0], color='black', s=100, marker='x', label='Optimum')
    ax.set_xlabel('X1', fontweight='bold')
    ax.set_ylabel('X2', fontweight='bold')
    ax.set_title(f'PSO Solutions Distribution\nBest: {opt.gbest_val}', fontweight='bold', fontsize=11)
    ax.legend()
    ax = axes[1]
    ax.plot(opt.history_best, color='green', linewidth=2.5)
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Best Fitness (Log Scale)', fontweight='bold')
    ax.set_title('PSO Convergence', fontweight='bold', fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_abc():
    print("\n[ABC] Visualizing Continuous Optimization Results...")
    np.random.seed(42)
    opt = abc_module.ABC(func=lambda x: np.sum(x**2), dim=2, bounds=(-5.12, 5.12),
                         n_foods=20, n_iter=100, limit=5, phi_max=1.0, p_select=2.0)
    opt.run()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    if hasattr(opt, 'history_pos') and len(opt.history_pos) > 0:
        all_pos = np.vstack(opt.history_pos)
    else:
        all_pos = np.array([opt.best])
    ax.scatter(all_pos[:, 0], all_pos[:, 1], s=10, alpha=0.25, c='blue', label='Visited')
    ax.scatter([opt.best[0]], [opt.best[1]], color='cyan', s=300, marker='*', edgecolors='black', linewidth=2, label='Best', zorder=10)
    ax.scatter([0], [0], color='yellow', s=200, marker='*', edgecolors='black', linewidth=2, label='Optimum', zorder=10)
    ax.set_xlabel('X1', fontweight='bold')
    ax.set_ylabel('X2', fontweight='bold')
    ax.set_title(f'ABC Solutions Distribution\nBest: {opt.best_val}', fontweight='bold', fontsize=11)
    ax.legend()
    ax = axes[1]
    ax.plot(opt.history_best, color='orange', linewidth=2.5)
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Best Fitness (Log Scale)', fontweight='bold')
    ax.set_title('ABC Convergence', fontweight='bold', fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_fa():
    print("\n[FA] Visualizing Continuous Optimization Results...")
    np.random.seed(42)
    opt = Firefly(func=lambda x: np.sum(x**2), dim=2, bounds=(-5.12, 5.12), n_iter=100)
    opt.run()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    if hasattr(opt, 'history_pos') and len(opt.history_pos) > 0:
        all_pos = np.vstack(opt.history_pos)
    else:
        all_pos = np.array([opt.best])
    ax.scatter(all_pos[:, 0], all_pos[:, 1], s=12, alpha=0.4, c='blue', label='Visited')
    ax.scatter([opt.best[0]], [opt.best[1]], color='cyan', s=300, marker='*', edgecolors='black', linewidth=2, label='Best', zorder=10)
    ax.scatter([0], [0], color='white', s=200, marker='*', edgecolors='black', linewidth=2, label='Optimum', zorder=10)
    ax.set_xlabel('X1', fontweight='bold')
    ax.set_ylabel('X2', fontweight='bold')
    ax.set_title(f'FA Solutions Distribution\nBest: {opt.best_val }', fontweight='bold', fontsize=11)
    ax.legend()
    ax = axes[1]
    ax.plot(opt.history_best, color='#f39c12', linewidth=2.5)
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Best Fitness (Log Scale)', fontweight='bold')
    ax.set_title('FA Convergence', fontweight='bold', fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_cs():
    print("\n[CS] Visualizing Continuous Optimization Results...")
    np.random.seed(42)
    opt = CuckooSearch(func=lambda x: np.sum(x**2), dim=2, bounds=(-5.12, 5.12), n_iter=100)
    opt.run()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    if hasattr(opt, 'history_pos') and len(opt.history_pos) > 0:
        all_pos = np.vstack(opt.history_pos)
    else:
        all_pos = np.array([opt.best])
    ax.scatter(all_pos[:, 0], all_pos[:, 1], s=12, alpha=0.4, c='blue', label='Visited')
    ax.scatter([opt.best[0]], [opt.best[1]], color='lime', s=300, marker='*', edgecolors='black', linewidth=2, label='Best', zorder=10)
    ax.scatter([0], [0], color='yellow', s=200, marker='*', edgecolors='black', linewidth=2, label='Optimum', zorder=10)
    ax.set_xlabel('X1', fontweight='bold')
    ax.set_ylabel('X2', fontweight='bold')
    ax.set_title(f'CS Solutions Distribution\nBest: {opt.best_val }', fontweight='bold', fontsize=11)
    ax.legend()
    ax = axes[1]
    ax.plot(opt.history_best, color='#9b59b6', linewidth=2.5)
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Best Fitness (Log Scale)', fontweight='bold')
    ax.set_title('CS Convergence', fontweight='bold', fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==================== HUMAN-LEARNING ALGORITHMS ====================
def visualize_human_algo(choice: int):
    if choice == 1:
        visualize_tlbo()

def visualize_tlbo():
    print("\n[TLBO] Visualizing Continuous Optimization + Convergence + Diversity...")
    np.random.seed(42)
    class Problem:
        def __init__(self):
            self.dim = 2
            self.lb = -5.12
            self.ub = 5.12
            self.func = lambda x: np.sum(x**2)
    opt = TLBO(Problem(), pop_size=50, max_iter=100)
    result = opt.run()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    x = np.linspace(-5.12, 5.12, 100)
    y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    contour = ax.contourf(X, Y, Z, levels=15, cmap='RdYlBu_r')
    ax.scatter([opt.best_solution[0]], [opt.best_solution[1]], color='red', s=300, marker='*', edgecolors='black', linewidth=2, label='Best', zorder=10)
    ax.scatter([0], [0], color='green', s=200, marker='*', edgecolors='black', linewidth=2, label='Optimum', zorder=10)
    ax.set_xlabel('X1', fontweight='bold')
    ax.set_ylabel('X2', fontweight='bold')
    ax.set_title(f'TLBO Solutions Distribution\nBest: {opt.best_fitness}', fontweight='bold', fontsize=11)
    plt.colorbar(contour, ax=ax)
    ax.legend()
    ax = axes[1]
    ax.plot(result['convergence'], color='#3498db', linewidth=2.5)
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Best Fitness (Log Scale)', fontweight='bold')
    ax.set_title('TLBO Convergence', fontweight='bold', fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax = axes[2]
    ax.plot(result['diversity'], color='#e74c3c', linewidth=2.5)
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Population Diversity', fontweight='bold')
    ax.set_title('TLBO Population Diversity', fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==================== MAIN ====================
def handle_category(category_name, algo_dict, viz_func):
    while True:
        print("\n" + "="*60)
        print(f"  {category_name}")
        print("="*60)
        print_submenu(algo_dict)
        try:
            choice = int(input("Enter choice: "))
            if choice == 0:
                break
            if choice in algo_dict:
                viz_func(choice)
                input("Press ENTER to continue...")
            else:
                print("\nInvalid choice.")
        except ValueError:
            print("\nInvalid input.")

def main():
    while True:
        main_menu()
        choice = input("Enter choice (A-F): ").strip().upper()
        if choice == 'F':
            print("\nExiting...")
            break
        if choice == 'A':
            handle_category("Classic Searching Algorithms", classic_algorithms(), visualize_classic_algo)
        elif choice == 'B':
            handle_category("Evolution Algorithms", evolution_algorithms(), visualize_evolution_algo)
        elif choice == 'C':
            handle_category("Physics-Based Algorithms", physics_algorithms(), visualize_physics_algo)
        elif choice == 'D':
            handle_category("Swarm Intelligence Algorithms", swarm_algorithms(), visualize_swarm_algo)
        elif choice == 'E':
            handle_category("Human-Based Learning Algorithms", human_algorithms(), visualize_human_algo)
        else:
            print("\nInvalid choice.")

if __name__ == "__main__":
    main()
