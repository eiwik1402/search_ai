import time
import numpy as np
from algo_classic.bfs import BFS
from algo_classic.dfs import DFS
from algo_classic.ucs import UCS
from algo_classic.gbfs import GBFS
from algo_classic.astar import AStar
import algo_classic.hill_climbing as hill_climbing
from algo_evolution.de import run_de_rastrigin
from algo_evolution.ga import run_ga_rastrigin
from algo_evolution.ga import run_ga_tsp
from algo_swarm.aco import AntColony, create_cluster_cities, ACOR
from algo_swarm.pso import PSO
import algo_swarm.abc as abc_module
from algo_swarm.fa import Firefly
from algo_swarm.cs import CuckooSearch
from algo_human_tlbo.tlbo import TLBO
import algo_physics.sa_rastrigin as SA

def main_menu():
    print("\n" + "="*60)
    print("            ALGORITHM DEMONSTRATION MENU")
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

def format_time(seconds):
    if seconds < 0.001:
        return f"{seconds:.6e}s"
    else:
        return f"{seconds:.6f}s"

def run_classic_algo(choice: int):          # Classic algorithms (BFS, DFS, ...)
    print("\n" + "-"*60)
    if choice == 1:
        print("[TEST CASE] Grid size: 20x20, Start: (0,0), Goal: (19,19)")
        print("[SPACE] Search space: (20x20)")
        solver = BFS(grid_size=20)
        start_time = time.perf_counter()
        success, path, nodes_exp, _ = solver.search()
        elapsed = time.perf_counter() - start_time
        print(f"Best length: {len(path) if success else 'N/A'}")
        print(f"Nodes expanded: {nodes_exp}")
        print(f"Success: {success}")
        print(f"Time: {format_time(elapsed)}")
    elif choice == 2:
        print("[TEST CASE] Grid size: 20x20, Start: (0,0), Goal: (19,19)")
        print("[SPACE] Search space: (20x20)")
        solver = DFS(grid_size=20)
        start_time = time.perf_counter()
        success, path, nodes_exp, _ = solver.search()
        elapsed = time.perf_counter() - start_time
        print(f"Best length: {len(path) if success else 'N/A'}")
        print(f"Nodes expanded: {nodes_exp}")
        print(f"Success: {success}")
        print(f"Time: {format_time(elapsed)}")
    elif choice == 3:
        print("[TEST CASE] Grid size: 20x20, Start: (0,0), Goal: (19,19)")
        print("[SPACE] Search space: (20x20)")
        solver = UCS(grid_size=20)
        start_time = time.perf_counter()
        success, path, nodes_exp, cost, _ = solver.search()
        elapsed = time.perf_counter() - start_time
        print(f"Best length: {len(path) if success else 'N/A'}")
        print(f"Best cost: {cost if success else 'N/A'}")
        print(f"Nodes expanded: {nodes_exp}")
        print(f"Success: {success}")
        print(f"Time: {format_time(elapsed)}")
    elif choice == 4:
        print("[TEST CASE] Grid size: 20x20, Start: (0,0), Goal: (19,19)")
        print("[SPACE] Search space: (20x20)")
        solver = GBFS(grid_size=20)
        start_time = time.perf_counter()
        success, path, nodes_exp, _ = solver.search()
        elapsed = time.perf_counter() - start_time
        print(f"Best length: {len(path) if success else 'N/A'}")
        print(f"Nodes expanded: {nodes_exp}")
        print(f"Success: {success}")
        print(f"Time: {format_time(elapsed)}")
    elif choice == 5:
        print("[TEST CASE] Grid size: 20x20, Start: (0,0), Goal: (19,19)")
        print("[SPACE] Search space: (20x20)")
        solver = AStar(grid_size=20)
        start_time = time.perf_counter()
        success, path, nodes_exp, _ = solver.search()
        elapsed = time.perf_counter() - start_time
        print(f"Best length: {len(path) if success else 'N/A'}")
        print(f"Nodes expanded: {nodes_exp}")
        print(f"Success: {success}")
        print(f"Time: {format_time(elapsed)}")
    elif choice == 6:
        print("[TEST CASE] Hill Climbing Discrete: Shortest Path")
        print("[SPACE] Grid size: 20x20, Start: (0,0), Goal: (19,19)")
        d_solver = hill_climbing.HillClimbingDiscrete(grid_size=20)
        d_success, d_path, d_nodes, d_time = d_solver.run()        
        print(f"Best length: {len(d_path) if d_success else 'N/A'}")
        print(f"Nodes expanded: {d_nodes}")
        print(f"Success: {d_success}")
        print(f"Time: {format_time(d_time)}")        
        print("-" * 30)
        print("[TEST CASE] Hill Climbing Continuous: Sphere function")
        print("[SPACE] Search space: (-5, 5)^2")
        c_solver = hill_climbing.HillClimbingContinuous(hill_climbing.sphere_function)
        c_best_sol, c_best_val, c_time = c_solver.run()        
        print(f"Best value: {c_best_val:.6f}")
        print(f"Best solution: {c_best_sol}")
        print(f"Success: True")
        print(f"Time: {format_time(c_time)}")
    print("-"*60)

def run_evolution_algo(choice: int):        # Evolution-based algorithms (GA, DE)
    print("\n" + "-" * 60)
    def sphere(x):
        return np.sum(x**2)
    np.random.seed(42)
    if choice == 1:
        print("[TEST CASE] GA: Rastrigin function")
        print("[DIM = 5, POP = 100, GENS = 200]")
        print("[SPACE] Search space: (-5.12, 5.12)^5")
        np.random.seed(42)
        start = time.perf_counter()
        best_val, elapsed, _ = run_ga_rastrigin(mut_rate=0.05)
        print(f"Best value: {best_val}")
        print(f"Time: {format_time(elapsed)}")
        print("\n[TEST CASE] GA: TSP (Discrete)")
        print("[SPACE] Permutation of cities")
        start_time = time.perf_counter()
        best_dist, best_path, elapsed, _ = run_ga_tsp(mut_rate=0.05)
        print(f"Best distance: {best_dist}")
        print(f"Best path: {best_path}")
        print(f"Time: {format_time(elapsed)}")
    elif choice == 2:
        print("[TEST CASE] DE: Rastrigin function")
        print("[DIM = 5, POP = 50, GENS = 200]")
        print("[SPACE] Search space: (-5.12, 5.12)^5")
        np.random.seed(42)
        start = time.perf_counter()
        best_val, elapsed, _ = run_de_rastrigin(F=0.7)
        print(f"Best value: {best_val}")
        print(f"Time: {format_time(elapsed)}")
    print("-" * 60)

def run_physics_algo(choice: int):          # Physics-based algorithms (SA)
    print("\n" + "-"*60)
    if choice == 1:
        print("[TEST CASE] SA: Rastrigin function, T_init=1000, T_min=1e-4")
        print("[SPACE] Search space: (-5.12, 5.12)^5")
        try:
            start_time = time.perf_counter()
            best_vals, avg_time, history = SA.run_sa_rastrigin_fast(0.95)
            elapsed = time.perf_counter() - start_time
            print(f"Best value: {np.min(best_vals)}")
            print(f"Best solution: {best_vals[:2]}")
            print(f"Time: {format_time(elapsed)}")
        except Exception as e:
            print(f"Error: {str(e)}")
    print("-"*60)

def run_swarm_algo(choice: int):            # Swarm algorithms (ACO, PSO, ...)
    print("\n" + "-"*60)
    if choice == 1:
        print("[TEST CASE] ACO (TSP - Clustered cities)")
        print("[SPACE] 5 clusters, 6 cities per cluster")
        coords = create_cluster_cities(n_clusters=5,cities_per_cluster=6,seed=42)
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :],axis=2)
        start_time = time.perf_counter()
        opt = AntColony(dist, n_ants=30, n_iter=100)
        opt.run()
        elapsed = time.perf_counter() - start_time
        print(f"Best tour length: {opt.best_length}")
        print(f"Best tour: {opt.best_path}")
        print(f"Time: {format_time(elapsed)}")
        print("\n[TEST CASE] ACOR: Sphere function (Continuous)")
        print("[DIM = 2, Ants = 30, Iterations = 100]")
        print("[SPACE] Search space: (-5, 5)^2")
        def sphere(x):
            return np.sum(x ** 2)
        start_time = time.perf_counter()
        opt = ACOR(
            func=sphere,
            dim=2,
            bounds=(-5, 5),
            n_ants=30,
            n_iter=100
        )
        opt.run()
        elapsed = time.perf_counter() - start_time
        print(f"Best value: {opt.best_val}")
        print(f"Best solution: {opt.best}")
        print(f"Time: {format_time(elapsed)}")
    elif choice == 2:
        print("[TEST CASE] PSO: Sphere function (continuous), Dim=2, n_particles=20, Iterations=100")
        print("[SPACE] Search space: (-5.12, 5.12)^2")
        np.random.seed(42)
        start_time = time.perf_counter()
        opt = PSO(func=lambda x: np.sum(x**2),dim=2,bounds=(-5.12, 5.12),n_particles=20,n_iter=100,w=0.9,w_min=0.6,c1=2.0,c2=2.0,discrete=False)
        opt.run()
        elapsed = time.perf_counter() - start_time
        print(f"Best value: {opt.gbest_val}")
        print(f"Best solution: {opt.gbest}")
        print(f"Time: {format_time(elapsed)}")
    elif choice == 3:
        print("[TEST CASE] ABC: Sphere function, Dim=2, Food sources=10, Iterations=100")
        print("[SPACE] Search space: (-5.12, 5.12)^2")
        start_time = time.perf_counter()
        opt = abc_module.ABC(func=lambda x: np.sum(x**2), dim=2, bounds=(-5.12,5.12), n_foods=20, n_iter=100, limit=5, phi_max=1.0, p_select=2.0)
        opt.run()
        elapsed = time.perf_counter() - start_time
        print(f"Best value: {opt.best_val if hasattr(opt, 'best_val') else 'N/A'}")
        print(f"Best solution: {opt.best if hasattr(opt, 'best') else 'N/A'}")
        print(f"Time: {format_time(elapsed)}")
    elif choice == 4:
        print("[TEST CASE] FA: Sphere function, Dim=2, Iterations=100")
        print("[SPACE] Search space: (-5.12, 5.12)^2")
        start_time = time.perf_counter()
        opt = Firefly(func=lambda x: np.sum(x**2), dim=2, bounds=(-5.12,5.12), n_iter=100)
        opt.run()
        elapsed = time.perf_counter() - start_time
        print(f"Best value: {opt.best_val if hasattr(opt, 'best_val') else 'N/A'}")
        print(f"Best solution: {opt.best if hasattr(opt, 'best') else 'N/A'}")
        print(f"Time: {format_time(elapsed)}")
    elif choice == 5:
        print("[TEST CASE] CS: Sphere function, Dim=2, Iterations=100")
        print("[SPACE] Search space: (-5.12, 5.12)^2")
        start_time = time.perf_counter()
        opt = CuckooSearch(func=lambda x: np.sum(x**2), dim=2, bounds=(-5.12,5.12), n_iter=100)
        opt.run()
        elapsed = time.perf_counter() - start_time
        print(f"Best value: {opt.best_val if hasattr(opt, 'best_val') else 'N/A'}")
        print(f"Best solution: {opt.best if hasattr(opt, 'best') else 'N/A'}")
        print(f"Time: {format_time(elapsed)}")
    print("-"*60)

def run_human_algo(choice: int):            # Human-based algorithms (TLBO)
    print("\n" + "-"*60)
    if choice == 1:
        print("[TEST CASE] TLBO: Sphere function")
        print("[DIM = 5, ITER = 50]")
        print("[SPACE] Search space: (-5.12, 5.12)^5")
        class Problem:
            def __init__(self):
                self.dim = 5
                self.lb = -5.12
                self.ub = 5.12
                self.func = lambda x: np.sum(x**2)
        opt = TLBO(Problem(), max_iter=50)
        result = opt.run()
        best_val = result.get("best_fitness", "N/A")
        best_sol = getattr(opt, "best_solution", "N/A")
        runtime  = result.get("runtime", None)
        print(f"Best value: {best_val}")
        print(f"Best solution: {best_sol}")
        print(f"Time: {format_time(runtime) if runtime else 'N/A'}")
    print("-"*60)

def main():
    while True:
        main_menu()
        choice = input("Enter choice (A-F): ").strip().upper()
        if choice == 'F':
            print("\nExiting program...")
            break
        if choice == 'A':
            handle_category("Classic Searching Algorithms", classic_algorithms(), run_classic_algo)
        elif choice == 'B':
            handle_category("Evolution Algorithms", evolution_algorithms(), run_evolution_algo)
        elif choice == 'C':
            handle_category("Physics-Based Algorithms", physics_algorithms(), run_physics_algo)
        elif choice == 'D':
            handle_category("Swarm Intelligence Algorithms", swarm_algorithms(), run_swarm_algo)
        elif choice == 'E':
            handle_category("Human-Based Learning Algorithms", human_algorithms(), run_human_algo)
        else:
            print("\nInvalid choice. Please try again.")

def handle_category(category_name, algo_dict, run_func):
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
                run_func(choice)
                input("Press ENTER to continue...")
            else:
                print("\nInvalid choice. Please try again.")
        except ValueError:
            print("\nInvalid input. Please enter a number.")

if __name__ == "__main__":
    main()