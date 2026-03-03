import time
import numpy as np
from algorithms import (
    classic_algorithms, evolution_algorithms, physics_algorithms, 
    swarm_algorithms, human_algorithms, print_submenu,
    run_classic_algo, run_evolution_algo, run_physics_algo,
    run_swarm_algo, run_human_algo, format_time
)

NUM_RUNS = 30

def collect_classic_results(choice: int):
    """Run classic algorithm and extract metrics"""
    results = []
    for _ in range(NUM_RUNS):
        start_time = time.perf_counter()
        
        if choice == 1:  # BFS
            from algo_classic.bfs import BFS
            solver = BFS(grid_size=20)
            success, path, nodes_exp, _ = solver.search()
            elapsed = time.perf_counter() - start_time
            best_val = len(path) if success else float('inf')
            
        elif choice == 2:  # DFS
            from algo_classic.dfs import DFS
            solver = DFS(grid_size=20)
            success, path, nodes_exp, _ = solver.search()
            elapsed = time.perf_counter() - start_time
            best_val = len(path) if success else float('inf')
            
        elif choice == 3:  # UCS
            from algo_classic.ucs import UCS
            solver = UCS(grid_size=20)
            success, path, nodes_exp, cost, _ = solver.search()
            elapsed = time.perf_counter() - start_time
            best_val = len(path) if success else float('inf')
            
        elif choice == 4:  # GBFS
            from algo_classic.gbfs import GBFS
            solver = GBFS(grid_size=20)
            success, path, nodes_exp, _ = solver.search()
            elapsed = time.perf_counter() - start_time
            best_val = len(path) if success else float('inf')
            
        elif choice == 5:  # A*
            from algo_classic.astar import AStar
            solver = AStar(grid_size=20)
            success, path, nodes_exp, _ = solver.search()
            elapsed = time.perf_counter() - start_time
            best_val = len(path) if success else float('inf')
            
        elif choice == 6:  # Hill Climbing
            import algo_classic.hill_climbing as hill_climbing
            d_solver = hill_climbing.HillClimbingDiscrete(grid_size=20)
            d_success, d_path, d_nodes, d_time = d_solver.run()
            d_best_val = len(d_path) if d_success else float('inf')
            
            elapsed = d_time
            success = d_success
            best_val = d_best_val
        
        results.append({
            'best_val': best_val,
            'elapsed': elapsed,
            'success': success
        })
    
    return results

def collect_evolution_results(choice: int):
    """Run evolution algorithm and extract metrics"""
    from algo_evolution.de import run_de_rastrigin
    from algo_evolution.ga import run_ga_rastrigin, run_ga_tsp
    
    results = []
    for _ in range(NUM_RUNS):
        if choice == 1:  # GA
            np.random.seed()  # Different seed each run
            start = time.perf_counter()
            best_val, elapsed, _ = run_ga_rastrigin(mut_rate=0.05)
            elapsed = time.perf_counter() - start
            
        elif choice == 2:  # DE
            np.random.seed()
            start = time.perf_counter()
            best_val, elapsed, _ = run_de_rastrigin(F=0.7)
            elapsed = time.perf_counter() - start
        
        results.append({
            'best_val': best_val,
            'elapsed': elapsed,
            'success': True  # Optimization algorithms don't have clear success threshold
        })
    
    return results

def collect_physics_results(choice: int):
    """Run physics algorithm and extract metrics"""
    import algo_physics.sa_rastrigin as SA
    
    results = []
    for _ in range(NUM_RUNS):
        if choice == 1:  # SA
            start_time = time.perf_counter()
            best_vals, avg_time, history = SA.run_sa_rastrigin_fast(0.95)
            elapsed = time.perf_counter() - start_time
            best_val = np.min(best_vals)
            
            results.append({
                'best_val': best_val,
                'elapsed': elapsed,
                'success': True
            })
    
    return results

def collect_swarm_results(choice: int):
    """Run swarm algorithm and extract metrics"""
    from algo_swarm.aco import AntColony, create_cluster_cities, ACOR
    from algo_swarm.pso import PSO
    import algo_swarm.abc as abc_module
    from algo_swarm.fa import Firefly
    from algo_swarm.cs import CuckooSearch
    
    results = []
    for _ in range(NUM_RUNS):
        if choice == 1:  # ACO
            coords = create_cluster_cities(n_clusters=5, cities_per_cluster=6, seed=None)
            dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
            start_time = time.perf_counter()
            opt = AntColony(dist, n_ants=30, n_iter=100)
            opt.run()
            elapsed = time.perf_counter() - start_time
            best_val = opt.best_length
            
        elif choice == 2:  # PSO
            np.random.seed()
            start_time = time.perf_counter()
            opt = PSO(func=lambda x: np.sum(x**2), dim=2, bounds=(-5.12, 5.12), 
                     n_particles=20, n_iter=100, w=0.9, w_min=0.6, c1=2.0, c2=2.0, discrete=False)
            opt.run()
            elapsed = time.perf_counter() - start_time
            best_val = opt.gbest_val
            
        elif choice == 3:  # ABC
            start_time = time.perf_counter()
            opt = abc_module.ABC(func=lambda x: np.sum(x**2), dim=2, bounds=(-5.12, 5.12), 
                                n_foods=20, n_iter=100, limit=5, phi_max=1.0, p_select=2.0)
            opt.run()
            elapsed = time.perf_counter() - start_time
            best_val = opt.best_val if hasattr(opt, 'best_val') else float('inf')
            
        elif choice == 4:  # FA
            start_time = time.perf_counter()
            opt = Firefly(func=lambda x: np.sum(x**2), dim=2, bounds=(-5.12, 5.12), n_iter=100)
            opt.run()
            elapsed = time.perf_counter() - start_time
            best_val = opt.best_val if hasattr(opt, 'best_val') else float('inf')
            
        elif choice == 5:  # CS
            start_time = time.perf_counter()
            opt = CuckooSearch(func=lambda x: np.sum(x**2), dim=2, bounds=(-5.12, 5.12), n_iter=100)
            opt.run()
            elapsed = time.perf_counter() - start_time
            best_val = opt.best_val if hasattr(opt, 'best_val') else float('inf')
        
        results.append({
            'best_val': best_val,
            'elapsed': elapsed,
            'success': True
        })
    
    return results

def collect_human_results(choice: int):
    """Run human-based algorithm and extract metrics"""
    from algo_human_tlbo.tlbo import TLBO
    
    results = []
    for _ in range(NUM_RUNS):
        if choice == 1:  # TLBO
            class Problem:
                def __init__(self):
                    self.dim = 5
                    self.lb = -5.12
                    self.ub = 5.12
                    self.func = lambda x: np.sum(x**2)
            
            start_time = time.perf_counter()
            opt = TLBO(Problem(), max_iter=50)
            result = opt.run()
            elapsed = time.perf_counter() - start_time
            best_val = result.get("best_fitness", float('inf'))
            
            results.append({
                'best_val': best_val,
                'elapsed': elapsed,
                'success': best_val != float('inf')
            })
    
    return results

def calculate_statistics(results):
    best_vals = [r['best_val'] for r in results if r['best_val'] != float('inf')]
    elapsed_times = [r['elapsed'] for r in results]
    successes = [r['success'] for r in results]
    
    if len(best_vals) == 0:
        return None
    
    stats = {
        'mean': np.mean(best_vals),
        'std': np.std(best_vals),
        'best': np.min(best_vals),
        'worst': np.max(best_vals),
        'success_rate': (sum(successes) / len(successes)) * 100,
        'avg_time': np.mean(elapsed_times)
    }
    
    return stats

def print_statistics(algo_name, stats):
    if stats is None:
        print(f"No valid results for {algo_name}")
        return
    
    print("\n" + "="*60)
    print(f"  STATISTICS: {algo_name}")
    print(f"  (Runs: {NUM_RUNS})")
    print("="*60)
    print(f"Mean value: {stats['mean']}")
    print(f"Std: {stats['std']}")
    print(f"Best value: {stats['best']}")
    print(f"Worst value: {stats['worst']}")
    print(f"Success rate: {stats['success_rate']:.2f}%")
    print(f"Average time elapsed: {format_time(stats['avg_time'])}")
    print("="*60)

def main():
    """Main menu for statistics collection"""
    while True:
        print("\n" + "="*60)
        print("         ALGORITHM STATISTICS (30 runs each)")
        print("="*60)
        print("Select algorithm category:")
        print("A. Classic Searching Algorithms - Discrete Problem Demo")
        print("B. Evolution Algorithms - Continuous Problem Demo")
        print("C. Physics-Based Algorithms - Continuous Problem Demo")
        print("D. Swarm Intelligence Algorithms - Continuous Problem Demo")
        print("E. Human-Based Learning Algorithms - Continuous Problem Demo")
        print("F. Exit")
        print("="*60)
        
        choice = input("Enter choice (A-F): ").strip().upper()
        
        if choice == 'F':
            print("\nExiting program...")
            break
        
        elif choice == 'A':
            while True:
                algo_dict = classic_algorithms()
                print("\n" + "="*60)
                print("  Classic Searching Algorithms")
                print("="*60)
                print_submenu(algo_dict)
                try:
                    algo_choice = int(input("Enter choice: "))
                    if algo_choice == 0:
                        break
                    elif algo_choice in algo_dict:
                        algo_name = algo_dict[algo_choice]
                        print(f"\nRunning {algo_name} {NUM_RUNS} times...")
                        results = collect_classic_results(algo_choice)
                        stats = calculate_statistics(results)
                        print_statistics(algo_name, stats)
                        input("Press ENTER to continue...")
                    else:
                        print("Invalid choice.")
                except ValueError:
                    print("Invalid input.")
        
        elif choice == 'B':
            while True:
                algo_dict = evolution_algorithms()
                print("\n" + "="*60)
                print("  Evolution Algorithms")
                print("="*60)
                print_submenu(algo_dict)
                try:
                    algo_choice = int(input("Enter choice: "))
                    if algo_choice == 0:
                        break
                    elif algo_choice in algo_dict:
                        algo_name = algo_dict[algo_choice]
                        print(f"\nRunning {algo_name} {NUM_RUNS} times...")
                        results = collect_evolution_results(algo_choice)
                        stats = calculate_statistics(results)
                        print_statistics(algo_name, stats)
                        input("Press ENTER to continue...")
                    else:
                        print("Invalid choice.")
                except ValueError:
                    print("Invalid input.")
        
        elif choice == 'C':
            while True:
                algo_dict = physics_algorithms()
                print("\n" + "="*60)
                print("  Physics-Based Algorithms")
                print("="*60)
                print_submenu(algo_dict)
                try:
                    algo_choice = int(input("Enter choice: "))
                    if algo_choice == 0:
                        break
                    elif algo_choice in algo_dict:
                        algo_name = algo_dict[algo_choice]
                        print(f"\nRunning {algo_name} {NUM_RUNS} times...")
                        results = collect_physics_results(algo_choice)
                        stats = calculate_statistics(results)
                        print_statistics(algo_name, stats)
                        input("Press ENTER to continue...")
                    else:
                        print("Invalid choice.")
                except ValueError:
                    print("Invalid input.")
        
        elif choice == 'D':
            while True:
                algo_dict = swarm_algorithms()
                print("\n" + "="*60)
                print("  Swarm Intelligence Algorithms")
                print("="*60)
                print_submenu(algo_dict)
                try:
                    algo_choice = int(input("Enter choice: "))
                    if algo_choice == 0:
                        break
                    elif algo_choice in algo_dict:
                        algo_name = algo_dict[algo_choice]
                        print(f"\nRunning {algo_name} {NUM_RUNS} times...")
                        results = collect_swarm_results(algo_choice)
                        stats = calculate_statistics(results)
                        print_statistics(algo_name, stats)
                        input("Press ENTER to continue...")
                    else:
                        print("Invalid choice.")
                except ValueError:
                    print("Invalid input.")
        
        elif choice == 'E':
            while True:
                algo_dict = human_algorithms()
                print("\n" + "="*60)
                print("  Human-Based Learning Algorithms")
                print("="*60)
                print_submenu(algo_dict)
                try:
                    algo_choice = int(input("Enter choice: "))
                    if algo_choice == 0:
                        break
                    elif algo_choice in algo_dict:
                        algo_name = algo_dict[algo_choice]
                        print(f"\nRunning {algo_name} {NUM_RUNS} times...")
                        results = collect_human_results(algo_choice)
                        stats = calculate_statistics(results)
                        print_statistics(algo_name, stats)
                        input("Press ENTER to continue...")
                    else:
                        print("Invalid choice.")
                except ValueError:
                    print("Invalid input.")
        
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()
