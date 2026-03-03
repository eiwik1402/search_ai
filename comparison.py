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
from algo_swarm.aco import AntColony, create_cluster_cities
from algo_swarm.aco_path import ACO_Greedy_Path, generate_grid
from algo_swarm.pso import PSO
import algo_swarm.abc as abc_module
from algo_swarm.fa import Firefly
from algo_swarm.cs import CuckooSearch
from algo_human_tlbo.tlbo import TLBO
import algo_physics.sa_rastrigin as SA

class Problem:
    def __init__(self):
        self.dim = 2
        self.lb = -5.12
        self.ub = 5.12
        self.func = lambda x: np.sum(x**2)
        
def format_time(seconds):
    return f"{seconds:.6f}s"

def print_table(headers, rows):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    header_line = ' | '.join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print('-' * len(header_line))
    
    # Print rows
    for row in rows:
        print(' | '.join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)))

def comparison_menu() -> None:
    """Main comparison menu"""
    while True:
        print('\n' + '=' * 70)
        print(' ALGORITHM COMPARISON MENU')
        print('=' * 70)
        print('1. Shortest Path (BFS, DFS, UCS, GBFS, A*, HC, ACO)')
        print('2. Travelling Salesman Problem (ACO)')
        print('3. Knapsack Problem (PSO, CS, FA)')
        print('4. Rastrigin function (DE, GA, SA, HC)')
        print('5. Sphere function (PSO, ABC, FA, CS, SA, HC, TLBO)')
        print('0. Exit')
        try:
            choice = int(input('Enter choice: ').strip())
        except ValueError:
            print('Invalid input.')
            continue

        if choice == 0:
            break
        elif choice == 1:
            comparison_option_1()
        elif choice == 2:
            comparison_option_2()
        elif choice == 3:
            comparison_option_3()
        elif choice == 4:
            comparison_option_4()
        elif choice == 5:
            comparison_option_5()
        else:
            print('Invalid choice.')

def comparison_option_1() -> None:
    print('\nOPTION 1: CLASSIC ALGORITHMS COMPARISON')
    print('Test Case: 30x30 Grid, Obstacle Probability = 0.15')
    results = []
    # Test BFS
    solver = BFS(grid_size=30)
    start = time.perf_counter()
    success, path, nodes_exp, _ = solver.search()
    elapsed = time.perf_counter() - start
    results.append(['BFS', len(path) if success else 'Failed', format_time(elapsed)])
    
    # Test DFS
    solver = DFS(grid_size=30)
    start = time.perf_counter()
    success, path, nodes_exp, _ = solver.search()
    elapsed = time.perf_counter() - start
    results.append(['DFS', len(path) if success else 'Failed', format_time(elapsed)])
    
    # Test UCS
    solver = UCS(grid_size=30)
    start = time.perf_counter()
    success, path, nodes_exp, cost, _ = solver.search()
    elapsed = time.perf_counter() - start
    results.append(['UCS', len(path) if success else 'Failed', format_time(elapsed)])
    
    # Test GBFS
    solver = GBFS(grid_size=30)
    start = time.perf_counter()
    success, path, nodes_exp, _ = solver.search()
    elapsed = time.perf_counter() - start
    results.append(['GBFS', len(path) if success else 'Failed', format_time(elapsed)])
    
    # Test A*
    solver = AStar(grid_size=30)
    start = time.perf_counter()
    success, path, nodes_exp, _ = solver.search()
    elapsed = time.perf_counter() - start
    results.append(['A*', len(path) if success else 'Failed', format_time(elapsed)])
    
    # Test Hill Climbing
    solver = hill_climbing.HillClimbingDiscrete(grid_size=30, obstacle_prob=0.15)
    success, path, nodes_exp, elapsed_hc = solver.run()
    results.append(['Hill Climbing', len(path) if success else 'Failed', format_time(elapsed_hc)])
    
    # Test ACO (Grid-based path finding)
    grid = generate_grid(size=30, obstacle_prob=0.15, seed=42)
    start = time.perf_counter()
    aco = ACO_Greedy_Path(grid, alpha=1.0, beta=2.0)
    success, path_len, nodes_exp, elapsed = aco.search()
    results.append(['ACO', path_len if success else 'Failed', format_time(elapsed)])
    print('\n==================RESULTS===================')
    headers = ['Algorithm Name', 'Optimal Result', 'Runtime']
    print_table(headers, results)
    input("\nPress ENTER to continue...")

def comparison_option_2() -> None:
    print('\nOPTION 2: ACO PARAMETER COMPARISON')
    print('Test Case 1: Varying Cluster Count (Cities Per Cluster = 6)')
    print('Test Case 2: Varying Cities Per Cluster (Cluster Count = 5)')
    
    # Test Case 1: Varying clusters
    print('\nRunning test case 1...')
    results_clusters = []

    for n_clusters in [5, 10, 20]:
        coords = create_cluster_cities(n_clusters=n_clusters, cities_per_cluster=6, seed=42)
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        
        start = time.perf_counter()
        opt = AntColony(dist, n_ants=30, n_iter=100)
        opt.run()
        elapsed = time.perf_counter() - start
        
        results_clusters.append([n_clusters, 6, f"{opt.best_length:.4f}", format_time(elapsed)])
        
    # Test Case 2: Varying cities per cluster
    print('Running test case 2...')
    results_cities = []
    for cities_per_cluster in [6, 12, 20]:
        coords = create_cluster_cities(n_clusters=5, cities_per_cluster=cities_per_cluster, seed=42)
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        start = time.perf_counter()
        opt = AntColony(dist, n_ants=30, n_iter=100)
        opt.run()
        elapsed = time.perf_counter() - start
        results_cities.append([5, cities_per_cluster, f"{opt.best_length:.4f}", format_time(elapsed)])
        
    all_results = results_clusters + results_cities
    print('\n==================RESULTS===================')
    headers = ['Number of Clusters', 'Cities Per Cluster', 'Best Tour Length', 'Runtime']
    print_table(headers, all_results)
    input("\nPress ENTER to continue...")

def comparison_option_3() -> None:
    print('\nOPTION 3: BINARY SWARM ALGORITHMS COMPARISON')
    print('Test Case: Binary Knapsack Problem (20 items, capacity = 50)')
    np.random.seed(42)
    n_items = 20
    weights = np.random.randint(5, 20, n_items)
    values = np.random.randint(10, 50, n_items)
    capacity = 50
    
    def knapsack_objective(x):
        total_weight = np.sum(x * weights)
        total_value = np.sum(x * values)
        if total_weight <= capacity:
            return total_value
        else:
            return total_value * 0.1  # Penalty
    
    results = []
    
    # Test PSO (Binary mode)
    start = time.perf_counter()
    pso = PSO(func=knapsack_objective, dim=n_items, bounds=[0, 1],
              n_particles=20, n_iter=100, discrete=True)
    pso.run()
    elapsed = time.perf_counter() - start
    results.append(['PSO (Binary)', f"{pso.gbest_val:.2f}", format_time(elapsed)])
    
    # Test Firefly (Binary mode)
    start = time.perf_counter()
    def fa_binary_obj(x):
        return -knapsack_objective(np.round(x))
    fa = Firefly(func=fa_binary_obj, dim=n_items, bounds=[0, 1],
                 n_fireflies=20, n_iter=100)
    fa.run()
    elapsed = time.perf_counter() - start
    results.append(['Firefly', f"{-fa.best_val:.2f}", format_time(elapsed)])
    
    # Test Cuckoo Search (Binary mode)
    start = time.perf_counter()
    cs = CuckooSearch(func=fa_binary_obj, dim=n_items, bounds=[0, 1],
                      n_nests=20, n_iter=100)
    cs.run()
    elapsed = time.perf_counter() - start
    results.append(['Cuckoo Search', f"{-cs.best_val:.2f}", format_time(elapsed)])
    # Print results
    print('\n==================RESULTS===================')
    headers = ['Algorithm Name', 'Best Value', 'Runtime']
    print_table(headers, results)
    input("\nPress ENTER to continue...")

def comparison_option_4() -> None:
    print('OPTION 4: CONTINUOUS ALGORITHMS COMPARISON')
    print('Test Case: Rastrigin Function (5D)')
    print('Search Space: (-5.12, 5.12)^5')
    results = []

    # Test DE
    start = time.perf_counter()
    best_val, elapsed, _ = run_de_rastrigin(F=0.7)
    results.append(['DE', f"{best_val}", format_time(elapsed)])
    
    # Test GA
    start = time.perf_counter()
    best_val, elapsed, _ = run_ga_rastrigin(mut_rate=0.05)
    results.append(['GA', f"{best_val}", format_time(elapsed)])
    
    # Test SA
    start = time.perf_counter()
    best_vals, avg_time, _ = SA.run_sa_rastrigin_fast(0.95)
    elapsed = time.perf_counter() - start
    results.append(['SA', f"{np.min(best_vals)}", format_time(elapsed)])
    
    # Test Hill Climbing
    def rastrigin_func(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    solver = hill_climbing.HillClimbingContinuous(func=rastrigin_func, bounds=(-5.12, 5.12))
    start = time.perf_counter()
    best_pos, best_val, elapsed_hc = solver.run()
    results.append(['Hill Climbing', f"{best_val}", format_time(elapsed_hc)])
    
    print('\n==================RESULTS===================')
    headers = ['Algorithm Name', 'Best Value', 'Runtime']
    print_table(headers, results)
    input("\nPress ENTER to continue...")

def comparison_option_5() -> None:
    print('OPTION 5: ALL ALGORITHMS COMPARISON')
    print('Test Case: Continuous Optimization in 2D')
    print('Search Space: (-5.12, 5.12)^2')
    print('-' * 70)
    
    def sphere_2d(x):
        return np.sum(x**2)
    results = []

    # Test PSO
    np.random.seed(42)
    start = time.perf_counter()
    pso = PSO(func=sphere_2d, dim=2, bounds=[-5.12, 5.12],
              n_particles=20, n_iter=100)
    pso.run()
    elapsed = time.perf_counter() - start
    results.append(['PSO', f"{pso.gbest_val}", format_time(elapsed)])
    
    # Test ABC
    np.random.seed(42)
    start = time.perf_counter()
    abc = abc_module.ABC(func=sphere_2d, dim=2, bounds=[-5.12, 5.12],
                         n_foods=20, n_iter=100, phi_max=1.0, p_select=2.0, limit=5)
    abc.run()
    elapsed = time.perf_counter() - start
    results.append(['ABC', f"{abc.best_val}", format_time(elapsed)])
    
    # Test Firefly
    np.random.seed(42)
    start = time.perf_counter()
    fa = Firefly(func=sphere_2d, dim=2, bounds=[-5.12, 5.12],
                 n_fireflies=20, n_iter=100)
    fa.run()
    elapsed = time.perf_counter() - start
    results.append(['Firefly', f"{fa.best_val}", format_time(elapsed)])
    
    # Test Cuckoo Search
    np.random.seed(42)
    start = time.perf_counter()
    cs = CuckooSearch(func=sphere_2d, dim=2, bounds=[-5.12, 5.12],
                      n_nests=20, n_iter=100)
    cs.run()
    elapsed = time.perf_counter() - start
    results.append(['Cuckoo Search', f"{cs.best_val}", format_time(elapsed)])
    
    # Test SA
    np.random.seed(42)
    start = time.perf_counter()
    # Modified SA for 2D sphere
    T_INIT = 100.0
    T_MIN = 0.001
    alpha = 0.99
    curr_x = np.random.uniform(-5.12, 5.12, 2)
    curr_val = sphere_2d(curr_x)
    best_x, best_val = curr_x.copy(), curr_val
    temp = T_INIT
    
    while temp > T_MIN:
        for _ in range(1):
            candidate_x = np.clip(curr_x + np.random.uniform(-0.5, 0.5, 2), -5.12, 5.12)
            candidate_val = sphere_2d(candidate_x)
            delta = candidate_val - curr_val
            
            if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                curr_x, curr_val = candidate_x, candidate_val
            
            if curr_val < best_val:
                best_x, best_val = curr_x.copy(), curr_val
        
        temp *= alpha
    
    elapsed = time.perf_counter() - start
    results.append(['SA', f"{best_val}", format_time(elapsed)])
    
    # Test Hill Climbing
    np.random.seed(42)
    start = time.perf_counter()
    solver = hill_climbing.HillClimbingContinuous(func=sphere_2d, bounds=(-5.12, 5.12))
    best_pos, best_val, elapsed_hc = solver.run()
    results.append(['Hill Climbing', f"{best_val}", format_time(elapsed_hc)])
    
    # Test TLBO
    np.random.seed(42)
    start = time.perf_counter()
    problem = Problem()
    tlbo = TLBO(problem, pop_size=50, max_iter=40)
    result = tlbo.run()

    elapsed = time.perf_counter() - start
    results.append(['TLBO', f"{result['best_fitness']}", format_time(elapsed)])
    print('\n==================RESULTS===================')
    headers = ['Algorithm Name', 'Best Value', 'Runtime']
    print_table(headers, results)
    input("\nPress ENTER to continue...")

def main_menu() -> None:
    comparison_menu()

if __name__ == '__main__':
    main_menu()