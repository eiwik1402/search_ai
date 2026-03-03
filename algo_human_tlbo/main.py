import sys
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)

try:
    from algorithms.tlbo import TLBO
    from algorithms.classic import HillClimbing, SimulatedAnnealing 
    from objectives.objective import PROBLEMS
    from visualization.visualization import *
except ImportError as e:
    print(f"ERROR Import failed: {e}"); sys.exit(1)

def generate_tlbo_statistical_summary(problem):
    """Tạo bảng thống kê chi tiết cho TLBO và xuất ra CSV"""
    dims = [10, 20, 30, 40, 50, 60]
    summary_data = []
    
    for d in dims:
        problem.dim = d
        times = []
        fits = []
        for seed in range(15):
            np.random.seed(seed)
            opt = TLBO(problem, pop_size=30, max_iter=50)
            out = opt.run()
            times.append(out['runtime'])
            fits.append(out['best_fitness'])
            
        summary_data.append({
            'Dimension': d,
            'Time_sec_mean': np.mean(times),
            'Time_sec_median': np.median(times),
            'Time_sec_std': np.std(times),
            'Time_sec_min': np.min(times),
            'Time_sec_max': np.max(times),
            'Fitness_mean': np.mean(fits),
            'Fitness_median': np.median(fits),
            'Fitness_std': np.std(fits),
            'Fitness_min': np.min(fits),
            'Fitness_max': np.max(fits)
        })
    problem.dim = 30 
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv("results/tlbo_summary.csv", index=False)

def main():
    for folder in ["results", "logs", "figures"]:
        path = os.path.join(BASE_DIR, folder)
        os.makedirs(path, exist_ok=True)

    PROBLEM_NAME = "Sphere"
    problem = PROBLEMS[PROBLEM_NAME]
    N_RUNS = 20
    print(f"\nRunning Experiments...")
    
    algos = {"TLBO": TLBO, "SimulatedAnnealing": SimulatedAnnealing, "HillClimbing": HillClimbing}
    tlbo_histories = []
    tlbo_final_fits = []
    avg_convergence = {}
    df_rows = []
    
    for name, algo_cls in algos.items():
        conv_list = []
        
        for seed in range(N_RUNS):
            np.random.seed(seed)
            if name == "TLBO": opt = algo_cls(problem, pop_size=30, max_iter=100)
            else: opt = algo_cls(problem, max_iter=100)
            
            out = opt.run()
            conv_list.append(out['convergence'])
            df_rows.append({"Algorithm": name, "Run": seed+1, "Best_Fitness": out['best_fitness']})
            
            if name == "TLBO":
                tlbo_histories.append(out['convergence'])
                tlbo_final_fits.append(out['best_fitness'])
        
        min_len = min(len(c) for c in conv_list)
        avg_convergence[name] = np.mean([c[:min_len] for c in conv_list], axis=0)

    df_results = pd.DataFrame(df_rows)
    df_conv_compare = pd.DataFrame(avg_convergence)

    print(f"\nExporting TLBO Specific Data to CSV...")

    #TLBO Summary
    generate_tlbo_statistical_summary(problem)

    #TLBO Convergence Data
    min_len = min(len(c) for c in tlbo_histories)
    tlbo_conv_matrix = np.array([c[:min_len] for c in tlbo_histories])
    df_tlbo_conv = pd.DataFrame({
        'Iteration': np.arange(1, min_len + 1),
        'Mean_Fitness': np.mean(tlbo_conv_matrix, axis=0),
        'Std_Dev': np.std(tlbo_conv_matrix, axis=0)
    })
    df_tlbo_conv.to_csv("results/tlbo_convergence_data.csv", index=False)

    #TLBO Sensitivity Data
    pop_sizes = [10, 20, 30, 50, 80, 100]
    sens_means, sens_stds = [], []
    for p in pop_sizes:
        fits = []
        for _ in range(5): 
            opt = TLBO(problem, pop_size=p, max_iter=60)
            fits.append(opt.run()['best_fitness'])
        sens_means.append(np.mean(fits))
        sens_stds.append(np.std(fits))
        
    df_tlbo_sens = pd.DataFrame({'Population_Size': pop_sizes, 'Mean_Fitness': sens_means, 'Std_Dev': sens_stds})
    df_tlbo_sens.to_csv("results/tlbo_sensitivity_data.csv", index=False)

    #TLBO Scalability Data
    dims = [10, 50, 100, 200, 300]
    scale_times = []
    for d in dims:
        problem.dim = d
        times = []
        for _ in range(3): 
            opt = TLBO(problem, pop_size=30, max_iter=50)
            times.append(opt.run()['runtime'])
        scale_times.append(np.mean(times))
    problem.dim = 30 # Trả lại cũ
    
    df_tlbo_scale = pd.DataFrame({'Dimension': dims, 'Runtime_Sec': scale_times})
    df_tlbo_scale.to_csv("results/tlbo_scalability_data.csv", index=False)

    print(f"\nGenerating Charts")
    
    try:
        # TLBO Charts
        plot_tlbo_convergence(tlbo_histories)
        plot_tlbo_distribution(tlbo_final_fits)
        plot_tlbo_sensitivity("Population Size", pop_sizes, sens_means, sens_stds)
        plot_tlbo_scalability(dims, scale_times)
        
        original_dim = problem.dim
        problem.dim = 2
        plot_3d_landscape(problem.func, PROBLEM_NAME, lb=-10, ub=10)
        problem.dim = original_dim
        
        # Compare Charts
        plot_cont_convergence(avg_convergence)
        plot_cont_robustness_bar(df_results)
        plot_cont_stability_std(df_results)
        
        tlbo_csv_path = os.path.join(BASE_DIR, "results", "tlbo_summary.csv")
        bfs_csv_path = os.path.join(BASE_DIR, "classic", "bfs_statistical_summary.csv") 
        dfs_csv_path = os.path.join(BASE_DIR, "classic", "dfs_statistical_summary.csv") 
        
        POP_SIZE = 30 
        
        plot_empirical_time_complexity(tlbo_csv_path, bfs_csv_path, dfs_csv_path)
        plot_empirical_space_complexity(POP_SIZE, tlbo_csv_path, bfs_csv_path, dfs_csv_path)
        
    except Exception as e:
        print(f"\nERROR Plotting failed: {e}")

    print("\nFinish!")

if __name__ == "__main__":
    main()