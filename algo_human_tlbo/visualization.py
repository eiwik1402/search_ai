import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")
plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
sns.set_style("whitegrid")


def plot_tlbo_convergence(convergence_list):
    plt.figure(figsize=(9, 6))
    min_len = min(len(c) for c in convergence_list)
    data = np.array([c[:min_len] for c in convergence_list])
    mean_line = np.mean(data, axis=0)
    std_line = np.std(data, axis=0)
    iter_range = range(1, len(mean_line) + 1)
    plt.plot(iter_range, mean_line, color='#d62728', linewidth=2.5, label='TLBO Mean')
    plt.fill_between(iter_range, mean_line - std_line, mean_line + std_line, color='#d62728', alpha=0.15, label='Std Dev Range')
    plt.title("TLBO 1: Convergence Ability")
    plt.xlabel("Iteration"); plt.ylabel("Best Fitness (Log)"); plt.yscale("log")
    plt.legend(); plt.tight_layout()
    plt.savefig("figures/TLBO_1_Convergence.png", dpi=300); plt.close()

def plot_tlbo_distribution(fitness_values):
    plt.figure(figsize=(7, 6))
    sns.boxplot(y=fitness_values, color='#ff9999', width=0.4)
    sns.stripplot(y=fitness_values, color='#800000', size=6, alpha=0.6)
    plt.title("TLBO 2: Internal Performance Distribution")
    plt.ylabel("Best Fitness (Log)"); plt.yscale("log"); plt.xlabel("TLBO Runs")
    plt.tight_layout(); plt.savefig("figures/TLBO_2_Distribution.png", dpi=300); plt.close()

def plot_tlbo_sensitivity(param_name, param_values, means, stds):
    plt.figure(figsize=(9, 6))
    plt.errorbar(param_values, means, yerr=stds, fmt='o-', color='#d62728', ecolor='gray', capsize=5, linewidth=2)
    plt.title(f"TLBO 3: Sensitivity Analysis ({param_name})")
    plt.xlabel(f"{param_name}"); plt.ylabel("Mean Best Fitness"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("figures/TLBO_3_Sensitivity.png", dpi=300); plt.close()

def plot_3d_landscape(problem_func, problem_name, lb=-10, ub=10):
    try:
        x = np.linspace(lb, ub, 100); y = np.linspace(lb, ub, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]): Z[i, j] = problem_func(np.array([X[i, j], Y[i, j]]))
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.85)
        ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap='viridis', alpha=0.3)
        ax.set_title(f"TLBO 4: 3D Landscape ({problem_name})")
        ax.set_xlabel('X1'); ax.set_ylabel('X2'); ax.set_zlabel('Fitness'); ax.view_init(30, 45)
        plt.savefig("figures/TLBO_4_3D_Surface.png", dpi=300); plt.close()
    except: pass

def plot_tlbo_scalability(dims, runtimes):
    plt.figure(figsize=(9, 6))
    plt.plot(dims, runtimes, 's-', color='#d62728', linewidth=2.5, markersize=8)
    z = np.polyfit(dims, runtimes, 1)
    p = np.poly1d(z)
    plt.plot(dims, p(dims), "k--", alpha=0.6, label=f"Linear Trend (Complexity O(D))")
    plt.title("TLBO 5: Scalability & Computational Cost")
    plt.xlabel("Problem Dimension"); plt.ylabel("Execution Time (s)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("figures/TLBO_5_Scalability_Cost.png", dpi=300); plt.close()

def plot_cont_convergence(algos_data):
    plt.figure(figsize=(10, 6))
    colors = {'TLBO': '#d62728', 'SimulatedAnnealing': '#2ca02c', 'HillClimbing': '#1f77b4'}
    for name, data in algos_data.items():
        plt.plot(data, label=name, linewidth=2.5, color=colors.get(name, 'gray'))
    plt.title("CONT 1: Convergence Speed Comparison")
    plt.xlabel("Iteration"); plt.ylabel("Best Fitness (Log)"); plt.yscale("log")
    plt.legend(); plt.tight_layout(); plt.savefig("figures/CONT_1_Convergence_Speed.png", dpi=300); plt.close()

def plot_cont_robustness_bar(df_results):
    plt.figure(figsize=(9, 6))
    order = ["TLBO", "SimulatedAnnealing", "HillClimbing"]
    sns.barplot(x="Algorithm", y="Best_Fitness", hue="Algorithm", data=df_results, order=order, palette="viridis", errorbar='sd', capsize=.1, legend=False)
    plt.title("CONT 2: Quality & Robustness (Mean ± Std Dev)")
    plt.ylabel("Mean Best Fitness"); plt.tight_layout(); plt.savefig("figures/CONT_2_Robustness_Bar.png", dpi=300); plt.close()

def plot_cont_stability_std(df_results):
    plt.figure(figsize=(8, 6))
    std_data = df_results.groupby("Algorithm")["Best_Fitness"].std().reset_index()
    std_data.columns = ["Algorithm", "StdDev"]
    sns.barplot(x="Algorithm", y="StdDev", hue="Algorithm", data=std_data, order=["TLBO", "SimulatedAnnealing", "HillClimbing"], palette="magma", legend=False)
    plt.title("CONT 4: Stability Metric (Std Dev)")
    plt.ylabel("Standard Deviation"); plt.tight_layout(); plt.savefig("figures/CONT_4_Stability_StdDev.png", dpi=300); plt.close()

def plot_empirical_time_complexity(tlbo_csv, bfs_csv, dfs_csv):
    plt.figure(figsize=(10, 6))
    
    if os.path.exists(tlbo_csv):
        df_tlbo = pd.read_csv(tlbo_csv)
        plt.plot(df_tlbo['Dimension'], df_tlbo['Time_sec_mean'], 's-', color='#d62728', linewidth=2.5, label='TLBO (Continuous Domain)')
        
    if os.path.exists(bfs_csv):
        df_bfs = pd.read_csv(bfs_csv)
        plt.plot(df_bfs['Grid_Size'], df_bfs['Time_sec_mean'], 'o-', color='#3498db', linewidth=2, label='BFS (Discrete Graph)')
        
    if os.path.exists(dfs_csv):
        df_dfs = pd.read_csv(dfs_csv)
        plt.plot(df_dfs['Grid_Size'], df_dfs['Time_sec_mean'], '^-', color='#2ecc71', linewidth=2, label='DFS (Discrete Graph)')

    plt.title("Empirical Scalability: Execution Time vs Problem Size")
    plt.xlabel("Problem Size N (Dimension for TLBO / Grid Size for BFS, DFS)")
    plt.ylabel("Execution Time (seconds) - Log Scale")
    plt.yscale("log") 
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/DISC_1_Empirical_Time_Complexity.png", dpi=300)
    plt.close()

def plot_empirical_space_complexity(tlbo_pop_size, tlbo_csv, bfs_csv, dfs_csv):
    plt.figure(figsize=(10, 6))
    
    if os.path.exists(bfs_csv):
        df_bfs = pd.read_csv(bfs_csv)
        plt.plot(df_bfs['Grid_Size'], df_bfs['Nodes_Expanded_mean'], 'o-', color='#3498db', linewidth=2, label='BFS (Nodes in Queue)')

    if os.path.exists(dfs_csv):
        df_dfs = pd.read_csv(dfs_csv)
        plt.plot(df_dfs['Grid_Size'], df_dfs['Nodes_Expanded_mean'], '^-', color='#2ecc71', linewidth=2, label='DFS (Nodes in Stack)')
        
    if os.path.exists(tlbo_csv):
        df_tlbo = pd.read_csv(tlbo_csv)
        plt.plot(df_tlbo['Dimension'], [tlbo_pop_size] * len(df_tlbo), 's--', color='#d62728', linewidth=2.5, label=f'TLBO (Fixed Population = {tlbo_pop_size})')

    plt.title("Empirical Scalability: Memory Usage vs Problem Size")
    plt.xlabel("Problem Size N (Dimension / Grid Size)")
    plt.ylabel("Items Stored in Memory - Log Scale")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/DISC_2_Empirical_Space_Complexity.png", dpi=300)
    plt.close()