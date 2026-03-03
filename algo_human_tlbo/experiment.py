import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from objectives.objective import PROBLEMS
from algorithms.tlbo import TLBO

N_RUNS = 30
MAX_ITER = 100

def run_robustness_experiment(algorithm_class, algorithm_name, problem_name, verbose=True):
    problem = PROBLEMS[problem_name]
    results = []
    all_convergence = []
    all_diversity = [] 
    
    
    for seed in range(N_RUNS):
        np.random.seed(seed)
        
        optimizer = algorithm_class(problem, pop_size=50, max_iter=MAX_ITER)
        out = optimizer.run()
               
        results.append({
            "Run": seed + 1,
            "Best_Fitness": out["best_fitness"],
            "Runtime": out["runtime"]
        })
        all_convergence.append(out["convergence"])
        if algorithm_name == "TLBO":
            all_diversity.append(out["diversity"])

    df = pd.DataFrame(results)
    df.to_csv(f"results/{algorithm_name}_{problem_name}_results.csv", index=False)
    np.save(f"logs/{algorithm_name}_{problem_name}_convergence.npy", np.array(all_convergence))
    if algorithm_name == "TLBO":
        np.save(f"logs/{algorithm_name}_{problem_name}_diversity.npy", np.array(all_diversity))
    

def run_scalability_test(problem_name, verbose=True):
    dims = [10, 30, 50]
    scalability_data = []
    
    for d in dims:
        base_prob = PROBLEMS[problem_name]
        base_prob.dim = d 
        
        runtimes = []
        fitnesses = []
        
        for seed in range(5):
            np.random.seed(seed)
            optimizer = TLBO(base_prob, pop_size=50, max_iter=100)
            out = optimizer.run()
            runtimes.append(out["runtime"])
            fitnesses.append(out["best_fitness"])
            
        mean_time = np.mean(runtimes)
        mean_fit = np.mean(fitnesses)
                
        scalability_data.append({
            "Dimension": d,
            "Mean_Fitness": mean_fit,
            "Avg_Runtime": mean_time
        })
        
    df = pd.DataFrame(scalability_data)
    df.to_csv(f"results/TLBO_{problem_name}_scalability.csv", index=False)
    
