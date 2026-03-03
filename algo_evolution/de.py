import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

NP = 50
DIM = 5
GENS_RAS = 200
GENS_TSP = 200
TRIALS = 30
NUM_CITIES = 20
CR = 0.9

np.random.seed(42)
TSP_COORDS = np.random.rand(NUM_CITIES, 2) * 100
TSP_DIST_MAT = np.zeros((NUM_CITIES, NUM_CITIES))
for i in range(NUM_CITIES):
    for j in range(NUM_CITIES):
        TSP_DIST_MAT[i][j] = np.linalg.norm(TSP_COORDS[i] - TSP_COORDS[j])

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def calc_tsp_dist(perm):
    d = np.sum([TSP_DIST_MAT[perm[i], perm[i+1]] for i in range(len(perm)-1)])
    d += TSP_DIST_MAT[perm[-1], perm[0]]
    return d

def run_de_rastrigin(F):
    bounds = [-5.12, 5.12]
    pop = np.random.uniform(bounds[0], bounds[1], (NP, DIM))
    fitness = 10 * DIM + np.sum(pop**2 - 10 * np.cos(2 * np.pi * pop), axis=1)
    
    start_time = time.time()
    history = [np.min(fitness)]
    
    candidates = np.array([np.delete(np.arange(NP), i) for i in range(NP)])
    row_idx = np.arange(NP)[:, None]
    
    for gen in range(GENS_RAS):
        rand_cols = np.random.rand(NP, NP-1).argsort(axis=1)[:, :3]
        chosen_idxs = candidates[row_idx, rand_cols]
        
        a, b, c = pop[chosen_idxs[:, 0]], pop[chosen_idxs[:, 1]], pop[chosen_idxs[:, 2]]
        mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])
        
        cross_points = np.random.rand(NP, DIM) < CR
        no_cross = ~np.any(cross_points, axis=1)
        if np.any(no_cross):
            force_cross = np.random.randint(0, DIM, size=np.sum(no_cross))
            cross_points[no_cross, force_cross] = True
            
        trial = np.where(cross_points, mutant, pop)
        f_trial = 10 * DIM + np.sum(trial**2 - 10 * np.cos(2 * np.pi * trial), axis=1)
        
        better = f_trial < fitness
        pop[better] = trial[better]
        fitness[better] = f_trial[better]
        history.append(np.min(fitness))
        
    return np.min(fitness), time.time() - start_time, history

def run_de_tsp(F):
    pop = np.random.rand(NP, NUM_CITIES)
    fitness = np.array([calc_tsp_dist(np.argsort(ind)) for ind in pop])
    best_idx = np.argmin(fitness)
    best_val = fitness[best_idx]
    best_vector = pop[best_idx].copy()
    
    for gen in range(GENS_TSP):
        new_pop = np.copy(pop)
        for i in range(NP):
            idxs = [idx for idx in range(NP) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + F * (b - c)
            
            cross_points = np.random.rand(NUM_CITIES) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, NUM_CITIES)] = True
                
            trial = np.where(cross_points, mutant, pop[i])
            f_trial = calc_tsp_dist(np.argsort(trial))
            
            if f_trial < fitness[i]:
                new_pop[i] = trial
                fitness[i] = f_trial
        pop = new_pop
        current_best = np.min(fitness)
        if current_best < best_val:
            best_val = current_best
            best_vector = pop[np.argmin(fitness)].copy()
            
    return best_val, np.argsort(best_vector)

def plot_rastrigin_3d():
    def rastrigin_2d(x, y):
        return 20 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y)
    
    x = np.linspace(-5.12, 5.12, 100)
    y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin_2d(X, Y)
    
    np.random.seed(42)
    de_pop_x = np.random.normal(0, 0.2, NP) 
    de_pop_y = np.random.normal(0, 0.2, NP)
    de_pop_z = rastrigin_2d(de_pop_x, de_pop_y) + 2

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7, linewidth=0, antialiased=False)
    ax.scatter(de_pop_x, de_pop_y, de_pop_z, c='red', s=30, label='Final Population')
    ax.set_title('3D Visualization: DE Convergence on Rastrigin')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Fitness Value')
    ax.legend()
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=45, azim=45)
    plt.savefig('de_3d_landscape.png', dpi=300)

def main():
    print("-" * 90)
    print(f"{'Scaling Factor (F)':<20} | {'Best Fitness':<15} | {'Mean Fitness':<15} | {'Std Dev':<15} | {'Avg Time (s)':<15}")
    print("-" * 90)
    
    f_values = [0.5, 0.7, 0.9]
    best_overall_history = None
    best_overall_val = float('inf')
    
    for f in f_values:
        fits, times = [], []
        for _ in range(TRIALS):
            fit, t, hist = run_de_rastrigin(F=f)
            fits.append(fit)
            times.append(t)
            if fit < best_overall_val:
                best_overall_val = fit
                best_overall_history = hist
        print(f"{f:<20} | {np.min(fits):<15.5f} | {np.mean(fits):<15.5f} | {np.std(fits):<15.5f} | {np.mean(times):<15.5f}")

    plt.figure(figsize=(8, 5))
    plt.plot(best_overall_history, color='blue', linewidth=2, label='DE Convergence')
    plt.yscale('log')
    plt.title('DE Convergence on Rastrigin Function')
    plt.xlabel('Generations')
    plt.ylabel('Best Fitness (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig('de_ras_convergence.png', dpi=300)

    print("\n" + "-" * 90)
    print(f"{'Algorithm':<25} | {'Best Dist':<12} | {'Mean Dist':<12} | {'Std Dev':<12} | {'Success Rate':<12}")
    print("-" * 90)
    
    de_results = []
    best_tsp_dist = float('inf')
    best_tsp_perm = None
    
    for _ in range(TRIALS):
        dist, perm = run_de_tsp(F=0.7)
        de_results.append(dist)
        if dist < best_tsp_dist:
            best_tsp_dist = dist
            best_tsp_perm = perm

    threshold = 400
    de_success = sum(1 for x in de_results if x < threshold) / TRIALS * 100
    
    print(f"{'Differential Evolution':<25} | {np.min(de_results):<12.2f} | {np.mean(de_results):<12.2f} | {np.std(de_results):<12.2f} | {de_success:>10.0f}%")

    ordered_coords = np.vstack([TSP_COORDS[best_tsp_perm], TSP_COORDS[best_tsp_perm][0]])
    plt.figure(figsize=(8, 5))
    plt.plot(ordered_coords[:, 0], ordered_coords[:, 1], 'o-', color='purple', linewidth=2, label='Optimized Route')
    for i, txt in enumerate(best_tsp_perm):
        plt.annotate(str(txt), (TSP_COORDS[txt, 0], TSP_COORDS[txt, 1]), fontsize=10, weight='bold', xytext=(5, 5), textcoords='offset points')
    plt.title(f'DE Optimized Route (Distance: {best_tsp_dist:.2f})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.savefig('de_tsp_route.png', dpi=300)
    
    plot_rastrigin_3d()

if __name__ == "__main__":
    main()