import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

DIM = 5
NUM_CITIES = 20
POP_SIZE = 100
GENS = 200
TRIALS = 30
CR = 0.90
ELITE = 2

np.random.seed(42)
TSP_COORDS = np.random.rand(NUM_CITIES, 2) * 100
TSP_DIST_MAT = np.zeros((NUM_CITIES, NUM_CITIES))
for i in range(NUM_CITIES):
    for j in range(NUM_CITIES):
        TSP_DIST_MAT[i][j] = np.linalg.norm(TSP_COORDS[i] - TSP_COORDS[j])

def rastrigin(pop):
    return 10 * DIM + np.sum(pop**2 - 10 * np.cos(2 * np.pi * pop), axis=1)

def calc_tsp_dist(pop):
    dists = []
    for p in pop:
        d = np.sum([TSP_DIST_MAT[p[i], p[i+1]] for i in range(NUM_CITIES-1)]) + TSP_DIST_MAT[p[-1], p[0]]
        dists.append(d)
    return np.array(dists)

def run_ga_rastrigin(mut_rate):
    start_time = time.time()
    bounds = [-5.12, 5.12]
    pop = np.random.uniform(bounds[0], bounds[1], (POP_SIZE, DIM))
    best_val = float('inf')
    history = []

    for _ in range(GENS):
        fitness = rastrigin(pop)
        curr_min = np.min(fitness)
        if curr_min < best_val:
            best_val = curr_min
        history.append(best_val)

        sorted_idx = np.argsort(fitness)
        new_pop = [pop[i].copy() for i in sorted_idx[:ELITE]]

        candidates = np.random.randint(0, POP_SIZE, (POP_SIZE, 3))
        winners = candidates[np.arange(POP_SIZE), np.argmin(fitness[candidates], axis=1)]
        parents = pop[winners]

        while len(new_pop) < POP_SIZE:
            p1, p2 = parents[np.random.randint(0, POP_SIZE, 2)]
            
            if np.random.rand() < CR:
                mask = np.random.rand(DIM) < 0.5
                c1 = np.where(mask, p1, p2)
                c2 = np.where(mask, p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()
                
            for child in [c1, c2]:
                if len(new_pop) < POP_SIZE:
                    mut_mask = np.random.rand(DIM) < mut_rate
                    child[mut_mask] += np.random.normal(0, 0.5, np.sum(mut_mask))
                    child = np.clip(child, bounds[0], bounds[1])
                    new_pop.append(child)

        pop = np.array(new_pop)

    return best_val, time.time() - start_time, history

def ox_crossover(p1, p2):
    if np.random.rand() >= CR:
        return p1.copy(), p2.copy()
    c1, c2 = np.full(NUM_CITIES, -1), np.full(NUM_CITIES, -1)
    cp1, cp2 = sorted(np.random.choice(NUM_CITIES, 2, replace=False))
    
    c1[cp1:cp2+1] = p1[cp1:cp2+1]
    c2[cp1:cp2+1] = p2[cp1:cp2+1]
    
    def fill(child, parent):
        curr = (cp2 + 1) % NUM_CITIES
        for g in np.roll(parent, -(cp2 + 1)):
            if g not in child:
                child[curr] = g
                curr = (curr + 1) % NUM_CITIES
    fill(c1, p2)
    fill(c2, p1)
    return c1, c2

def run_ga_tsp(mut_rate):
    start_time = time.time()
    pop = np.array([np.random.permutation(NUM_CITIES) for _ in range(POP_SIZE)])
    best_dist = float('inf')
    best_path = None
    history = []

    for _ in range(GENS):
        fitness = calc_tsp_dist(pop)
        curr_min_idx = np.argmin(fitness)
        if fitness[curr_min_idx] < best_dist:
            best_dist = fitness[curr_min_idx]
            best_path = pop[curr_min_idx].copy()
        history.append(best_dist)

        sorted_idx = np.argsort(fitness)
        new_pop = [pop[i].copy() for i in sorted_idx[:ELITE]]

        candidates = np.random.randint(0, POP_SIZE, (POP_SIZE, 3))
        winners = candidates[np.arange(POP_SIZE), np.argmin(fitness[candidates], axis=1)]
        parents = pop[winners]

        while len(new_pop) < POP_SIZE:
            p1, p2 = parents[np.random.randint(0, POP_SIZE, 2)]
            c1, c2 = ox_crossover(p1, p2)
            
            for child in [c1, c2]:
                if len(new_pop) < POP_SIZE:
                    if np.random.rand() < mut_rate:
                        i1, i2 = np.random.choice(NUM_CITIES, 2, replace=False)
                        child[i1], child[i2] = child[i2], child[i1]
                    new_pop.append(child)

        pop = np.array(new_pop)

    return best_dist, best_path, time.time() - start_time, history

def plot_ga_3d():
    def rast_2d(x, y):
        return 20 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y)
    
    pop = np.random.uniform(-5.12, 5.12, (POP_SIZE, 2))
    for _ in range(30):
        fit = rast_2d(pop[:,0], pop[:,1])
        winners = np.random.randint(0, POP_SIZE, (POP_SIZE, 3))
        parents = pop[winners[np.arange(POP_SIZE), np.argmin(fit[winners], axis=1)]]
        new_pop = []
        new_pop.append(pop[np.argmin(fit)])
        while len(new_pop) < POP_SIZE:
            p1, p2 = parents[np.random.randint(0, POP_SIZE, 2)]
            mask = np.random.rand(2) < 0.5
            c = np.where(mask, p1, p2)
            if np.random.rand() < 0.1:
                c += np.random.normal(0, 0.2, 2)
            new_pop.append(np.clip(c, -5.12, 5.12))
        pop = np.array(new_pop)

    X = np.linspace(-5.12, 5.12, 100)
    Y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(X, Y)
    Z = rast_2d(X, Y)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6, linewidth=0, antialiased=False)
    ax.scatter(pop[:,0], pop[:,1], rast_2d(pop[:,0], pop[:,1]) + 2, c='red', s=50, edgecolors='black', label='Final Population')
    ax.set_title('3D Visualization: GA Convergence on Rastrigin')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Fitness Value')
    ax.legend()
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=45, azim=45)
    plt.savefig('ga_3d_landscape.png', dpi=300)
    plt.close()

def main():
    print("-" * 90)
    print(f"{'Mutation Rate':<15} | {'Best Fit':<12} | {'Mean Fit':<12} | {'Std Dev':<10} | {'Success(%)':<12} | {'Avg Time(s)':<12}")
    print("-" * 90)

    mut_rates = [0.01, 0.05, 0.20]
    best_overall_history = None
    best_overall_val = float('inf')

    for mr in mut_rates:
        fits, times = [], []
        for _ in range(TRIALS):
            fit, t, hist = run_ga_rastrigin(mr)
            fits.append(fit)
            times.append(t)
            if fit < best_overall_val:
                best_overall_val = fit
                best_overall_history = hist
        
        success_rate = sum(1 for x in fits if x < 1e-2) / TRIALS * 100
        print(f"{mr:<15} | {np.min(fits):<12.5f} | {np.mean(fits):<12.5f} | {np.std(fits):<10.5f} | {success_rate:<12.1f} | {np.mean(times):<12.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(best_overall_history, color='green', linewidth=2, label='GA Convergence')
    plt.yscale('log')
    plt.title('GA Convergence on Rastrigin Function')
    plt.xlabel('Generations')
    plt.ylabel('Best Fitness (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig('ga_ras_convergence.png', dpi=300)
    plt.close()

    print("\n" + "-" * 90)
    print(f"{'Algorithm':<15} | {'Best Dist':<12} | {'Mean Dist':<12} | {'Std Dev':<10} | {'Success(%)':<12} | {'Avg Time(s)':<12}")
    print("-" * 90)

    ga_results, ga_times = [], []
    best_tsp_dist = float('inf')
    best_tsp_path = None
    best_tsp_history = None

    for _ in range(TRIALS):
        dist, path, t, hist = run_ga_tsp(0.05)
        ga_results.append(dist)
        ga_times.append(t)
        if dist < best_tsp_dist:
            best_tsp_dist = dist
            best_tsp_path = path
            best_tsp_history = hist

    success_rate = sum(1 for x in ga_results if x < 400.0) / TRIALS * 100
    print(f"{'Genetic Algo':<15} | {np.min(ga_results):<12.2f} | {np.mean(ga_results):<12.2f} | {np.std(ga_results):<10.2f} | {success_rate:<12.1f} | {np.mean(ga_times):<12.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(best_tsp_history, color='blue', linewidth=2, label='GA Convergence')
    plt.title('GA Convergence on TSP')
    plt.xlabel('Generations')
    plt.ylabel('Total Distance')
    plt.legend()
    plt.grid(True)
    plt.savefig('ga_tsp_convergence.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.boxplot(ga_results, patch_artist=True, boxprops=dict(facecolor="lightblue", color="blue"), medianprops=dict(color="red", linewidth=2))
    plt.title(f'Stability Analysis over {TRIALS} runs (GA TSP)')
    plt.xlabel('Algorithm')
    plt.ylabel('Best Distance Found')
    plt.xticks([1], ['Genetic Algorithm'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.plot(np.random.normal(1, 0.04, size=len(ga_results)), ga_results, 'r.', alpha=0.5, label='Trial Results')
    plt.legend()
    plt.savefig('ga_tsp_boxplot.png', dpi=300)
    plt.close()

    ordered_coords = np.vstack([TSP_COORDS[best_tsp_path], TSP_COORDS[best_tsp_path][0]])
    plt.figure(figsize=(8, 5))
    plt.plot(ordered_coords[:, 0], ordered_coords[:, 1], 'o-', color='red', linewidth=2, label='Optimized Route')
    for i, txt in enumerate(best_tsp_path):
        plt.annotate(str(txt), (TSP_COORDS[txt, 0], TSP_COORDS[txt, 1]), fontsize=10, weight='bold', xytext=(5, 5), textcoords='offset points')
    plt.title(f'GA Optimized Route (Distance: {best_tsp_dist:.2f})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.savefig('ga_tsp_route.png', dpi=300)
    plt.close()

    plot_ga_3d()

if __name__ == "__main__":
    main()