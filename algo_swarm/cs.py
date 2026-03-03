import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def sphere(x):
    return np.sum(x**2)

def knapsack(x, w, v, W):
    weight = np.sum(x * w)
    value = np.sum(x * v)
    return -value if weight <= W else -value * 0.1

def levy(beta=1.5):
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma)
    v = np.random.normal(0, 1)
    return u / abs(v) ** (1 / beta)

class CuckooSearch:
    def __init__(self, func, dim, bounds, n_nests=20, n_iter=100, pa=0.25):
        self.func = func
        self.dim = dim
        self.lb, self.ub = bounds
        self.n_nests = n_nests
        self.n_iter = n_iter
        self.pa = pa

        self.nests = np.random.uniform(self.lb, self.ub, (n_nests, dim))
        self.fitness = np.array([func(x) for x in self.nests])

        self.best = None
        self.best_val = np.inf
        self.history_best = []
        self.history_pos = []

    def run(self):
        for _ in range(self.n_iter):
            for i in range(self.n_nests):
                step = levy() * (self.nests[i] - self.nests[np.random.randint(self.n_nests)])
                new = self.nests[i] + step
                new = np.clip(new, self.lb, self.ub)

                f = self.func(new)
                j = np.random.randint(self.n_nests)
                if f < self.fitness[j]:
                    self.nests[j] = new
                    self.fitness[j] = f

            abandon = np.random.rand(self.n_nests) < self.pa
            self.nests[abandon] = np.random.uniform(self.lb, self.ub, (np.sum(abandon), self.dim))
            self.fitness = np.array([self.func(x) for x in self.nests])

            idx = np.argmin(self.fitness)
            if self.fitness[idx] < self.best_val:
                self.best_val = self.fitness[idx]
                self.best = self.nests[idx].copy()

            self.history_best.append(self.best_val)
            self.history_pos.append(self.nests.copy())

class BinaryCuckooSearch:
    def __init__(self, w, v, W, n_nests=20, n_iter=100, pa=0.25):
        self.w, self.v, self.W = w, v, W
        self.n = len(w)
        self.n_nests = n_nests
        self.n_iter = n_iter
        self.pa = pa
        self.history_pos = []
        self.nests = np.random.randint(0, 2, (n_nests, self.n))
        self.fitness = np.array([knapsack(x, w, v, W) for x in self.nests])
        self.best = None
        self.best_val = -np.inf
        self.history_best = []

    def run(self):
        for _ in range(self.n_iter):
            for i in range(self.n_nests):
                step = levy()
                prob = 1 / (1 + np.exp(-step))
                new = np.where(np.random.rand(self.n) < prob, 1 - self.nests[i], self.nests[i])

                f = knapsack(new, self.w, self.v, self.W)
                j = np.random.randint(self.n_nests)
                if f < self.fitness[j]:
                    self.nests[j] = new
                    self.fitness[j] = f
            abandon = np.random.rand(self.n_nests) < self.pa
            self.nests[abandon] = np.random.randint(0, 2, (np.sum(abandon), self.n))
            self.fitness = np.array([knapsack(x, self.w, self.v, self.W) for x in self.nests])

            idx = np.argmax(self.fitness)
            if self.fitness[idx] > self.best_val:
                self.best_val = self.fitness[idx]
                self.best = self.nests[idx].copy()

            self.history_best.append(self.best_val)
            self.history_pos.append(self.nests.copy())

def animate_cs(cs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    def update(i):
        ax1.clear()
        ax2.clear()

        X = cs.history_pos[i]
        ax1.scatter(X[:, 0], X[:, 1], alpha=0.7)
        ax1.scatter(cs.best[0], cs.best[1], marker="*", s=200)
        ax1.set_title(f"Cuckoo Search (Iter {i+1})")
        ax1.grid(alpha=0.3)

        ax2.plot(cs.history_best[:i+1], lw=2.5)
        ax2.set_title("Convergence")
        ax2.grid(alpha=0.3)

    anim = FuncAnimation(fig, update, frames=len(cs.history_best), interval=250)
    plt.tight_layout()
    plt.show()
    
def animate_binary_cs(bcs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def update(frame):
        ax1.clear()
        ax2.clear()
        
        # Plot item frequency at this frame
        nests_at_frame = bcs.history_pos[frame]
        item_frequency = np.sum(nests_at_frame, axis=0)
        items = [f"Item {i+1}" for i in range(bcs.n)]
        ax1.bar(items, item_frequency, color='steelblue', alpha=0.7)
        ax1.set_title(f"Item Selection Frequency (Iteration {frame + 1}/100)")
        ax1.set_ylabel("Selection Count")
        ax1.set_ylim(0, bcs.n_nests)
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot convergence curve to current iteration
        ax2.plot(bcs.history_best[:frame + 1], linewidth=2.5, color='darkred', label='Best Fitness')
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Best Fitness Value")
        ax2.set_title(f"Convergence Curve ({frame + 1}/100)")
        ax2.grid(alpha=0.3)
        ax2.legend()

    anim = FuncAnimation(fig, update, frames=len(bcs.history_best), interval=100, repeat=False)
    plt.tight_layout()
    plt.show()
    plt.close()

def multiple_runs_cs():
    results = []
    for _ in range(20):
        cs = CuckooSearch(sphere, 2, (-5.12, 5.12), pa=0.1)
        cs.run()
        results.append(cs.best_val)

    r = np.array(results)
    print("\n===== CONTINUOUS CS MULTIPLE RUNS (30 runs) =====")
    print(f"Mean:   {r.mean():.6e}")
    print(f"Std:    {r.std():.6e}")
    print(f"Best:   {r.min():.6e}")
    print(f"Worst:  {r.max():.6e}")

def multiple_runs_binary_cs(n_runs=20):
    w = np.array([2, 3, 4, 5, 9])
    v = np.array([3, 4, 5, 8, 10])
    W = 15
    results = []
    for _ in range(n_runs):
        bcs = BinaryCuckooSearch(w, v, W)
        bcs.run()
        results.append(bcs.best_val)

    r = np.array(results)
    print(f"\n===== BINARY CS MULTIPLE RUNS ({n_runs} runs) =====")
    print(f"Mean:   {r.mean():.6e}")
    print(f"Std:    {r.std():.6e}")
    print(f"Best:   {r.min():.6e}")
    print(f"Worst:  {r.max():.6e}")

def plot_cs_report(cs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    X = np.vstack(cs.history_pos)
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.6, label="Nests")
    ax1.scatter(cs.best[0], cs.best[1],
                marker="*", s=250, c="red", label="Best")
    ax1.set_title(f"Best Solution (f = {cs.best_val})")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(cs.history_best, lw=2.5)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best fitness")
    ax2.set_title("Convergence Curve")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_binary_cs_report(bcs, w):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    items = [f"Item {i+1}" for i in range(bcs.n)]
    colors = ['blue' if bcs.best[i] == 1 else 'gray' for i in range(bcs.n)]
    ax1.bar(items, bcs.best * w, color=colors, alpha=0.7)
    ax1.set_title(f"Best Solution (Value = {bcs.best_val:.2f})")
    ax1.set_ylabel("Weight")
    ax1.set_xlabel("Items")
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.plot(bcs.history_best, lw=2.5, color='darkred')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best Fitness Value")
    ax2.set_title("Convergence Curve")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_3d_sphere(cs=None):
    from mpl_toolkits.mplot3d import Axes3D
    
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    fig = plt.figure(figsize=(14, 6))
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_zlabel('f(x)')
    ax3.set_title('3D Surface: Sphere Function')

    if cs is not None:
        # overlay visited points from search history
        pts = np.vstack(cs.history_pos)
        Zpts = np.sum(pts**2, axis=1)
        ax3.scatter(pts[:, 0], pts[:, 1], Zpts,
                    c='red', s=40, alpha=0.6, edgecolors='black',
                    label='Visited')
        ax3.legend()

    cont = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title('2D Contour: Sphere Function')

    if cs is not None:
        ax2.scatter(pts[:, 0], pts[:, 1], c='red', s=20, alpha=0.6,
                    edgecolors='black')
    fig.colorbar(cont, ax=ax2)

    plt.tight_layout()
    plt.show()

def plot_binary_evolution_3d(bcs):
    """Display evolution of binary solutions in 3D"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    iterations = []
    solution_indices = []
    fitness_values = []
    
    for iter_idx in range(len(bcs.history_pos)):
        iter_nests = bcs.history_pos[iter_idx]
        for sol_idx in range(len(iter_nests)):
            fit_val = knapsack(iter_nests[sol_idx], bcs.w, bcs.v, bcs.W)
            iterations.append(iter_idx)
            solution_indices.append(sol_idx)
            fitness_values.append(fit_val)
    
    scatter = ax.scatter(iterations, solution_indices, fitness_values, 
                        c=fitness_values, cmap='viridis', s=50, alpha=0.6)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Nest Index')
    ax.set_zlabel('Fitness Value')
    ax.set_title('Binary CS: Population Evolution')
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Fitness')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("1. Continuous CS")
    print("2. Binary CS")
    print("3. 3D Visualization (Continuous)")
    print("4. 3D Visualization (Binary)")
    choice = input("Choose: ")

    if choice == "1":
        cs = CuckooSearch(sphere, 2, (-5.12, 5.12), pa=0.1)
        cs.run()
        print("\n===== FINAL RESULT =====")
        print("Best value:", cs.best_val)
        print("Best solution:", cs.best)
        animate_cs(cs)
        plot_cs_report(cs)
        cont = input("Chạy multiple test để thống kê (Y/N): ").strip()
        if cont == "Y":
            multiple_runs_cs()
    elif choice == "2":
        w = np.array([2, 3, 4, 5, 9])
        v = np.array([3, 4, 5, 8, 10])
        W = 15
        bcs = BinaryCuckooSearch(w, v, W)
        bcs.run()
        print("\n===== FINAL RESULT =====")
        print("Best solution:", bcs.best)
        print("Best value:", bcs.best_val)
        print("Total weight:", np.sum(bcs.best * w))
        animate_binary_cs(bcs)
        plot_binary_cs_report(bcs, w)
        cont = input("Chạy multiple test để thống kê (Y/N): ").strip()
        if cont == "Y":
            multiple_runs_binary_cs(n_runs=20)
    elif choice == "3":
        cs = CuckooSearch(sphere, 2, (-5, 5))
        cs.run()
        plot_3d_sphere(cs)
    else:
        w = np.array([2, 3, 4, 5, 9])
        v = np.array([3, 4, 5, 8, 10])
        W = 15
        bcs = BinaryCuckooSearch(w, v, W)
        bcs.run()
        plot_binary_evolution_3d(bcs)
        
