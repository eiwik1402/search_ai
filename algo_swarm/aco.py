import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_cluster_cities(n_clusters=5, cities_per_cluster=6, seed=42):
    np.random.seed(seed)
    coords = []
    for i in range(n_clusters):
        angle = 2 * np.pi * i / n_clusters
        center = np.array([5 * np.cos(angle), 5 * np.sin(angle)])
        cluster = np.random.randn(cities_per_cluster, 2) * 0.4 + center
        coords.append(cluster)
    return np.vstack(coords)

def sphere(x):
    return np.sum(x ** 2)

class ACOR:
    def __init__(self, func, dim, bounds,
                 n_ants=30, n_iter=100,
                 q=0.5, xi=0.85):

        self.func = func
        self.dim = dim
        self.lb, self.ub = bounds
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.q = q
        self.xi = xi

        self.archive = np.random.uniform(
            self.lb, self.ub, (n_ants, dim)
        )
        self.fitness = np.array([self.func(x) for x in self.archive])

        self.best = None
        self.best_val = np.inf

        self.history_best = []
        self.history_pos = []

    def run(self):
        for _ in range(self.n_iter):
            idx = np.argsort(self.fitness)
            self.archive = self.archive[idx]
            self.fitness = self.fitness[idx]
            if self.fitness[0] < self.best_val:
                self.best_val = self.fitness[0]
                self.best = self.archive[0].copy()

            weights = np.exp(-np.arange(self.n_ants) ** 2 /
                             (2 * (self.q * self.n_ants) ** 2))
            weights /= np.sum(weights)
            sigma = np.zeros((self.n_ants, self.dim))
            for i in range(self.n_ants):
                sigma[i] = self.xi * np.mean(
                    np.abs(self.archive[i] - self.archive), axis=0
                )
            new_solutions = []
            for _ in range(self.n_ants):
                k = np.random.choice(self.n_ants, p=weights)
                x = np.random.normal(self.archive[k], sigma[k])
                x = np.clip(x, self.lb, self.ub)
                new_solutions.append(x)

            new_solutions = np.array(new_solutions)
            new_fitness = np.array([self.func(x) for x in new_solutions])
            self.archive = np.vstack((self.archive, new_solutions))
            self.fitness = np.hstack((self.fitness, new_fitness))

            idx = np.argsort(self.fitness)[:self.n_ants]
            self.archive = self.archive[idx]
            self.fitness = self.fitness[idx]
            self.history_best.append(self.fitness[0])
            self.history_pos.append(self.archive.copy())

def animate_acor(acor):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    def update(frame):
        ax1.clear()
        ax2.clear()

        X = acor.history_pos[frame]
        ax1.scatter(X[:, 0], X[:, 1], alpha=0.7)
        ax1.scatter(acor.best[0], acor.best[1], marker='*', s=200)
        ax1.set_title(f"Solutions (Iter {frame + 1})")
        ax1.grid(alpha=0.3)
        ax2.plot(acor.history_best[:frame + 1], linewidth=2.5)
        ax2.set_title("Convergence Curve")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Best f(x)")
        ax2.grid(alpha=0.3)

    anim = FuncAnimation(
        fig, update,
        frames=len(acor.history_best),
        interval=250,
        repeat=False
    )
    plt.tight_layout()
    plt.show()

class AntColony:
    def __init__(self, dist, n_ants=30, n_iter=100,
                 alpha=1.0, beta=4.5, rho=0.5, Q=100,):
        self.initial_length = None
        self.dist = dist
        self.n = dist.shape[0]
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.tau = np.ones((self.n, self.n))
        self.eta = 1 / (dist + 1e-10)
        self.best_path = None
        self.best_length = np.inf
        self.history_best = []
        self.history_path = []
        self.history_tau = []

    def _select_next(self, current, visited):
        prob = np.zeros(self.n)
        for j in range(self.n):
            if j not in visited:
                prob[j] = (self.tau[current, j] ** self.alpha) * \
                          (self.eta[current, j] ** self.beta)
        prob /= prob.sum()
        return np.random.choice(self.n, p=prob)

    def _path_length(self, path):
        L = sum(self.dist[path[i], path[i + 1]] for i in range(self.n - 1))
        return L + self.dist[path[-1], path[0]]

    def run(self):
        for it in range(self.n_iter):
            paths, lengths = [], []

            for _ in range(self.n_ants):
                start = np.random.randint(self.n)
                path = [start]
                visited = set(path)

                while len(path) < self.n:
                    nxt = self._select_next(path[-1], visited)
                    path.append(nxt)
                    visited.add(nxt)

                L = self._path_length(path)
                paths.append(path)
                lengths.append(L)

                if L < self.best_length:
                    self.best_length = L
                    self.best_path = path.copy()
            if it == 0:
                self.initial_length = self.best_length
            self.tau *= (1 - self.rho)
            for path, L in zip(paths, lengths):
                for i in range(self.n - 1):
                    self.tau[path[i], path[i + 1]] += self.Q / L
                self.tau[path[-1], path[0]] += self.Q / L

            self.history_best.append(self.best_length)
            self.history_path.append(self.best_path.copy())
            self.history_tau.append(self.tau.copy())

def animate_aco(coords, aco):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax1, ax2, ax3 = axes
    x, y = coords[:, 0], coords[:, 1]

    def update(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear()

        path = aco.history_path[frame]
        for i in range(len(path)):
            j = (i + 1) % len(path)
            ax1.plot([x[path[i]], x[path[j]]],
                     [y[path[i]], y[path[j]]], linewidth=2)

        ax1.scatter(x, y, s=120, alpha=0.8)
        ax1.set_aspect("equal")
        ax1.set_title(f"Best Tour (Iter {frame + 1})")
        ax1.grid(alpha=0.3)

        ax2.plot(aco.history_best[:frame + 1], linewidth=2.5)
        ax2.set_title("Convergence Curve")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Best length")
        ax2.grid(alpha=0.3)

        im = ax3.imshow(aco.history_tau[frame], cmap="YlOrBr", aspect="auto")
        ax3.set_title("Pheromone Matrix")
        ax3.set_xlabel("City")
        ax3.set_ylabel("City")

    anim = FuncAnimation(
        fig, update,
        frames=len(aco.history_best),
        interval=250,
        repeat=False
    )
    plt.tight_layout()
    plt.show()

def plot_final_report(coords, aco):
    x, y = coords[:, 0], coords[:, 1]
    path = aco.best_path
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
    for i in range(len(path)):
        j = (i + 1) % len(path)
        ax1.plot([x[path[i]], x[path[j]]],
                 [y[path[i]], y[path[j]]], linewidth=2)

    ax1.scatter(x, y, s=120)
    ax1.set_aspect("equal")
    ax1.set_title(f"Best Tour (Length = {aco.best_length:.2f})")
    ax1.grid(alpha=0.3)

    ax2.plot(aco.history_best, linewidth=2.5)
    ax2.set_title("Convergence Curve")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best tour length")
    ax2.grid(alpha=0.3)

    im = ax3.imshow(aco.tau, cmap="YlOrBr", aspect="auto")
    ax3.set_title("Final Pheromone Matrix")
    ax3.set_xlabel("City")
    ax3.set_ylabel("City")
    plt.colorbar(im, ax=ax3, fraction=0.046)

    plt.tight_layout()
    plt.show()
    
def report_acor(acor):
    print("\n===== ACOR FINAL REPORT =====")
    print(f"Best fitness value : {acor.best_val:.6f}")
    print(f"Best solution      : {acor.best}")
    
def plot_acor_convergence(acor):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    X = np.vstack(acor.history_pos)
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.7)
    ax1.scatter(acor.best[0], acor.best[1],
                marker='*', s=200, label='Best')
    ax1.set_title("Final Solutions Distribution")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.legend()
    ax1.grid(alpha=0.2)
    ax2.plot(acor.history_best, linewidth=2.0)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best f(x)")
    ax2.set_title("ACOR Convergence Curve")
    ax2.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

def multiple_runs(dist, n_runs=20):
    results = []
    for _ in range(n_runs):
        aco = AntColony(dist)
        aco.run()
        results.append(aco.best_length)

    results = np.array(results)
    print("\n===== MULTIPLE RUNS STATISTICS (ACO) =====")
    print(f"Runs : {n_runs}")
    print(f"Mean : {results.mean():.4f}")
    print(f"Std  : {results.std():.4f}")
    print(f"Best : {results.min():.4f}")
    print(f"Worst: {results.max():.4f}")

def multiple_runs_acor(n_runs=20):
    best_vals = []
    for _ in range(n_runs):
        acor = ACOR(
            func=sphere,
            dim=2,
            bounds=(-5, 5),
            n_ants=30,
            n_iter=100
        )
        acor.run()
        best_vals.append(acor.best_val)
    best_vals = np.array(best_vals)
    print("\n===== MULTIPLE RUNS STATISTICS (ACOR) =====")
    print(f"Runs : {n_runs}")
    print(f"Mean : {best_vals.mean()}")
    print(f"Std  : {best_vals.std()}")
    print(f"Best : {best_vals.min()}")
    print(f"Worst: {best_vals.max()}")


def plot_search_dynamics_acor(acor):
    """Hiển thị không gian tìm kiếm + phân bố solutions với contour cạnh bên"""
    from mpl_toolkits.mplot3d import Axes3D
    
    x = np.linspace(acor.lb, acor.ub, 40)
    y = np.linspace(acor.lb, acor.ub, 40)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    fig = plt.figure(figsize=(14, 7))
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3, edgecolor='none')
    all_solutions = np.vstack(acor.history_pos)
    Z_solutions = np.sum(all_solutions**2, axis=1)
    ax3.scatter(all_solutions[:, 0], all_solutions[:, 1], Z_solutions,
               c='red', marker='o', s=50, alpha=0.5, label='All Ants')
    Z_best = acor.best_val
    ax3.scatter(acor.best[0], acor.best[1], Z_best,
               c='gold', marker='*', s=800, edgecolors='black', linewidths=2,
               label='Best solution', zorder=10)
    ax3.set_xlabel('x₁')
    ax3.set_ylabel('x₂')
    ax3.set_zlabel('f(x)')
    ax3.set_title('3D Surface + Search Dynamics (ACOR)')
    ax3.legend()

    cont = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax2.scatter(all_solutions[:, 0], all_solutions[:, 1], c='red', s=20, alpha=0.5)
    ax2.scatter(acor.best[0], acor.best[1], c='gold', s=100, edgecolors='black')
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_title('Contour with Ant Locations')
    fig.colorbar(cont, ax=ax2)

    plt.tight_layout()
    plt.show()

def plot_tsp_evolution_3d(aco):
    """Hiển thị evolution của TSP solutions dưới dạng 3D"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    iterations = []
    ant_indices = []
    tour_lengths = []
    
    for iter_idx in range(len(aco.history_best)):
        best_length = aco.history_best[iter_idx]
        for ant_idx in range(min(aco.n_ants, 10)):  # Lấy max 10 ant
            iterations.append(iter_idx)
            ant_indices.append(ant_idx)
            tour_lengths.append(best_length + np.random.randn() * 0.1)  # Thêm noise để tránh overlap
    
    scatter = ax.scatter(iterations, ant_indices, tour_lengths, 
                        c=tour_lengths, cmap='viridis', s=50, alpha=0.6)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Ant Index')
    ax.set_zlabel('Tour Length')
    ax.set_title('TSP ACO: Population Evolution')
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Tour Length')
    
    plt.tight_layout()
    plt.show()

def plot_tsp_3d_heatmap(aco):
    """Hiển thị evolution của TSP lưới tour dưới dạng heatmap"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap: tệula matrix evolution
    tau_evolution = []
    for tau in aco.history_tau:
        tau_flat = tau.flatten()
        tau_evolution.append(tau_flat)
    
    heatmap = np.array(tau_evolution)
    im = ax1.imshow(heatmap, cmap='YlOrBr', aspect='auto')
    ax1.set_xlabel('Edge (Pheromone)')
    ax1.set_ylabel('Iteration')
    ax1.set_title('TSP Pheromone Evolution Over Iterations')
    plt.colorbar(im, ax=ax1, label='Pheromone Level')
    
    # Convergence curve
    ax2.plot(aco.history_best, linewidth=2.5, color='darkblue')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best Tour Length')
    ax2.set_title('Convergence Curve')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("1. ACOR (Continuous)")
    print("2. ACO (TSP - Discrete)")
    print("3. 3D Visualization (Continuous)")
    print("4. 3D Visualization (TSP)")
    choice = input("Choose: ")

    if choice == "1":
        acor = ACOR(
            func=sphere,
            dim=2,
            bounds=(-5, 5),
            n_ants=30,
            n_iter=100
        )
        acor.run()
        print("\n===== FINAL RESULT =====")
        print("Best value:", acor.best_val)
        print("Best solution:", acor.best)
        animate_acor(acor)
        plot_acor_convergence(acor)
        cont = input("Chạy multiple test để thống kê (Y/N): ").strip()
        if cont == "Y":
            multiple_runs_acor(n_runs=20)
    elif choice == "2":
        coords = create_cluster_cities()
        dist = np.linalg.norm(
            coords[:, np.newaxis, :] -
            coords[np.newaxis, :, :],
            axis=2
        )
        aco = AntColony(dist, n_ants=30, n_iter=100)
        aco.run()
        print("\n===== FINAL RESULT =====")
        print("Best length:", aco.best_length)
        print("Best path:", aco.best_path)
        animate_aco(coords, aco)
        plot_final_report(coords, aco)
        cont = input("Chạy multiple test để thống kê (Y/N): ").strip()
        if cont == "Y":
            multiple_runs(dist, n_runs=20)
    elif choice == "3":
        acor = ACOR(
            func=sphere,
            dim=2,
            bounds=(-5, 5),
            n_ants=30,
            n_iter=100
        )
        acor.run()
        plot_search_dynamics_acor(acor)
    else:
        coords = create_cluster_cities()
        dist = np.linalg.norm(
            coords[:, np.newaxis, :] -
            coords[np.newaxis, :, :],
            axis=2
        )
        aco = AntColony(dist, n_ants=30, n_iter=100)
        aco.run()
        plot_tsp_evolution_3d(aco)
