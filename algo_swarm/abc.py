import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def sphere(x):
    return np.sum(x ** 2)

class ABC:
    def __init__(self, func, dim, bounds,
                 n_foods=20, n_iter=100, limit=20, phi_max=1.0, p_select=1.0):

        self.func = func
        self.dim = dim
        self.x_min, self.x_max = bounds
        self.n_foods = n_foods
        self.n_iter = n_iter
        self.limit = limit
        self.phi_max = phi_max
        self.p_select = p_select

        # Food sources
        self.X = np.random.uniform(self.x_min, self.x_max,
                                   (n_foods, dim))
        self.fitness = np.array([self.func(x) for x in self.X])
        self.trial = np.zeros(n_foods)

        idx = np.argmin(self.fitness)
        self.best = self.X[idx].copy()
        self.best_val = self.fitness[idx]

        self.history_pos = []
        self.history_best = []

    def _generate_neighbor(self, i):
        k = np.random.choice([j for j in range(self.n_foods) if j != i])
        phi = np.random.uniform(-self.phi_max,
                        self.phi_max,
                        self.dim)
        v = self.X[i] + phi * (self.X[i] - self.X[k])
        return np.clip(v, self.x_min, self.x_max)

    def run(self):
        for _ in range(self.n_iter):

            # ----- Employed bees -----
            for i in range(self.n_foods):
                v = self._generate_neighbor(i)
                fv = self.func(v)
                if fv < self.fitness[i]:
                    self.X[i] = v
                    self.fitness[i] = fv
                    self.trial[i] = 0
                else:
                    self.trial[i] += 1

            # ----- Onlooker bees -----
            prob = (1 / (1 + self.fitness)) ** self.p_select
            prob /= np.sum(prob)
            for _ in range(self.n_foods):
                i = np.random.choice(range(self.n_foods), p=prob)
                v = self._generate_neighbor(i)
                fv = self.func(v)
                if fv < self.fitness[i]:
                    self.X[i] = v
                    self.fitness[i] = fv
                    self.trial[i] = 0
                else:
                    self.trial[i] += 1

            # ----- Scout bees -----
            for i in range(self.n_foods):
                if self.trial[i] > self.limit:
                    self.X[i] = np.random.uniform(self.x_min, self.x_max, self.dim)
                    self.fitness[i] = self.func(self.X[i])
                    self.trial[i] = 0

            # Update global best
            idx = np.argmin(self.fitness)
            if self.fitness[idx] < self.best_val:
                self.best_val = self.fitness[idx]
                self.best = self.X[idx].copy()

            self.history_pos.append(self.X.copy())
            self.history_best.append(self.best_val)


def animate_abc(abc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    def update(frame):
        ax1.clear()
        ax2.clear()

        X = abc.history_pos[frame]

        ax1.scatter(X[:, 0], X[:, 1], alpha=0.6)
        ax1.scatter(abc.best[0], abc.best[1],
                    c='red', marker='*', s=150)
        ax1.set_xlim(abc.x_min, abc.x_max)
        ax1.set_ylim(abc.x_min, abc.x_max)
        ax1.set_title(f"Iteration {frame + 1}")

        ax2.plot(abc.history_best[:frame + 1], linewidth=2)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Best f(x)")
        ax2.set_title("Convergence curve")
        ax2.grid(alpha=0.3)

    anim = FuncAnimation(fig, update,
                         frames=abc.n_iter,
                         interval=250)
    plt.show()


def plot_convergence(abc):
    all_positions = np.vstack(abc.history_pos)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(all_positions[:, 0],
                all_positions[:, 1],
                s=10, alpha=0.25, label="Visited solutions")

    ax1.scatter(abc.best[0], abc.best[1],
                c='red', marker='*', s=180,
                label="Best solution")

    ax1.set_xlim(abc.x_min, abc.x_max)
    ax1.set_ylim(abc.x_min, abc.x_max)
    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax1.set_title("Solution distribution")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(abc.history_best, linewidth=2.5)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best f(x)")
    ax2.set_title("Convergence curve")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.show()

def multiple_runs_statistics_abc(n_runs=20):
    results = []

    for i in range(n_runs):
        abc = ABC(
            func=sphere,
            dim=2,
            bounds=(-5.12, 5.12),
            n_foods=20,
            n_iter=100,
            limit=5,
            phi_max=1.0,
            p_select=2.0
        )
        abc.run()
        results.append(abc.best_val)

    results = np.array(results)

    print("\n===== MULTIPLE RUNS STATISTICS (ABC) =====")
    print(f"Number of runs: {n_runs}")
    print(f"Mean best value: {results.mean():.6e}")
    print(f"Std deviation:  {results.std():.6e}")
    print(f"Best value:     {results.min():.6e}")
    print(f"Worst value:   {results.max():.6e}")

    return results

def plot_search_dynamics(abc):
    """Hiển thị không gian tìm kiếm + phân bố solutions từ tất cả iterations

    Kèm thêm contour 2D để người dùng nhìn thấy mặt cắt cạnh biểu đồ 3D.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    x = np.linspace(abc.x_min, abc.x_max, 40)
    y = np.linspace(abc.x_min, abc.x_max, 40)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    fig = plt.figure(figsize=(14, 7))
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3, edgecolor='none')
    all_positions = np.vstack(abc.history_pos)
    Z_solutions = np.sum(all_positions**2, axis=1)
    ax3.scatter(all_positions[:, 0], all_positions[:, 1], Z_solutions,
               c='red', marker='o', s=50, alpha=0.5, label='All Solutions')
    Z_best = abc.best_val
    ax3.scatter(abc.best[0], abc.best[1], Z_best,
               c='gold', marker='*', s=800, edgecolors='black', linewidths=2,
               label='Best solution', zorder=10)
    ax3.set_xlabel('x₁')
    ax3.set_ylabel('x₂')
    ax3.set_zlabel('f(x)')
    ax3.set_title('3D Surface + Search Dynamics (ABC)')
    ax3.legend()

    cont = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax2.scatter(all_positions[:, 0], all_positions[:, 1], c='red', s=20, alpha=0.5)
    ax2.scatter(abc.best[0], abc.best[1], c='gold', s=100, edgecolors='black')
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_title('Contour with Food Locations')
    fig.colorbar(cont, ax=ax2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("1. Bài toán liên tục")
    print("2. 3D Visualization")
    choice = input("Choose (1 or 2): ")

    if choice == "1":
        abc = ABC(
            func=sphere,
            dim=2,
            bounds=(-5.12, 5.12),
            n_foods=20,
            n_iter=100,
            limit=5,
            phi_max=1.0,
            p_select=2.0
        )

        abc.run()
        print("\n===== FINAL RESULT =====")
        print("Best value:", abc.best_val)
        print("Best solution:", abc.best)

        animate_abc(abc)
        plot_convergence(abc)
        cont = input("Chạy multiple test để thống kê (Y/N): ").strip()
        if cont == "Y": 
            multiple_runs_statistics_abc(n_runs=20)
    else:
        abc = ABC(
            func=sphere,
            dim=2,
            bounds=(-5.12, 5.12),
            n_foods=20,
            n_iter=100,
            limit=5,
            phi_max=1.0,
            p_select=2.0
        )
        abc.run()
        plot_search_dynamics(abc)
