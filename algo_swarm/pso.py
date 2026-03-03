import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def sphere(x):
    return np.sum(x ** 2)

def binary_objective(x):
    return np.sum(x)

class PSO:
    def __init__(self, func, dim, bounds,
                 n_particles=20, n_iter=100,
                 w=0.9, w_min=0.6,
                 c1=2.0, c2=2.0,
                 discrete=False):

        self.func = func
        self.dim = dim
        self.x_min, self.x_max = bounds
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.w = w
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.discrete = discrete

        if discrete:
            self.X = np.random.randint(0, 2, (n_particles, dim))
        else:
            self.X = np.random.uniform(self.x_min, self.x_max,
                                       (n_particles, dim))

        self.V = np.zeros((n_particles, dim))

        self.pbest = self.X.copy()
        self.pbest_val = np.array([self.func(x) for x in self.X])

        idx = np.argmin(self.pbest_val)
        self.gbest = self.pbest[idx].copy()
        self.gbest_val = self.pbest_val[idx]

        self.history_pos = []
        self.history_best = []

    def sigmoid(self, v):
        return 1 / (1 + np.exp(-v))

    def run(self):
        for t in range(self.n_iter):
            w_t = self.w - (self.w - self.w_min) * (t / self.n_iter)

            for i in range(self.n_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                self.V[i] = (
                    w_t * self.V[i]
                    + self.c1 * r1 * (self.pbest[i] - self.X[i])
                    + self.c2 * r2 * (self.gbest - self.X[i])
                )

                if self.discrete:
                    prob = self.sigmoid(self.V[i])
                    self.X[i] = (np.random.rand(self.dim) < prob).astype(int)
                else:
                    self.X[i] += self.V[i]
                    self.X[i] = np.clip(self.X[i], self.x_min, self.x_max)

                val = self.func(self.X[i])
                if val < self.pbest_val[i]:
                    self.pbest[i] = self.X[i].copy()
                    self.pbest_val[i] = val

            idx = np.argmin(self.pbest_val)
            if self.pbest_val[idx] < self.gbest_val:
                self.gbest_val = self.pbest_val[idx]
                self.gbest = self.pbest[idx].copy()

            self.history_pos.append(self.X.copy())
            self.history_best.append(self.gbest_val)

def animate_pso(pso):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    def update(frame):
        ax1.clear()
        ax2.clear()

        X = pso.history_pos[frame]
        if not pso.discrete and pso.dim == 2:
            ax1.scatter(X[:, 0], X[:, 1], alpha=0.6)
            ax1.scatter(pso.gbest[0], pso.gbest[1],
                        c='red', marker='*', s=150)
            ax1.set_xlim(pso.x_min, pso.x_max)
            ax1.set_ylim(pso.x_min, pso.x_max)
            ax1.set_title(f"Iteration {frame + 1}")

        else:
            fitness = np.array([pso.func(x) for x in X])
            ax1.scatter(range(len(fitness)), fitness, alpha=0.6)
            ax1.scatter(np.argmin(fitness), np.min(fitness),
                        c='red', marker='*', s=150)
            ax1.set_xlabel("Particle index")
            ax1.set_ylabel("Fitness")
            ax1.set_title(f"Iteration {frame + 1}")

        ax2.plot(pso.history_best[:frame + 1], linewidth=2)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Best f(x)")
        ax2.set_title("Convergence curve")
        ax2.grid(alpha=0.3)

    anim = FuncAnimation(fig, update,
                         frames=pso.n_iter,
                         interval=150)
    plt.show()

def plot_continuous_report(pso):
    X_final = np.vstack(pso.history_pos)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(X_final[:, 0], X_final[:, 1], alpha=0.6, label="Particles")
    ax[0].scatter(
        pso.gbest[0], pso.gbest[1],
        marker="*", s=200, label="Global best"
    )
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")
    ax[0].set_title("Solution distribution (final iteration)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(pso.history_best, linewidth=2)
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Best f(x)")
    ax[1].set_title("Convergence curve")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_discrete_report(pso):
    X_final = np.vstack(pso.history_pos)
    fitness = np.array([pso.func(x) for x in X_final])
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(fitness, bins=10, alpha=0.7)
    ax[0].set_xlabel("Fitness value")
    ax[0].set_ylabel("Frequency")
    ax[0].set_title("Fitness distribution (final iteration)")
    ax[0].grid(alpha=0.3)
    ax[1].plot(pso.history_best, linewidth=2)
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Best f(x)")
    ax[1].set_title("Convergence curve")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def multiple_runs_statistics(n_runs=30):
    results = []

    for i in range(n_runs):
        pso = PSO(
            func=sphere,
            dim=2,
            bounds=(-5, 5),
            n_particles=60,
            n_iter=100,
            discrete=False
        )
        pso.run()
        results.append(pso.gbest_val)

    results = np.array(results)

    print("\n===== MULTIPLE RUNS STATISTICS =====")
    print(f"Number of runs: {n_runs}")
    print(f"Mean best value: {results.mean():.6e}")
    print(f"Std deviation:  {results.std():.6e}")
    print(f"Best value:     {results.min():.6e}")
    print(f"Worst value:   {results.max():.6e}")

    return results

def plot_search_dynamics_pso(pso):
    """Hiển thị không gian tìm kiếm + phân bố solutions từ tất cả iterations

    Cài đặt hai biểu đồ: bề mặt 3D và contour 2D bên cạnh để đơn báo cáo trực quan hóa.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # tạo lưới điểm cho đồ thị
    x = np.linspace(pso.x_min, pso.x_max, 40)
    y = np.linspace(pso.x_min, pso.x_max, 40)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    # mở figure với hai subplot (3D và 2D)
    fig = plt.figure(figsize=(14, 7))
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    # --- vẽ bề mặt 3D như trước ---
    ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3, edgecolor='none')
    all_solutions = np.vstack(pso.history_pos)
    Z_solutions = np.sum(all_solutions**2, axis=1)
    ax3.scatter(all_solutions[:, 0], all_solutions[:, 1], Z_solutions,
                c='red', marker='o', s=50, alpha=0.5, label='All Particles')
    Z_best = pso.gbest_val
    ax3.scatter(pso.gbest[0], pso.gbest[1], Z_best,
                c='gold', marker='*', s=800, edgecolors='black', linewidths=2,
                label='Best solution', zorder=10)
    ax3.set_xlabel('x₁')
    ax3.set_ylabel('x₂')
    ax3.set_zlabel('f(x)')
    ax3.set_title('3D Surface + Search Dynamics')
    ax3.legend()

    # --- vẽ contour plan phía bên ---
    cont = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax2.scatter(all_solutions[:, 0], all_solutions[:, 1], c='red', s=20, alpha=0.5)
    ax2.scatter(pso.gbest[0], pso.gbest[1], c='gold', s=100, edgecolors='black')
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_title('Contour with Particle Locations')
    fig.colorbar(cont, ax=ax2)

    plt.tight_layout()
    plt.show()

def plot_3d_sphere():
    """Hiển thị phối hợp Surface 3D và Contour 2D của hàm Sphere."""
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

    cont = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title('2D Contour: Sphere Function')
    fig.colorbar(cont, ax=ax2)

    plt.tight_layout()
    plt.show()

def plot_binary_evolution_3d(pso):
    """Hiển thị evolution của binary solutions dưới dạng 3D"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    iterations = []
    particle_indices = []
    fitness_values = []
    
    for iter_idx in range(len(pso.history_pos)):
        iter_X = pso.history_pos[iter_idx]
        for particle_idx in range(len(iter_X)):
            fit_val = pso.func(iter_X[particle_idx])
            iterations.append(iter_idx)
            particle_indices.append(particle_idx)
            fitness_values.append(fit_val)
    
    scatter = ax.scatter(iterations, particle_indices, fitness_values, 
                        c=fitness_values, cmap='viridis', s=50, alpha=0.6)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Particle Index')
    ax.set_zlabel('Fitness Value')
    ax.set_title('Binary PSO: Population Evolution')
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Fitness')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("1. PSO rời rạc (Binary)")
    print("2. PSO liên tục")
    print("3. 3D Visualization (Continuous)")
    print("4. 3D Visualization (Binary)")
    choice = input("Choose (1, 2, 3 or 4): ")

    if choice == "1":
        pso = PSO(
            func=binary_objective,
            dim=30,
            bounds=(0, 1),
            n_particles=20,
            n_iter=100,
            discrete=True
        )
    elif choice == "2":
        pso = PSO(
            func=sphere,
            dim=2,
            bounds=(-5, 5),
            n_particles=20,
            n_iter=100,
            discrete=False
        )
    elif choice == "3":
        pso = PSO(
            func=sphere,
            dim=2,
            bounds=(-5, 5),
            n_particles=20,
            n_iter=100,
            discrete=False
        )
        pso.run()
        plot_search_dynamics_pso(pso)
        exit()
    else:
        pso = PSO(
            func=binary_objective,
            dim=30,
            bounds=(0, 1),
            n_particles=20,
            n_iter=100,
            discrete=True
        )
        pso.run()
        plot_binary_evolution_3d(pso)
        exit()

    pso.run()

    print("\n===== FINAL RESULT =====")
    print("Best value:", pso.gbest_val)
    print("Best solution:", pso.gbest)
    animate_pso(pso)
    if pso.discrete:
        plot_discrete_report(pso)
    else:
        plot_continuous_report(pso)
    cont = input("Chạy multiple test để thống kê (Y/N): ").strip()
    if cont == "Y": 
        multiple_runs_statistics(n_runs=20)
