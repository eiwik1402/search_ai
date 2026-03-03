import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def sphere(x):
    return np.sum(x**2)

def knapsack(x, w, v, W):
    weight = np.sum(x * w)
    value = np.sum(x * v)
    # Penalty mạnh hơn cho nghiệm không hợp lệ
    return value if weight <= W else value - 1000 * (weight - W)

class Firefly:
    def __init__(self, func, dim, bounds,
                 n_fireflies=20, n_iter=100,
                 alpha=0.2, beta0=1.0, gamma=2.0):

        self.func = func
        self.dim = dim
        self.lb, self.ub = bounds
        self.n_fireflies = n_fireflies
        self.n_iter = n_iter

        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

        self.X = np.random.uniform(self.lb, self.ub, (n_fireflies, dim))
        self.fitness = np.array([func(x) for x in self.X])

        self.best = None
        self.best_val = np.inf
        self.history_best = []
        self.history_pos = []

    def run(self):
        for _ in range(self.n_iter):
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if self.fitness[j] < self.fitness[i]:
                        r = np.linalg.norm(self.X[i] - self.X[j])
                        beta = self.beta0 * np.exp(-self.gamma * r**2)
                        self.X[i] += beta * (self.X[j] - self.X[i]) + \
                                     self.alpha * (np.random.rand(self.dim) - 0.5)

                self.X[i] = np.clip(self.X[i], self.lb, self.ub)
                self.fitness[i] = self.func(self.X[i])

            idx = np.argmin(self.fitness)
            if self.fitness[idx] < self.best_val:
                self.best_val = self.fitness[idx]
                self.best = self.X[idx].copy()

            self.history_best.append(self.best_val)
            self.history_pos.append(self.X.copy())

class BinaryFirefly:
    def __init__(self, w, v, W, n_fireflies=30, n_iter=100, alpha=0.2, beta0=1.0, gamma=1.0):
        self.w, self.v, self.W = w, v, W
        self.n = len(w)
        self.n_fireflies = n_fireflies
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

        self.X = np.random.randint(0, 2, (n_fireflies, self.n))
        self.fitness = np.array([knapsack(x, w, v, W) for x in self.X])

        self.best = None
        self.best_val = -np.inf
        self.history_best = []
        self.history_pos = []

    def run(self):
        for iteration in range(self.n_iter):
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if self.fitness[j] > self.fitness[i]:
                        # Hamming distance
                        r = np.sum(self.X[i] != self.X[j]) / self.n
                        
                        # Attractiveness
                        beta = self.beta0 * np.exp(-self.gamma * r**2)
                        
                        # Move towards brighter firefly
                        for k in range(self.n):
                            if np.random.rand() < beta:
                                self.X[i][k] = self.X[j][k]
                
                # Random walk (alpha parameter)
                if np.random.rand() < self.alpha:
                    flip_idx = np.random.randint(0, self.n)
                    self.X[i][flip_idx] = 1 - self.X[i][flip_idx]
                
                # Update fitness
                self.fitness[i] = knapsack(self.X[i], self.w, self.v, self.W)
            
            # Update best solution
            idx = np.argmax(self.fitness)
            if self.fitness[idx] > self.best_val:
                self.best_val = self.fitness[idx]
                self.best = self.X[idx].copy()

            self.history_best.append(self.best_val)
            self.history_pos.append(self.X.copy())

def animate_fa(fa):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    def update(i):
        ax1.clear()
        ax2.clear()

        X = fa.history_pos[i]
        ax1.scatter(X[:, 0], X[:, 1], alpha=0.7)
        ax1.scatter(fa.best[0], fa.best[1], marker="*", s=200, c='red')
        ax1.set_title(f"Firefly Algorithm (Iter {i+1})")
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.grid(alpha=0.3)

        ax2.plot(fa.history_best[:i+1], lw=2.5)
        ax2.set_title("Convergence")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Best Fitness")
        ax2.grid(alpha=0.3)

    anim = FuncAnimation(fig, update, frames=len(fa.history_best), interval=250)
    plt.tight_layout()
    plt.show()

def animate_binary_fa(bfa):
    fig, ax = plt.subplots(figsize=(8, 5))

    def update(i):
        ax.clear()
        ax.plot(bfa.history_best[:i+1], lw=2.5, color='blue')
        ax.set_xlim(0, len(bfa.history_best))

        if i > 0:
            ymin = min(bfa.history_best[:i+1])
            ymax = max(bfa.history_best[:i+1])
            margin = (ymax - ymin) * 0.1 if ymax > ymin else 1
            ax.set_ylim(ymin - margin, ymax + margin)

        ax.set_title(f"Binary Firefly – Convergence (Iter {i+1})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness (Value)")
        ax.grid(alpha=0.3)

    anim = FuncAnimation(
        fig, update,
        frames=len(bfa.history_best),
        interval=250,
        repeat=False
    )
    plt.tight_layout()
    plt.show()

def plot_fa_report(fa):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    X = np.vstack(fa.history_pos)
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.6, label="Fireflies")
    ax1.scatter(fa.best[0], fa.best[1],
                marker="*", s=250, c="red", label="Best")
    ax1.set_title(f"Best Solution (f = {fa.best_val:.4f})")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(fa.history_best, lw=2.5)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best fitness")
    ax2.set_title("Convergence Curve")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def multiple_runs_fa():
    results = []
    for _ in range(20):
        fa = Firefly(sphere, 2, (-5, 5))
        fa.run()
        results.append(fa.best_val)

    r = np.array(results)
    print("\n===== CONTINUOUS FA MULTIPLE RUNS =====")
    print(f"Mean:   {r.mean():.6e}")
    print(f"Std:    {r.std():.6e}")
    print(f"Best:   {r.min():.6e}")
    print(f"Worst:  {r.max():.6e}")

def plot_binary_fa_report(bfa, w):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    items = [f"Item {i+1}" for i in range(bfa.n)]
    colors = ['red' if bfa.best[i] == 1 else 'gray' for i in range(bfa.n)]
    ax1.bar(items, bfa.best * w, color=colors, alpha=0.7)
    ax1.set_title(f"Best Solution (Value = {bfa.best_val:.2f})")
    ax1.set_ylabel("Weight")
    ax1.set_xlabel("Items")
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.plot(bfa.history_best, lw=2.5, color='darkred')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best Fitness Value")
    ax2.set_title("Convergence Curve")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def multiple_runs_binary_fa(w, v, W, n_runs=20):
    results = []
    for _ in range(n_runs):
        bfa = BinaryFirefly(w, v, W)
        bfa.run()
        results.append(bfa.best_val)
    
    r = np.array(results)
    print(f"\n===== BINARY FA MULTIPLE RUNS ({n_runs} runs) =====")
    print(f"Mean:   {r.mean():.6e}")
    print(f"Std:    {r.std():.6e}")
    print(f"Best:   {r.max():.6e}")
    print(f"Worst:  {r.min():.6e}")

def plot_search_dynamics_fa(fa):
    """Hiển thị không gian tìm kiếm + phân bố solutions từ tất cả iterations

    Kèm thêm contour 2D bên cạnh biểu đồ 3D để báo cáo trực quan.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # chuẩn bị lưới
    x = np.linspace(fa.lb, fa.ub, 40)
    y = np.linspace(fa.lb, fa.ub, 40)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    fig = plt.figure(figsize=(14, 7))
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3, edgecolor='none')
    all_solutions = np.vstack(fa.history_pos)
    Z_solutions = np.sum(all_solutions**2, axis=1)
    ax3.scatter(all_solutions[:, 0], all_solutions[:, 1], Z_solutions,
               c='red', marker='o', s=50, alpha=0.5, label='All Solutions')
    Z_best = fa.best_val
    ax3.scatter(fa.best[0], fa.best[1], Z_best,
               c='gold', marker='*', s=800, edgecolors='black', linewidths=2,
               label='Best solution', zorder=10)
    ax3.set_xlabel('x₁')
    ax3.set_ylabel('x₂')
    ax3.set_zlabel('f(x)')
    ax3.set_title('3D Surface + Search Dynamics (FA)')
    ax3.legend()

    cont = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax2.scatter(all_solutions[:, 0], all_solutions[:, 1], c='red', s=20, alpha=0.5)
    ax2.scatter(fa.best[0], fa.best[1], c='gold', s=100, edgecolors='black')
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_title('Contour with Firefly Locations')
    fig.colorbar(cont, ax=ax2)

    plt.tight_layout()
    plt.show()

def plot_3d_sphere():
    """Hiển thị cả surface 3D và contour của hàm Sphere để so sánh."""
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

def plot_binary_evolution_3d(bfa):
    """Hiển thị evolution của binary solutions dưới dạng 3D"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    iterations = []
    solution_indices = []
    fitness_values = []
    
    for iter_idx in range(len(bfa.history_pos)):
        iter_X = bfa.history_pos[iter_idx]
        for sol_idx in range(len(iter_X)):
            fit_val = knapsack(iter_X[sol_idx], bfa.w, bfa.v, bfa.W)
            iterations.append(iter_idx)
            solution_indices.append(sol_idx)
            fitness_values.append(fit_val)
    
    scatter = ax.scatter(iterations, solution_indices, fitness_values, 
                        c=fitness_values, cmap='viridis', s=50, alpha=0.6)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Firefly Index')
    ax.set_zlabel('Fitness Value')
    ax.set_title('Binary FA: Population Evolution')
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Fitness')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("1. Continuous FA")
    print("2. Binary FA")
    print("3. 3D Visualization (Continuous)")
    print("4. 3D Visualization (Binary)")
    choice = input("Choose: ")

    if choice == "1":
        fa = Firefly(sphere, 2, (-5, 5))
        fa.run()
        print("\n===== FINAL RESULT =====")
        print("Best value:", fa.best_val)
        print("Best solution:", fa.best)
        animate_fa(fa)
        plot_fa_report(fa)
        cont = input("Chạy multiple test để thống kê (Y/N): ").strip()
        if cont == "Y":
            multiple_runs_fa()
    elif choice == "2":
        w = np.array([2, 3, 4, 5, 9])
        v = np.array([3, 4, 5, 8, 10])
        W = 15

        bfa = BinaryFirefly(w, v, W)
        bfa.run()
        print("\n===== FINAL RESULT =====")
        print("Best solution:", bfa.best)
        print("Best value:", bfa.best_val)
        print("Total weight:", np.sum(bfa.best * w))
        
        animate_binary_fa(bfa)
        plot_binary_fa_report(bfa, w)
        cont = input("Chạy multiple test để thống kê (Y/N): ").strip()
        if cont == "Y":
            multiple_runs_binary_fa(w, v, W, n_runs=20)
    elif choice == "3":
        fa = Firefly(sphere, 2, (-5, 5))
        fa.run()
        plot_search_dynamics_fa(fa)
    else:
        w = np.array([2, 3, 4, 5, 9])
        v = np.array([3, 4, 5, 8, 10])
        W = 15
        bfa = BinaryFirefly(w, v, W)
        bfa.run()
        plot_binary_evolution_3d(bfa)