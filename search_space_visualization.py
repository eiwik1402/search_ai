import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

try:
    from algo_swarm.aco import create_cluster_cities
except Exception:
    def create_cluster_cities(n_clusters=5, cities_per_cluster=6, seed=42):
        np.random.seed(seed)
        coords = []
        for i in range(n_clusters):
            angle = 2 * np.pi * i / n_clusters
            center = np.array([5 * np.cos(angle), 5 * np.sin(angle)])
            cluster = np.random.randn(cities_per_cluster, 2) * 0.4 + center
            coords.append(cluster)
        return np.vstack(coords)

def plot_shortest_path_space(grid_size: int = 30, obstacle_prob: float = 0.15, seed: int = 42) -> None:
    np.random.seed(seed)
    grid = (np.random.rand(grid_size, grid_size) > obstacle_prob).astype(int)
    start = (0, 0)
    goal = (grid_size - 1, grid_size - 1)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='Greys_r', origin='upper')
    plt.scatter(start[1], start[0], color='green', s=180, edgecolors='black', linewidth=2, label='Start')
    plt.scatter(goal[1], goal[0], color='red', s=180, edgecolors='black', linewidth=2, label='Goal')
    plt.title('Shortest Path — Search Space (20 x 20)')
    plt.axis('off')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_tsp_space(n_clusters: int = 5, cities_per_cluster: int = 6, seed: int = 42) -> None:
    coords = create_cluster_cities(n_clusters=n_clusters, cities_per_cluster=cities_per_cluster, seed=seed)
    plt.figure(figsize=(7, 6))
    plt.scatter(coords[:, 0], coords[:, 1], s=120, color='tab:blue', zorder=5)
    for i, (x, y) in enumerate(coords):
        plt.annotate(str(i), (x, y), fontsize=8, ha='center', va='center')
    plt.title('TSP — Cities (5 clusters - 6 cities each)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_knapsack_space(n_items: int = 20, capacity: int = 50, seed: int = 42) -> None:
    """Visualize Knapsack Problem - Items with weights and values"""
    np.random.seed(seed)
    weights = np.random.randint(5, 20, n_items)
    values = np.random.randint(10, 50, n_items)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Items with weights and values
    indices = np.arange(n_items)
    ax1.scatter(weights, values, s=150, color='tab:blue', alpha=0.7, edgecolors='black', linewidth=1.5)
    for i, (w, v) in enumerate(zip(weights, values)):
        ax1.annotate(str(i), (w, v), fontsize=7, ha='center', va='center', fontweight='bold')
    ax1.set_xlabel('Weight', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax1.set_title(f'Knapsack Problem — {n_items} Items (Capacity = {capacity})', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.axhline(y=np.mean(values), color='red', linestyle='--', alpha=0.5, label='Mean Value')
    ax1.axvline(x=np.mean(weights), color='green', linestyle='--', alpha=0.5, label='Mean Weight')
    ax1.legend()
    
    # Plot 2: Value-to-Weight Ratio (Greedy Indicator)
    ratios = values / weights
    sorted_indices = np.argsort(ratios)[::-1]
    sorted_ratios = ratios[sorted_indices]
    sorted_weights = weights[sorted_indices]
    sorted_values = values[sorted_indices]
    
    colors = ['green' if sum(sorted_weights[:i+1]) <= capacity else 'red' 
              for i in range(len(sorted_weights))]
    
    bars = ax2.barh(range(n_items), sorted_ratios, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_yticks(range(n_items))
    ax2.set_yticklabels([f"Item {int(sorted_indices[i])}" for i in range(n_items)], fontsize=8)
    ax2.set_xlabel('Value/Weight Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Items Sorted by Value-to-Weight Ratio', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def rastrigin(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return 20 + X ** 2 - 10 * np.cos(2 * np.pi * X) + Y ** 2 - 10 * np.cos(2 * np.pi * Y)

def plot_rastrigin(bounds: Tuple[float, float] = (-5.12, 5.12), res: int = 200) -> None:
    x = np.linspace(bounds[0], bounds[1], res)
    y = np.linspace(bounds[0], bounds[1], res)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin(X, Y)
    fig = plt.figure(figsize=(12, 6))
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax3.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.9)
    ax3.set_title('Rastrigin Function — 3D Surface')
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_zlabel('f(x)')
    fig.colorbar(surf, ax=ax3, shrink=0.6)
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax2.scatter(0, 0, color='gold', s=160, marker='*', edgecolors='black', label='Global Optimum')
    ax2.set_title('Rastrigin Function — 2D Contour')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    fig.colorbar(contour, ax=ax2)
    ax2.legend()
    plt.tight_layout()
    plt.show()

def plot_sphere(bounds: Tuple[float, float] = (-5.12, 5.12), res: int = 200) -> None:
    x = np.linspace(bounds[0], bounds[1], res)
    y = np.linspace(bounds[0], bounds[1], res)
    X, Y = np.meshgrid(x, y)
    Z = X ** 2 + Y ** 2
    fig = plt.figure(figsize=(12, 6))
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax3.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.9)
    ax3.set_title('Sphere Function — 3D Surface')
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_zlabel('f(x)')
    fig.colorbar(surf, ax=ax3, shrink=0.6, pad=0.08)
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax2.scatter(0, 0, color='black', s=100, marker='x', label='Optimum')
    ax2.set_title('Sphere Function — 2D Contour')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    fig.colorbar(contour, ax=ax2)
    ax2.legend()
    plt.tight_layout()
    plt.show()


def main_menu() -> None:
    while True:
        print('\n' + '=' * 50)
        print(' Search Space Visualizations')
        print('=' * 50)
        print('1. Shortest Path (Grid search space) - Discrete')
        print('2. Travelling Salesman Problem (Clustered cities) - Discrete')
        print('3. Knapsack Problem (Items with weights/values) - Discrete')
        print('4. Rastrigin function (Continuous)')
        print('5. Sphere function (Continuous)')
        print('0. Exit')
        try:
            choice = int(input('Enter choice: ').strip())
        except ValueError:
            print('Invalid input.')
            continue

        if choice == 0:
            break
        elif choice == 1:
            plot_shortest_path_space()
            input("Press ENTER to continue...")
        elif choice == 2:
            plot_tsp_space()
            input("Press ENTER to continue...")
        elif choice == 3:
            plot_knapsack_space()
            input("Press ENTER to continue...")
        elif choice == 4:
            plot_rastrigin()
            input("Press ENTER to continue...")
        elif choice == 5:
            plot_sphere()
            input("Press ENTER to continue...")
        else:
            print('Invalid choice.')

if __name__ == '__main__':
    main_menu()
