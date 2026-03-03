import os
import sys

try:
    from bfs import BFS, run_comprehensive_stats as run_bfs_stats
    from dfs import DFS, run_dfs_statistics
    from ucs import UCS, run_ucs_statistics
    from gbfs import GBFS, run_gbfs_statistics
    from astar import AStar, run_astar_statistics
    from hill_climbing import execute_and_report as run_hc
except ImportError as e:
    print(f"[!] Lỗi: Không tìm thấy file {e.name}.")
    sys.exit()

def print_menu():
    print("\n" + "="*50)
    print("      CLASSICAL SEARCH:")
    print("="*50)
    print("1. Breadth-First Search (BFS)")
    print("2. Depth-First Search (DFS)")
    print("3. Uniform Cost Search (UCS)")
    print("4. Greedy Best-First Search (GBFS)")
    print("5. A* Search")
    print("6. Hill Climbing (Steepest Ascent)")
    print("0. Thoát chương trình")
    print("-" * 50)

def main():
    while True:
        print_menu()
        choice = input("Chọn thuật toán muốn thực thi (0-7): ")

        if choice == '0':
            print("Cảm ơn bạn đã sử dụng chương trình. Tạm biệt!")
            break

        if choice == '1':
            print("\n[*] Đang chạy BFS (Shortest Path)...")
            demo = BFS(grid_size=20, obstacle_prob=0.15)
            ok, p, n, _ = demo.search()
            if ok: demo.save_visual_result(p, n)
            run_bfs_stats(n_runs=20) 

        elif choice == '2':
            print("\n[*] Đang chạy DFS...")
            demo = DFS(grid_size=20, obstacle_prob=0.15)
            ok, p, n, _ = demo.search()
            if ok: demo.save_visual_result(p, n)
            run_dfs_statistics(n_runs=20)

        elif choice == '3':
            print("\n[*] Đang chạy UCS...")
            demo = UCS(grid_size=20, obstacle_prob=0.15)
            ok, p, n, _, c = demo.search()
            if ok: demo.save_visual_result(p, n, c)
            run_ucs_statistics(n_runs=20)

        elif choice == '4':
            print("\n[*] Đang chạy GBFS...")
            demo = GBFS(grid_size=20, obstacle_prob=0.15)
            ok, p, n, _ = demo.search()
            if ok: demo.save_visual_result(p, n)
            run_gbfs_statistics(n_runs=20)

        elif choice == '5':
            print("\n[*] Đang chạy A*...")
            demo = AStar(grid_size=20, obstacle_prob=0.15)
            ok, p, n, _ = demo.search()
            if ok: demo.save_visual_result(p, n)
            run_astar_statistics(n_runs=20)

        elif choice == '6':
            print("\n[*] Đang chạy Hill Climbing (Discrete & Continuous)...")
            run_hc() 

        else:
            print("[!] Lựa chọn không hợp lệ, vui lòng chọn lại.")

        print("\n[V] Đã hoàn thành! Hãy kiểm tra các file .png và .csv trong thư mục.")

if __name__ == "__main__":
    main()