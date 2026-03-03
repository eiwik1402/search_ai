# Đồ Án 1: Search & Nature-Inspired Algorithms

## Thông Tin Chung

- **Môn học:** CSC14003 - Cơ sở Trí tuệ Nhân tạo
- **Khoa:** Công nghệ Thông tin - Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM
- **Học kỳ:** II - Năm học: 2025-2026

---

## Mô Tả Dự Án

Dự án triển khai, phân tích và so sánh các thuật toán tìm kiếm cổ điển trên đồ thị và các thuật toán tối ưu hóa lấy cảm hứng từ tự nhiên.

---

## Danh Sách Thuật Toán 

### Tìm kiếm cổ điển (Classical Graph Search)

- **Uninformed Search:** Breadth-First Search (BFS), Depth-First Search (DFS), Uniform Cost Search (UCS)
- **Informed Search:** Greedy Best-First Search (GBFS), A* Search
- **Local Search:** Hill Climbing (Steepest Ascent)

### Biology-based (Swarm Intelligence)

- Ant Colony Optimization (ACO)
- Particle Swarm Optimization (PSO)
- Artificial Bee Colony (ABC)
- Firefly Algorithm (FA)
- Cuckoo Search (CS)

### Evolution-based

- Genetic Algorithm (GA)
- Differential Evolution (DE)

### Physics-based

- Simulated Annealing (SA)

### Human behavior-based 

- Teaching-Learning-Based Optimization (TLBO)

---

## Cấu Trúc Thư Mục

```text
├── final_results/
│   ├── comparison/                            # Kết quả so sánh của từng bài toán
│   └── statistics + scalability/              # Thống kê scalability
│       ├── classic/
│       ├── human/
│       ├── physics + evolution/
│       └── swarm/
└── src/
    ├── algo_classic/                          # Tìm kiếm cổ điển
    │   └── bfs.py, dfs.py, ucs.py, gbfs.py, astar.py, hill_climbing.py, main_classical.py
    ├── algo_evolution/                        # Evolution-based
    │   └── de.py, ga.py
    ├── algo_human_tlbo/                       # Human behavior-based
    │   └── experiment.py, main.py, objective.py, tlbo.py, visualization.py
    ├── algo_physics/                          # Physics-based
    │   └── sa_rastrigin.py, sa_sphere.py
    ├── algo_swarm/                            # Biology-based (Swarm Intelligence)
    │   └── abc.py, aco_path.py, aco.py, cs.py, fa.py, pso.py
    ├── algorithms.py                          # Menu chạy demo từng thuật toán
    ├── comparison.py                          # Menu so sánh hiệu năng
    ├── results_visualization.py               # Trực quan hóa kết quả
    ├── run_statistics.py                      # Chạy thống kê
    ├── search_space_visualization.py          # Không gian tìm kiếm
    └── requirements.txt
```

---

## Cài Đặt & Chạy
### Clone Repository
```bash
git clone https://github.com/...
cd ...
```

### Cài đặt Dependencies

Cài đặt nhanh qua file `requirements.txt`:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

### Chạy chương trình

**1. Demo Từng Thuật Toán (`algorithms.py`)**

Menu chọn lựa để chạy từng thuật toán trên các Test Case mặc định.

```bash
python algorithms.py
```

**2. So Sánh Đối Chiếu (`comparison.py`)**

Thực hiện benchmark so sánh hiệu năng giữa các nhóm thuật toán trên 5 bài toán: Shortest Path, TSP, Knapsack, Rastrigin và Sphere.

```bash
python comparison.py
```

**3. Trực Quan Hóa Kết Quả (`results_visualization.py`)**

Chạy và hiển thị kết quả từng thuật toán thông qua CLI menu tương tác. Chọn nhóm thuật toán → chọn thuật toán cụ thể → biểu đồ hiển thị tự động.

```bash
python results_visualization.py
```

**4. Xem Landscape Không Gian Tìm Kiếm (`search_space_visualization.py`)**

Vẽ các hàm mục tiêu 3D Surface plots và minh họa không gian bài toán.

```bash
python search_space_visualization.py
```

---

## Bài Toán Thử Nghiệm (Test Problems)

- **Continuous Optimization:** 
  - Sphere function (unimodal)
  - Rastrigin function (multimodal)
- **Discrete Optimization:** 
  - Traveling Salesman Problem (TSP)
  - Knapsack Problem (KP)
  - Shortest Path (GridWorld)

---

## Đọc Biểu Đồ

**Path Finding Plot** (Classic Search)

- Ô đen: vật cản
- Ô trắng: ô trống
- Điểm xanh lá: Start
- Điểm xanh dương: Goal
- Đường màu: Đường đi tìm được

**Solutions Distribution Plot** (Swarm / Evolution / Physics)

- Các chấm xanh dương: Tất cả vị trí agent đã khám phá qua các vòng lặp
- Ngôi sao Cyan / Lime / Đỏ: Nghiệm tốt nhất tìm được (Best solution)
- Ngôi sao Vàng / Trắng / Xanh lá: Vị trí tối ưu toàn cục (Global optimum)

**Convergence Plot** (Swarm / Evolution / Physics / TLBO)

- Trục X: Số vòng lặp 
- Trục Y: Giá trị fitness tốt nhất (log scale)
- Đường dốc xuống nhanh → Thuật toán hội tụ nhanh
- Đường phẳng sớm → Bị kẹt cục bộ hoặc đã hội tụ

**Population Diversity Plot** (TLBO)

- Trục X: Số vòng lặp
- Trục Y: Độ đa dạng quần thể
- Giá trị cao → Quần thể còn đang khám phá rộng
- Giá trị thấp → Quần thể đã co cụm về vùng nghiệm tốt

## Tiêu Chí Đánh Giá

| Tiêu chí | Mô tả |
|---|---|
| **Convergence Speed** | Tốc độ hội tụ qua từng vòng lặp |
| **Solution Quality** | Chất lượng nghiệm tốt nhất và trung bình |
| **Robustness** | Tính ổn định (Mean và Std) qua nhiều lần chạy độc lập |
| **Scalability** | Hiệu suất khi thay đổi kích thước bài toán |

---

## Tác giả
**Nhóm sinh viên - Đồ án 1**
| STT | MSSV | Họ và Tên | 
|---|---|---|
| 1 | 24127089 | Hồ Thị Như Ngọc | 
| 2 | 24127194 | Hoàng Trung Kiên | 
| 3 | 24127586 | Trần Tường Vi | 
| 4 | 24127595 | Lê Thị Như Ý |

**Môn học:** CSC14003 - Cơ sở Trí tuệ Nhân tạo
**Khoa:** Công nghệ Thông tin - ĐHKHTN TPHCM
**Năm học:** 2025-2026

--- 