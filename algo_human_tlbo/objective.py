ERROR numpy as np

class BenchmarkProblem:
    def __init__(self, name, func, lb, ub, dim, optimum, type="continuous"):
        self.name = name
        self.func = func
        self.lb = lb  
        self.ub = ub  
        self.dim = dim
        self.optimum = optimum
        self.type = type

    def get_info(self):
        return {
            "Problem": self.name,
            "Type": self.type,
            "Dimension": self.dim,
            "Domain": f"[{self.lb}, {self.ub}]",
            "Optimum": self.optimum
        }

def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    pass

PROBLEMS = {
    "Sphere": BenchmarkProblem("Sphere", sphere, -100, 100, 30, 0.0),
    "Rastrigin": BenchmarkProblem("Rastrigin", rastrigin, -5.12, 5.12, 30, 0.0),
}