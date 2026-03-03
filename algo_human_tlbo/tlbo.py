import numpy as np
import time

class TLBO:
    def __init__(self, problem, pop_size=50, max_iter=100):
        self.problem = problem
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = problem.dim
        self.lb = problem.lb
        self.ub = problem.ub
        self.history_best = []
        self.history_diversity = []
        self.best_fitness = float('inf')

    def run(self):
        start_time = time.time()
        
        population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self.problem.func(ind) for ind in population])
        self.best_fitness = np.min(fitness)
        self.best_solution = population[np.argmin(fitness)].copy()

        for it in range(self.max_iter):
            mean_pop = np.mean(population, axis=0)
            teacher = population[np.argmin(fitness)]
            new_pop = population + np.random.rand(self.pop_size, self.dim) * (teacher - np.random.randint(1, 3) * mean_pop)
            new_pop = np.clip(new_pop, self.lb, self.ub)
            
            for i in range(self.pop_size):
                if self.problem.func(new_pop[i]) < fitness[i]:
                    population[i] = new_pop[i]
                    fitness[i] = self.problem.func(new_pop[i])

            for i in range(self.pop_size):
                idxs = [x for x in range(self.pop_size) if x != i]
                j = np.random.choice(idxs)
                
                if fitness[i] < fitness[j]:
                    step = population[i] - population[j]
                else:
                    step = population[j] - population[i]
                
                new_sol = np.clip(population[i] + np.random.rand(self.dim) * step, self.lb, self.ub)
                if self.problem.func(new_sol) < fitness[i]:
                    population[i] = new_sol
                    fitness[i] = self.problem.func(new_sol)

            current_best = np.min(fitness)
            if current_best < self.best_fitness:
                self.best_fitness = current_best
                self.best_solution = population[np.argmin(fitness)].copy()
            self.history_best.append(self.best_fitness)
            
            center = np.mean(population, axis=0)
            diversity = np.mean(np.sqrt(np.sum((population - center)**2, axis=1)))
            self.history_diversity.append(diversity)

        return {
            "best_fitness": self.best_fitness,
            "convergence": np.array(self.history_best),
            "diversity": np.array(self.history_diversity),
            "runtime": time.time() - start_time
        }