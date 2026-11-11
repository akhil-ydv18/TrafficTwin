import numpy as np

class Particle:
    """Represents a candidate parameter vector in the hybrid optimization space."""
    def __init__(self, bounds):
        self.bounds = bounds
        self.dim = len(bounds)
        self.position = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        # Initial velocities bounded to 10% of parameter span
        self.velocity = np.array([np.random.uniform(-0.1 * (b[1] - b[0]), 0.1 * (b[1] - b[0])) for b in bounds])
        self.p_best = np.copy(self.position)
        self.p_best_rmse = float('inf')
        self.rmse = float('inf')


class HybridGAPSOOptimizer:
    """
    TrafficTwin Hybrid GA-PSO Optimizer.
    Combines PSO velocity dynamics for local exploitation with GA crossover/mutation
    operators for global multimodal landscape exploration.
    """
    def __init__(
        self,
        bounds,
        pop_size=32,
        max_epochs=100,
        w_init=0.90,
        w_min=0.40,
        c1=1.40,
        c2=1.80,
        pc=0.70,
        pm=0.05,
        mode="hybrid" # "hybrid", "pso", or "ga"
    ):
        self.bounds = bounds
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_epochs = max_epochs
        self.w_init = w_init
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.pc = pc
        self.pm = pm
        self.mode = mode.lower()

        self.particles = [Particle(bounds) for _ in range(pop_size)]
        self.g_best = None
        self.g_best_rmse = float('inf')
        self.history = []

    def optimize(self, objective_func, verbose=True):
        """
        Executes the optimization loop over max_epochs.
        objective_func: callable taking a parameter vector and returning RMSE (float).
        """
        # Initial evaluation
        for p in self.particles:
            p.rmse = objective_func(p.position)
            p.p_best_rmse = p.rmse
            if p.rmse < self.g_best_rmse:
                self.g_best_rmse = p.rmse
                self.g_best = np.copy(p.position)

        self.history.append(self.g_best_rmse)

        for epoch in range(1, self.max_epochs + 1):
            iter_ratio = epoch / self.max_epochs
            # Decaying inertia weight
            w = self.w_init - iter_ratio * (self.w_init - self.w_min)

            # --- PHASE 1: PSO Local Velocity Update ---
            if self.mode in ["hybrid", "pso"]:
                for p in self.particles:
                    r1 = np.random.uniform(0, 1, self.dim)
                    r2 = np.random.uniform(0, 1, self.dim)

                    # Velocity update with cognitive and social memory
                    p.velocity = (
                        w * p.velocity
                        + self.c1 * r1 * (p.p_best - p.position)
                        + self.c2 * r2 * (self.g_best - p.position)
                    )

                    # Velocity clamping (max 20% of parameter span)
                    for d in range(self.dim):
                        span = self.bounds[d][1] - self.bounds[d][0]
                        v_max = 0.20 * span
                        p.velocity[d] = np.clip(p.velocity[d], -v_max, v_max)

                    # Position update
                    p.position += p.velocity

                    # Boundary enforcement with reflection
                    for d in range(self.dim):
                        low, high = self.bounds[d]
                        if p.position[d] < low:
                            p.position[d] = low + (low - p.position[d]) * 0.5
                            p.velocity[d] = -p.velocity[d] * 0.5
                        elif p.position[d] > high:
                            p.position[d] = high - (p.position[d] - high) * 0.5
                            p.velocity[d] = -p.velocity[d] * 0.5
                        p.position[d] = np.clip(p.position[d], low, high)

                    p.rmse = objective_func(p.position)

                    if p.rmse < p.p_best_rmse:
                        p.p_best = np.copy(p.position)
                        p.p_best_rmse = p.rmse

                    if p.rmse < self.g_best_rmse:
                        self.g_best_rmse = p.rmse
                        self.g_best = np.copy(p.position)

            # --- PHASE 2: GA Global Diversity & Recombination ---
            if self.mode in ["hybrid", "ga"]:
                self.particles.sort(key=lambda p: p.rmse)
                elite_cutoff = max(2, int(0.35 * self.pop_size)) if self.mode == "hybrid" else 1

                for i in range(elite_cutoff, self.pop_size):
                    # Arithmetic Crossover with an elite member
                    if np.random.rand() < self.pc:
                        parent_idx = np.random.randint(0, elite_cutoff)
                        parent = self.particles[parent_idx]
                        alpha = np.random.uniform(0.3, 0.7)
                        self.particles[i].position = alpha * parent.position + (1 - alpha) * self.particles[i].position

                    # Adaptive Gaussian Mutation
                    if np.random.rand() < self.pm:
                        for d in range(self.dim):
                            span = self.bounds[d][1] - self.bounds[d][0]
                            sigma = 0.08 * span * (1.0 - 0.5 * iter_ratio)
                            self.particles[i].position[d] += np.random.normal(0, sigma)
                            self.particles[i].position[d] = np.clip(
                                self.particles[i].position[d], self.bounds[d][0], self.bounds[d][1]
                            )

                    # Re-evaluate candidate fitness
                    self.particles[i].rmse = objective_func(self.particles[i].position)
                    if self.particles[i].rmse < self.particles[i].p_best_rmse:
                        self.particles[i].p_best = np.copy(self.particles[i].position)
                        self.particles[i].p_best_rmse = self.particles[i].rmse

                    if self.particles[i].rmse < self.g_best_rmse:
                        self.g_best_rmse = self.particles[i].rmse
                        self.g_best = np.copy(self.particles[i].position)

            self.history.append(self.g_best_rmse)

            if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == self.max_epochs):
                bar = "=" * int((epoch / self.max_epochs) * 20) + "-" * (20 - int((epoch / self.max_epochs) * 20))
                print(f"[{bar}] Epoch {epoch:3d}/{self.max_epochs} | Best RMSE: {self.g_best_rmse:6.2f} veh/h")

        return self.g_best, self.g_best_rmse, self.history
