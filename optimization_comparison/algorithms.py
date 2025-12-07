# optimization_comparison/algorithms.py
import math
import random
from typing import Tuple, List
from .types import Vector, Bounds, Objective, random_neighbor


def threshold_algorithm(
    func: Objective,
    bounds: Bounds,
    x0: Vector | None = None,
    eps0: float = 1.0,
    alpha: float = 0.95,
    n_outer: int = 50,
    n_inner: int = 20,
    step0: float = 1.0,
) -> Tuple[Vector, float, List[float], List[Vector]]:
    """
    Пороговый алгоритм глобальной оптимизации (Threshold Accepting).
    """
    if x0 is None:
        x0 = [random.uniform(low, high) for (low, high) in bounds]

    current = x0[:]
    current_val = func(current)
    best = current[:]
    best_val = current_val
    eps = eps0
    step = step0

    history_outer: List[float] = []
    path: List[Vector] = [current[:]]

    for _ in range(n_outer):
        for _ in range(n_inner):
            candidate = random_neighbor(current, step, bounds)
            candidate_val = func(candidate)
            delta = candidate_val - current_val

            if delta <= eps:
                current, current_val = candidate, candidate_val
                if current_val < best_val:
                    best, best_val = current[:], current_val

            path.append(current[:])

        history_outer.append(best_val)
        eps *= alpha
        step *= alpha ** 0.5

    return best, best_val, history_outer, path


def simulated_annealing(
    func: Objective,
    bounds: Bounds,
    x0: Vector | None = None,
    T0: float = 1.0,
    alpha: float = 0.95,
    n_outer: int = 50,
    n_inner: int = 20,
    step0: float = 1.0,
) -> Tuple[Vector, float, List[float], List[Vector]]:
    """
    Алгоритм имитации отжига (Simulated Annealing, SA).
    """
    if x0 is None:
        x0 = [random.uniform(low, high) for (low, high) in bounds]

    current = x0[:]
    current_val = func(current)
    best = current[:]
    best_val = current_val
    T = T0
    step = step0

    history_outer: List[float] = []
    path: List[Vector] = [current[:]]

    for _ in range(n_outer):
        for _ in range(n_inner):
            candidate = random_neighbor(current, step, bounds)
            candidate_val = func(candidate)
            delta = candidate_val - current_val

            if delta <= 0 or random.random() < math.exp(-delta / T):
                current, current_val = candidate, candidate_val
                if current_val < best_val:
                    best, best_val = current[:], current_val

            path.append(current[:])

        history_outer.append(best_val)
        T *= alpha
        step *= alpha ** 0.5

    return best, best_val, history_outer, path


def particle_swarm_optimization(
    func: Objective,
    bounds: Bounds,
    n_particles: int = 30,
    n_iters: int = 50,
    w: float = 0.5,
    c1: float = 1.5,
    c2: float = 1.5,
) -> Tuple[Vector, float, List[float], List[Vector], List[List[Vector]]]:
    """
    Метод роя частиц (Particle Swarm Optimization, PSO).
    """
    dim = len(bounds)

    # Инициализация частиц
    particles: List[Vector] = []
    velocities: List[Vector] = []
    pbest_pos: List[Vector] = []
    pbest_val: List[float] = []

    for _ in range(n_particles):
        pos = [random.uniform(low, high) for (low, high) in bounds]
        vel = [0.0 for _ in range(dim)]
        particles.append(pos)
        velocities.append(vel)
        val = func(pos)
        pbest_pos.append(pos[:])
        pbest_val.append(val)

    # глобальный лидер
    gbest_index = min(range(n_particles), key=lambda i: pbest_val[i])
    gbest_pos = pbest_pos[gbest_index][:]
    gbest_val = pbest_val[gbest_index]

    history: List[float] = [gbest_val]
    path: List[Vector] = [gbest_pos[:]]
    swarm_history: List[List[Vector]] = [[p[:] for p in particles]]

    for _ in range(n_iters):
        for i in range(n_particles):
            # обновляем скорость
            new_vel = []
            for d in range(dim):
                r1 = random.random()
                r2 = random.random()
                inertia = w * velocities[i][d]
                cognitive = c1 * r1 * (pbest_pos[i][d] - particles[i][d])
                social = c2 * r2 * (gbest_pos[d] - particles[i][d])
                v_d = inertia + cognitive + social
                new_vel.append(v_d)
            velocities[i] = new_vel

            # обновляем позицию
            new_pos = []
            for d, (low, high) in enumerate(bounds):
                x_d = particles[i][d] + velocities[i][d]
                x_d = max(low, min(high, x_d))
                new_pos.append(x_d)
            particles[i] = new_pos

            # пересчитываем значение функции
            val = func(new_pos)

            # обновляем личного лидера
            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest_pos[i] = new_pos[:]

        # обновляем глобального лидера
        best_i = min(range(n_particles), key=lambda j: pbest_val[j])
        if pbest_val[best_i] < gbest_val:
            gbest_val = pbest_val[best_i]
            gbest_pos = pbest_pos[best_i][:]

        history.append(gbest_val)
        path.append(gbest_pos[:])
        swarm_history.append([p[:] for p in particles])

    return gbest_pos, gbest_val, history, path, swarm_history