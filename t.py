import math
import random
from typing import List, Tuple, Callable

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from matplotlib import animation

# ---------- Типы ----------
Vector = List[float]
Bounds = List[Tuple[float, float]]
Objective = Callable[[Vector], float]


# ---------- Целевые функции ----------
def paraboloid(x: Vector) -> float:
    """
    Параболоид: f(x) = x1^2 + x2^2 + ... + xn^2
    Глобальный минимум в (0,...,0), f = 0.
    """
    return sum(xi * xi for xi in x)


def rastrigin(x: Vector) -> float:
    """
    Функция Растригина. Многомодальная, с множеством локальных минимумов.
    Глобальный минимум в (0,...,0), f = 0.
    """
    A = 10.0
    n = len(x)
    return A * n + sum(xi * xi - A * math.cos(2 * math.pi * xi) for xi in x)


def schwefel(x: Vector) -> float:
    """
    Функция Швеффеля (вариант 2.26).
    Обычно область поиска: [-500, 500]^n,
    глобальный минимум около (420.9687, ..., 420.9687).
    """
    n = len(x)
    return 418.982887 * n - sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x)


# ---------- Генерация соседа ----------
def random_neighbor(x: Vector, step: float, bounds: Bounds) -> Vector:
    """
    Сосед: добавляем гауссовский шум и отражаем/обрезаем по границам.
    """
    new: Vector = []
    for xi, (low, high) in zip(x, bounds):
        y = xi + random.gauss(0.0, step)

        # отражение от границ
        if y < low:
            y = low + (low - y)
        if y > high:
            y = high - (y - high)

        # обрезка
        y = max(low, min(high, y))
        new.append(y)

    return new


# ---------- Пороговый алгоритм ----------
def threshold_algorithm(
    func: Objective,
    bounds: Bounds,
    x0: Vector | None = None,
    eps0: float = 1.0,
    alpha: float = 0.95,
    n_outer: int = 50,
    n_inner: int = 20,
    step0: float = 1.0,
):
    """
    Пороговый алгоритм глобальной оптимизации (Threshold Accepting).

    Возвращает:
      best          – лучшее найденное решение
      best_val      – значение функции в лучшей точке
      history_outer – лучший минимум после каждой внешней итерации
      path          – траектория (состояние после КАЖДОЙ внутренней итерации)
    """
    if x0 is None:
        x0 = [random.uniform(low, high) for (low, high) in bounds]

    current = x0[:]
    current_val = func(current)

    best = current[:]
    best_val = current_val

    eps = eps0
    step = step0

    history_outer: list[float] = []
    path: list[Vector] = [current[:]]

    for _ in range(n_outer):
        for _ in range(n_inner):
            candidate = random_neighbor(current, step, bounds)
            candidate_val = func(candidate)
            delta = candidate_val - current_val

            # допускаем ухудшение до eps
            if delta <= eps:
                current, current_val = candidate, candidate_val
                if current_val < best_val:
                    best, best_val = current[:], current_val

            # фиксируем состояние ПОСЛЕ итерации
            path.append(current[:])

        history_outer.append(best_val)
        eps *= alpha
        step *= alpha ** 0.5

    return best, best_val, history_outer, path


# ---------- Имитация отжига ----------
def simulated_annealing(
    func: Objective,
    bounds: Bounds,
    x0: Vector | None = None,
    T0: float = 1.0,
    alpha: float = 0.95,
    n_outer: int = 50,
    n_inner: int = 20,
    step0: float = 1.0,
):
    """
    Алгоритм имитации отжига (Simulated Annealing, SA).

    path содержит состояние после каждой внутренней итерации
    → визуализация строго по итерациям.
    """
    if x0 is None:
        x0 = [random.uniform(low, high) for (low, high) in bounds]

    current = x0[:]
    current_val = func(current)

    best = current[:]
    best_val = current_val

    T = T0
    step = step0

    history_outer: list[float] = []
    path: list[Vector] = [current[:]]

    for _ in range(n_outer):
        for _ in range(n_inner):
            candidate = random_neighbor(current, step, bounds)
            candidate_val = func(candidate)
            delta = candidate_val - current_val

            # критерий Метрополиса
            if delta <= 0 or random.random() < math.exp(-delta / T):
                current, current_val = candidate, candidate_val
                if current_val < best_val:
                    best, best_val = current[:], current_val

            path.append(current[:])

        history_outer.append(best_val)
        T *= alpha
        step *= alpha ** 0.5

    return best, best_val, history_outer, path


# ---------- Рой частиц (PSO) ----------
def particle_swarm_optimization(
    func: Objective,
    bounds: Bounds,
    n_particles: int = 30,
    n_iters: int = 50,
    w: float = 0.5,
    c1: float = 1.5,
    c2: float = 1.5,
):
    """
    Метод роя частиц (Particle Swarm Optimization, PSO).

    Возвращает:
      gbest_pos      – лучшее найденное решение (глобальный лидер роя)
      gbest_val      – значение функции в лучшей точке
      history        – лучшее значение после каждой итерации (эпохи)
      path           – траектория глобального лидера (gbest) по эпохам
      swarm_history  – список положений всех частиц на каждой итерации
    """
    dim = len(bounds)

    # Инициализация частиц
    particles: list[Vector] = []
    velocities: list[Vector] = []
    pbest_pos: list[Vector] = []
    pbest_val: list[float] = []

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

    history: list[float] = [gbest_val]
    path: list[Vector] = [gbest_pos[:]]

    # история всего роя
    swarm_history: list[list[Vector]] = [[p[:] for p in particles]]

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
                x_d = max(low, min(high, x_d))  # обрезка по границам
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


# ---------- Анимация 3D-траекторий ----------
def create_animation(
    func: Objective,
    func_name: str,
    bounds: Bounds,
    thr_path: list[Vector],
    sa_path: list[Vector],
    pso_path: list[Vector],
    swarm_history: list[list[Vector]],
    filename: str = "animation.gif",
    max_frames: int = 200,      # максимум кадров в GIF
):
    """
    Строит 3D-поверхность и анимирует траектории TA, SA и PSO по итерациям.
    Показывает:
      - путь TA (красная линия + точка),
      - путь SA (синяя линия + точка),
      - путь глобального лидера PSO (зелёная линия + точка),
      - весь рой PSO (чёрные точки).
    Сохраняет результат в GIF.
    """
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    # Поверхность функции (чуть более грубая, чтобы ускорить)
    n_grid = 30
    xs = np.linspace(x_min, x_max, n_grid)
    ys = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros_like(X)
    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = func([float(X[i, j]), float(Y[i, j])])

    ax.plot_surface(X, Y, Z, alpha=0.7, linewidth=0, antialiased=True)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")
    ax.set_title(func_name)

    # Предвычисляем значения f вдоль траекторий
    thr_z_all = [func(p) for p in thr_path]
    sa_z_all = [func(p) for p in sa_path]
    pso_z_all = [func(p) for p in pso_path]

    # Линии и текущие точки
    line_thr, = ax.plot([], [], [], "r-", label="Threshold")
    point_thr, = ax.plot([], [], [], "ro", markersize=4)

    line_sa, = ax.plot([], [], [], "b-", label="SA")
    point_sa, = ax.plot([], [], [], "bo", markersize=4)

    line_pso, = ax.plot([], [], [], "g-", label="PSO best")
    point_pso, = ax.plot([], [], [], "go", markersize=4)

    # Весь рой: scatter, координаты задаём через _offsets3d
    swarm_scatter = ax.scatter([], [], [], c="k", marker=".", alpha=0.5, label="PSO swarm")

    ax.legend()
    text_iter = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    len_thr = len(thr_path)
    len_sa = len(sa_path)
    len_pso = len(pso_path)
    len_swarm = len(swarm_history)
    len_max = max(len_thr, len_sa, len_pso, len_swarm)

    # Сколько кадров реально рисуем
    n_frames = min(len_max, max_frames)

    # функция, которая по номеру кадра возвращает индекс в пути
    def sample_idx(frame: int, path_len: int) -> int:
        if path_len <= 1:
            return 0
        if n_frames <= 1:
            return path_len - 1
        # равномерное отображение диапазона [0, n_frames-1] -> [0, path_len-1]
        return int(frame * (path_len - 1) / (n_frames - 1))

    def init():
        for artist in (line_thr, point_thr, line_sa, point_sa, line_pso, point_pso):
            artist.set_data([], [])
            artist.set_3d_properties([])
        # пустой рой
        swarm_scatter._offsets3d = ([], [], [])
        text_iter.set_text("")
        return line_thr, point_thr, line_sa, point_sa, line_pso, point_pso, swarm_scatter, text_iter

    def update(frame):
        # TA
        k_thr = sample_idx(frame, len_thr)
        xs_thr = [p[0] for p in thr_path[: k_thr + 1]]
        ys_thr = [p[1] for p in thr_path[: k_thr + 1]]
        zs_thr = thr_z_all[: k_thr + 1]
        line_thr.set_data(xs_thr, ys_thr)
        line_thr.set_3d_properties(zs_thr)
        point_thr.set_data([xs_thr[-1]], [ys_thr[-1]])
        point_thr.set_3d_properties([zs_thr[-1]])

        # SA
        k_sa = sample_idx(frame, len_sa)
        xs_sa = [p[0] for p in sa_path[: k_sa + 1]]
        ys_sa = [p[1] for p in sa_path[: k_sa + 1]]
        zs_sa = sa_z_all[: k_sa + 1]
        line_sa.set_data(xs_sa, ys_sa)
        line_sa.set_3d_properties(zs_sa)
        point_sa.set_data([xs_sa[-1]], [ys_sa[-1]])
        point_sa.set_3d_properties([zs_sa[-1]])

        # PSO (gbest по эпохам)
        k_pso = sample_idx(frame, len_pso)
        xs_pso = [p[0] for p in pso_path[: k_pso + 1]]
        ys_pso = [p[1] for p in pso_path[: k_pso + 1]]
        zs_pso = pso_z_all[: k_pso + 1]
        line_pso.set_data(xs_pso, ys_pso)
        line_pso.set_3d_properties(zs_pso)
        point_pso.set_data([xs_pso[-1]], [ys_pso[-1]])
        point_pso.set_3d_properties([zs_pso[-1]])

        # Рой PSO на этой итерации
        k_swarm = sample_idx(frame, len_swarm)
        swarm = swarm_history[k_swarm]
        swarm_x = [p[0] for p in swarm]
        swarm_y = [p[1] for p in swarm]
        swarm_z = [func(p) for p in swarm]
        swarm_scatter._offsets3d = (swarm_x, swarm_y, swarm_z)

        text_iter.set_text(f"Итерация (семпл): {frame + 1}/{n_frames}")
        return line_thr, point_thr, line_sa, point_sa, line_pso, point_pso, swarm_scatter, text_iter

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=100,  # мс между кадрами
        blit=False,
    )

    print(f"Сохраняю GIF ({n_frames} кадров): {filename}")
    anim.save(filename, writer="pillow", fps=10)
    plt.close(fig)


# ---------- Один рисунок = эпохи сверху + 3D снизу ----------
def run_for_function(func: Objective, func_name: str, bounds: Bounds, gif_filename: str | None = None):
    """
    Запускает три алгоритма на одной функции и рисует одну фигуру:
    - сверху: 2D-график сходимости (эпохи = внешние итерации / итерации PSO),
    - снизу: 3D-поверхность + траектории.
    Дополнительно (если задан gif_filename) создаёт GIF-анимацию траекторий.
    """
    random.seed(0)

    # параметры для SA/TA
    n_outer = 50
    n_inner = 20

    # одинаковая стартовая точка для честного сравнения SA и TA
    x0 = [random.uniform(low, high) for (low, high) in bounds]

    best_thr, val_thr, hist_thr, thr_path = threshold_algorithm(
        func,
        bounds,
        x0=x0,
        eps0=1.0,
        alpha=0.95,
        n_outer=n_outer,
        n_inner=n_inner,
        step0=1.0,
    )

    best_sa, val_sa, hist_sa, sa_path = simulated_annealing(
        func,
        bounds,
        x0=x0,
        T0=1.0,
        alpha=0.95,
        n_outer=n_outer,
        n_inner=n_inner,
        step0=1.0,
    )

    # PSO: подберём число итераций так, чтобы примерно сопоставимо было число оценок
    n_particles = 30
    n_iters = max(1, (n_outer * n_inner) // n_particles)

    best_pso, val_pso, hist_pso, pso_path, swarm_history = particle_swarm_optimization(
        func,
        bounds,
        n_particles=n_particles,
        n_iters=n_iters,
        w=0.5,
        c1=1.5,
        c2=1.5,
    )

    print(f"=== {func_name} ===")
    print("Начальная точка (для SA/TA):", x0)
    print("Пороговый алгоритм: x* =", best_thr, "f(x*) =", val_thr)
    print("Имитация отжига:     x* =", best_sa, "f(x*) =", val_sa)
    print("Рой частиц (gbest):  x* =", best_pso, "f(x*) =", val_pso)
    print()

    # ---- создаём фигуру с двумя подграфиками ----
    fig = plt.figure(figsize=(10, 8))

    # ----- ВЕРХ: эпохи (2D-график сходимости) -----
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(hist_thr, label="Threshold Accepting")
    ax1.plot(hist_sa, label="Simulated Annealing")
    ax1.plot(hist_pso, label="Particle Swarm (gbest)")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Лучшее значение f(x)")
    ax1.set_title(f"Сходимость алгоритмов ({func_name})")
    ax1.grid(True)
    ax1.legend()

    # ----- НИЗ: 3D-поверхность + траектории -----
    ax2 = fig.add_subplot(2, 1, 2, projection="3d")

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    n_grid = 60
    xs = np.linspace(x_min, x_max, n_grid)
    ys = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros_like(X)
    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = func([float(X[i, j]), float(Y[i, j])])

    ax2.plot_surface(X, Y, Z, alpha=0.7, linewidth=0, antialiased=True)

    # траектория порогового алгоритма
    thr_x = [p[0] for p in thr_path]
    thr_y = [p[1] for p in thr_path]
    thr_z = [func(p) for p in thr_path]
    ax2.plot(thr_x, thr_y, thr_z, color="red", label="Threshold")
    ax2.scatter(thr_x[0], thr_y[0], thr_z[0], color="red", marker="o")    # старт
    ax2.scatter(thr_x[-1], thr_y[-1], thr_z[-1], color="red", marker="^")  # финиш

    # траектория SA
    sa_x = [p[0] for p in sa_path]
    sa_y = [p[1] for p in sa_path]
    sa_z = [func(p) for p in sa_path]
    ax2.plot(sa_x, sa_y, sa_z, color="blue", label="SA")
    ax2.scatter(sa_x[0], sa_y[0], sa_z[0], color="blue", marker="o")
    ax2.scatter(sa_x[-1], sa_y[-1], sa_z[-1], color="blue", marker="^")

    # траектория PSO (глобальный лидер по эпохам)
    pso_x = [p[0] for p in pso_path]
    pso_y = [p[1] for p in pso_path]
    pso_z = [func(p) for p in pso_path]
    ax2.plot(pso_x, pso_y, pso_z, color="green", label="PSO best")
    ax2.scatter(pso_x[0], pso_y[0], pso_z[0], color="green", marker="o")
    ax2.scatter(pso_x[-1], pso_y[-1], pso_z[-1], color="green", marker="^")

    # Дополнительно: рой в финальной итерации
    swarm_last = swarm_history[-1]
    swarm_x = [p[0] for p in swarm_last]
    swarm_y = [p[1] for p in swarm_last]
    swarm_z = [func(p) for p in swarm_last]
    ax2.scatter(swarm_x, swarm_y, swarm_z, color="black", marker=".", alpha=0.5, label="PSO swarm (last)")

    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("f(x)")
    ax2.set_title(func_name)
    ax2.legend()

    fig.tight_layout()

    # --- GIF-анимация по итерациям ---
    if gif_filename is not None:
        print(f"Сохраняю GIF-анимацию: {gif_filename}")
        create_animation(func, func_name, bounds, thr_path, sa_path, pso_path, swarm_history, gif_filename)


# ---------- Главная функция ----------
def main():
    # 1) Параболоид
    bounds_paraboloid: Bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    run_for_function(
        paraboloid,
        "Параболоид f(x)=x1^2+x2^2",
        bounds_paraboloid,
        gif_filename="paraboloid.gif",
    )

    # 2) Растригин
    bounds_rastrigin: Bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    run_for_function(
        rastrigin,
        "Функция Растригина",
        bounds_rastrigin,
        gif_filename="rastrigin.gif",
    )

    # 3) Швеффеля
    bounds_schwefel: Bounds = [(-500.0, 500.0), (-500.0, 500.0)]
    run_for_function(
        schwefel,
        "Функция Швеффеля",
        bounds_schwefel,
        gif_filename="schwefel.gif",
    )

    # показать все созданные статические графики
    plt.show()


if __name__ == "__main__":
    main()