# optimization_comparison/visualization.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from typing import List
from .types import Vector, Bounds, Objective


def create_animation(
    func: Objective,
    func_name: str,
    bounds: Bounds,
    thr_path: List[Vector],
    sa_path: List[Vector],
    pso_path: List[Vector],
    swarm_history: List[List[Vector]],
    filename: str = "animation.gif",
    max_frames: int = 200,
    fps: int = 10,
):
    """
    Создает GIF-анимацию траекторий алгоритмов.
    
    Args:
        func: Целевая функция
        func_name: Название функции
        bounds: Границы поиска
        thr_path: Путь порогового алгоритма
        sa_path: Путь имитации отжига
        pso_path: Путь PSO (глобальный лидер)
        swarm_history: История роя PSO
        filename: Имя выходного файла
        max_frames: Максимальное количество кадров
        fps: Кадров в секунду
    """
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    # Поверхность функции
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
    line_thr, = ax.plot([], [], [], "r-", label="Threshold", linewidth=1.5)
    point_thr, = ax.plot([], [], [], "ro", markersize=6)

    line_sa, = ax.plot([], [], [], "b-", label="SA", linewidth=1.5)
    point_sa, = ax.plot([], [], [], "bo", markersize=6)

    line_pso, = ax.plot([], [], [], "g-", label="PSO best", linewidth=1.5)
    point_pso, = ax.plot([], [], [], "go", markersize=6)

    # Весь рой
    swarm_scatter = ax.scatter([], [], [], c="k", marker=".", alpha=0.7, 
                               label="PSO swarm", s=20)

    ax.legend(loc='upper left')
    text_iter = ax.text2D(0.05, 0.95, "", transform=ax.transAxes,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    len_thr = len(thr_path)
    len_sa = len(sa_path)
    len_pso = len(pso_path)
    len_swarm = len(swarm_history)
    len_max = max(len_thr, len_sa, len_pso, len_swarm)

    n_frames = min(len_max, max_frames)

    def sample_idx(frame: int, path_len: int) -> int:
        if path_len <= 1:
            return 0
        if n_frames <= 1:
            return path_len - 1
        return int(frame * (path_len - 1) / (n_frames - 1))

    def init():
        for artist in (line_thr, point_thr, line_sa, point_sa, line_pso, point_pso):
            artist.set_data([], [])
            artist.set_3d_properties([])
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

        # PSO
        k_pso = sample_idx(frame, len_pso)
        xs_pso = [p[0] for p in pso_path[: k_pso + 1]]
        ys_pso = [p[1] for p in pso_path[: k_pso + 1]]
        zs_pso = pso_z_all[: k_pso + 1]
        line_pso.set_data(xs_pso, ys_pso)
        line_pso.set_3d_properties(zs_pso)
        point_pso.set_data([xs_pso[-1]], [ys_pso[-1]])
        point_pso.set_3d_properties([zs_pso[-1]])

        # Рой PSO
        k_swarm = sample_idx(frame, len_swarm)
        swarm = swarm_history[k_swarm]
        swarm_x = [p[0] for p in swarm]
        swarm_y = [p[1] for p in swarm]
        swarm_z = [func(p) for p in swarm]
        swarm_scatter._offsets3d = (swarm_x, swarm_y, swarm_z)

        text_iter.set_text(f"Кадр: {frame + 1}/{n_frames}")
        return line_thr, point_thr, line_sa, point_sa, line_pso, point_pso, swarm_scatter, text_iter

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=1000//fps,  # мс между кадрами
        blit=False,
    )

    print(f"  Сохраняю GIF: {filename} ({n_frames} кадров, {fps} FPS)")
    anim.save(filename, writer="pillow", fps=fps, dpi=100)
    plt.close(fig)


def plot_comparison(
    func: Objective,
    func_name: str,
    bounds: Bounds,
    hist_thr: List[float],
    hist_sa: List[float],
    hist_pso: List[float],
    thr_path: List[Vector],
    sa_path: List[Vector],
    pso_path: List[Vector],
    swarm_history: List[List[Vector]],
):
    """
    Создает график сравнения алгоритмов.
    
    Args:
        func: Целевая функция
        func_name: Название функции
        bounds: Границы поиска
        hist_thr: История значений порогового алгоритма
        hist_sa: История значений имитации отжига
        hist_pso: История значений PSO
        thr_path: Путь порогового алгоритма
        sa_path: Путь имитации отжига
        pso_path: Путь PSO (глобальный лидер)
        swarm_history: История роя PSO
    
    Returns:
        matplotlib.figure.Figure: Созданная фигура
    """
    fig = plt.figure(figsize=(10, 8))

    # ----- ВЕРХ: эпохи -----
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(hist_thr, label="Threshold Accepting", color='red', linewidth=2)
    ax1.plot(hist_sa, label="Simulated Annealing", color='blue', linewidth=2)
    ax1.plot(hist_pso, label="Particle Swarm (gbest)", color='green', linewidth=2)
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Лучшее значение f(x)")
    ax1.set_title(f"Сходимость алгоритмов ({func_name})")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Добавляем минимумы на график
    min_val = min(min(hist_thr), min(hist_sa), min(hist_pso))
    ax1.axhline(y=min_val, color='gray', linestyle='--', alpha=0.5, linewidth=1)

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

    ax2.plot_surface(X, Y, Z, alpha=0.7, linewidth=0, antialiased=True, cmap='viridis')

    # траектория порогового алгоритма
    thr_x = [p[0] for p in thr_path]
    thr_y = [p[1] for p in thr_path]
    thr_z = [func(p) for p in thr_path]
    ax2.plot(thr_x, thr_y, thr_z, color="red", label="Threshold", linewidth=2)
    ax2.scatter(thr_x[0], thr_y[0], thr_z[0], color="red", marker="o", s=100, edgecolor='black', linewidth=1.5)    # старт
    ax2.scatter(thr_x[-1], thr_y[-1], thr_z[-1], color="red", marker="^", s=150, edgecolor='black', linewidth=1.5)  # финиш

    # траектория SA
    sa_x = [p[0] for p in sa_path]
    sa_y = [p[1] for p in sa_path]
    sa_z = [func(p) for p in sa_path]
    ax2.plot(sa_x, sa_y, sa_z, color="blue", label="SA", linewidth=2)
    ax2.scatter(sa_x[0], sa_y[0], sa_z[0], color="blue", marker="o", s=100, edgecolor='black', linewidth=1.5)
    ax2.scatter(sa_x[-1], sa_y[-1], sa_z[-1], color="blue", marker="^", s=150, edgecolor='black', linewidth=1.5)

    # траектория PSO (глобальный лидер по эпохам)
    pso_x = [p[0] for p in pso_path]
    pso_y = [p[1] for p in pso_path]
    pso_z = [func(p) for p in pso_path]
    ax2.plot(pso_x, pso_y, pso_z, color="green", label="PSO best", linewidth=2)
    ax2.scatter(pso_x[0], pso_y[0], pso_z[0], color="green", marker="o", s=100, edgecolor='black', linewidth=1.5)
    ax2.scatter(pso_x[-1], pso_y[-1], pso_z[-1], color="green", marker="^", s=150, edgecolor='black', linewidth=1.5)

    # Дополнительно: рой в финальной итерации
    swarm_last = swarm_history[-1]
    swarm_x = [p[0] for p in swarm_last]
    swarm_y = [p[1] for p in swarm_last]
    swarm_z = [func(p) for p in swarm_last]
    ax2.scatter(swarm_x, swarm_y, swarm_z, color="black", marker=".", 
                alpha=0.5, s=50, label="PSO swarm (last)")

    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("f(x)")
    ax2.set_title(func_name)
    ax2.legend(loc='upper right')

    fig.tight_layout()
    return fig


def create_static_plot(
    func: Objective,
    func_name: str,
    bounds: Bounds,
    thr_path: List[Vector],
    sa_path: List[Vector],
    pso_path: List[Vector],
    swarm_history: List[List[Vector]],
    filename: str = "static_plot.png",
):
    """
    Создает статический график траекторий (без анимации).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    # Поверхность функции
    n_grid = 50
    xs = np.linspace(x_min, x_max, n_grid)
    ys = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros_like(X)
    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = func([float(X[i, j]), float(Y[i, j])])

    ax.plot_surface(X, Y, Z, alpha=0.6, linewidth=0, antialiased=True, cmap='plasma')

    # траектория порогового алгоритма
    thr_x = [p[0] for p in thr_path]
    thr_y = [p[1] for p in thr_path]
    thr_z = [func(p) for p in thr_path]
    ax.plot(thr_x, thr_y, thr_z, color="red", label="Threshold", linewidth=2)
    ax.scatter(thr_x[0], thr_y[0], thr_z[0], color="red", marker="o", s=100, edgecolor='black')
    ax.scatter(thr_x[-1], thr_y[-1], thr_z[-1], color="red", marker="^", s=150, edgecolor='black')

    # траектория SA
    sa_x = [p[0] for p in sa_path]
    sa_y = [p[1] for p in sa_path]
    sa_z = [func(p) for p in sa_path]
    ax.plot(sa_x, sa_y, sa_z, color="blue", label="SA", linewidth=2)
    ax.scatter(sa_x[0], sa_y[0], sa_z[0], color="blue", marker="o", s=100, edgecolor='black')
    ax.scatter(sa_x[-1], sa_y[-1], sa_z[-1], color="blue", marker="^", s=150, edgecolor='black')

    # траектория PSO
    pso_x = [p[0] for p in pso_path]
    pso_y = [p[1] for p in pso_path]
    pso_z = [func(p) for p in pso_path]
    ax.plot(pso_x, pso_y, pso_z, color="green", label="PSO best", linewidth=2)
    ax.scatter(pso_x[0], pso_y[0], pso_z[0], color="green", marker="o", s=100, edgecolor='black')
    ax.scatter(pso_x[-1], pso_y[-1], pso_z[-1], color="green", marker="^", s=150, edgecolor='black')

    # Рой PSO в финальной итерации
    swarm_last = swarm_history[-1]
    swarm_x = [p[0] for p in swarm_last]
    swarm_y = [p[1] for p in swarm_last]
    swarm_z = [func(p) for p in swarm_last]
    ax.scatter(swarm_x, swarm_y, swarm_z, color="black", marker=".", 
               alpha=0.5, s=30, label="PSO swarm (final)")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")
    ax.set_title(f"{func_name} - Траектории алгоритмов")
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Статический график сохранен: {filename}")