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

    В анимации два окна:
      - слева: 3D-поверхность + траектории + рой,
      - справа: 2D-карта (contourf) + проекция траекторий и роя.
    """
    # Две оси: 3D слева, 2D справа
    fig = plt.figure(figsize=(10, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

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

    # 3D поверхность
    ax3d.plot_surface(X, Y, Z, alpha=0.7, linewidth=0, antialiased=True)
    ax3d.set_xlabel("x1")
    ax3d.set_ylabel("x2")
    ax3d.set_zlabel("f(x)")
    ax3d.set_title(func_name)

    # 2D — контурная карта
    cf = ax2d.contourf(X, Y, Z, levels=30)
    ax2d.set_xlabel("x1")
    ax2d.set_ylabel("x2")
    ax2d.set_title(f"{func_name} (вид сверху)")
    fig.colorbar(cf, ax=ax2d, shrink=0.8)

    # Предвычисляем значения f вдоль траекторий
    thr_z_all = [func(p) for p in thr_path]
    sa_z_all = [func(p) for p in sa_path]
    pso_z_all = [func(p) for p in pso_path]

    # --------- ЛИНИИ И ТОЧКИ (3D + 2D) ---------
    # TA
    line_thr_3d, = ax3d.plot([], [], [], "r-", label="Threshold", linewidth=1.5)
    point_thr_3d, = ax3d.plot([], [], [], "ro", markersize=6)
    line_thr_2d, = ax2d.plot([], [], "r-", label="Threshold")
    point_thr_2d, = ax2d.plot([], [], "ro", markersize=4)

    # SA
    line_sa_3d, = ax3d.plot([], [], [], "b-", label="SA", linewidth=1.5)
    point_sa_3d, = ax3d.plot([], [], [], "bo", markersize=6)
    line_sa_2d, = ax2d.plot([], [], "b-", label="SA")
    point_sa_2d, = ax2d.plot([], [], "bo", markersize=4)

    # PSO (gbest)
    line_pso_3d, = ax3d.plot([], [], [], "g-", label="PSO best", linewidth=1.5)
    point_pso_3d, = ax3d.plot([], [], [], "go", markersize=6)
    line_pso_2d, = ax2d.plot([], [], "g-", label="PSO best")
    point_pso_2d, = ax2d.plot([], [], "go", markersize=4)

    # --------- РОЙ (3D + 2D) ---------
    swarm_scatter_3d = ax3d.scatter(
        [], [], [], c="k", marker=".", alpha=0.7, s=20, label="PSO swarm"
    )
    swarm_scatter_2d = ax2d.scatter([], [], c="k", marker=".", alpha=0.7, s=10)

    ax3d.legend(loc='upper left')
    ax2d.legend(loc='upper right')

    text_iter = ax3d.text2D(
        0.05,
        0.95,
        "",
        transform=ax3d.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

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
        # 3D линии/точки
        for artist in (
            line_thr_3d,
            point_thr_3d,
            line_sa_3d,
            point_sa_3d,
            line_pso_3d,
            point_pso_3d,
        ):
            artist.set_data([], [])
            artist.set_3d_properties([])

        # 2D линии/точки
        for artist in (
            line_thr_2d,
            point_thr_2d,
            line_sa_2d,
            point_sa_2d,
            line_pso_2d,
            point_pso_2d,
        ):
            artist.set_data([], [])

        swarm_scatter_3d._offsets3d = ([], [], [])
        swarm_scatter_2d.set_offsets(np.empty((0, 2)))

        text_iter.set_text("")
        return (
            line_thr_3d,
            point_thr_3d,
            line_sa_3d,
            point_sa_3d,
            line_pso_3d,
            point_pso_3d,
            swarm_scatter_3d,
            line_thr_2d,
            point_thr_2d,
            line_sa_2d,
            point_sa_2d,
            line_pso_2d,
            point_pso_2d,
            swarm_scatter_2d,
            text_iter,
        )

    def update(frame):
        # ----- TA -----
        k_thr = sample_idx(frame, len_thr)
        xs_thr = [p[0] for p in thr_path[: k_thr + 1]]
        ys_thr = [p[1] for p in thr_path[: k_thr + 1]]
        zs_thr = thr_z_all[: k_thr + 1]

        line_thr_3d.set_data(xs_thr, ys_thr)
        line_thr_3d.set_3d_properties(zs_thr)
        if xs_thr:
            point_thr_3d.set_data([xs_thr[-1]], [ys_thr[-1]])
            point_thr_3d.set_3d_properties([zs_thr[-1]])

        line_thr_2d.set_data(xs_thr, ys_thr)
        if xs_thr:
            point_thr_2d.set_data([xs_thr[-1]], [ys_thr[-1]])

        # ----- SA -----
        k_sa = sample_idx(frame, len_sa)
        xs_sa = [p[0] for p in sa_path[: k_sa + 1]]
        ys_sa = [p[1] for p in sa_path[: k_sa + 1]]
        zs_sa = sa_z_all[: k_sa + 1]

        line_sa_3d.set_data(xs_sa, ys_sa)
        line_sa_3d.set_3d_properties(zs_sa)
        if xs_sa:
            point_sa_3d.set_data([xs_sa[-1]], [ys_sa[-1]])
            point_sa_3d.set_3d_properties([zs_sa[-1]])

        line_sa_2d.set_data(xs_sa, ys_sa)
        if xs_sa:
            point_sa_2d.set_data([xs_sa[-1]], [ys_sa[-1]])

        # ----- PSO (gbest) -----
        k_pso = sample_idx(frame, len_pso)
        xs_pso = [p[0] for p in pso_path[: k_pso + 1]]
        ys_pso = [p[1] for p in pso_path[: k_pso + 1]]
        zs_pso = pso_z_all[: k_pso + 1]

        line_pso_3d.set_data(xs_pso, ys_pso)
        line_pso_3d.set_3d_properties(zs_pso)
        if xs_pso:
            point_pso_3d.set_data([xs_pso[-1]], [ys_pso[-1]])
            point_pso_3d.set_3d_properties([zs_pso[-1]])

        line_pso_2d.set_data(xs_pso, ys_pso)
        if xs_pso:
            point_pso_2d.set_data([xs_pso[-1]], [ys_pso[-1]])

        # ----- Рой PSO -----
        k_swarm = sample_idx(frame, len_swarm)
        swarm = swarm_history[k_swarm]
        swarm_x = [p[0] for p in swarm]
        swarm_y = [p[1] for p in swarm]
        swarm_z = [func(p) for p in swarm]

        swarm_scatter_3d._offsets3d = (swarm_x, swarm_y, swarm_z)

        if swarm_x:
            offsets2d = np.column_stack([swarm_x, swarm_y])
        else:
            offsets2d = np.empty((0, 2))
        swarm_scatter_2d.set_offsets(offsets2d)

        text_iter.set_text(f"Кадр: {frame + 1}/{n_frames}")
        return (
            line_thr_3d,
            point_thr_3d,
            line_sa_3d,
            point_sa_3d,
            line_pso_3d,
            point_pso_3d,
            swarm_scatter_3d,
            line_thr_2d,
            point_thr_2d,
            line_sa_2d,
            point_sa_2d,
            line_pso_2d,
            point_pso_2d,
            swarm_scatter_2d,
            text_iter,
        )

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=1000 // fps,
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
    Создает статический график сравнения алгоритмов:
      - наверху: графики сходимости,
      - внизу: 3D-поверхность и траектории.
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
    ax2.scatter(thr_x[0], thr_y[0], thr_z[0],
                color="red", marker="o", s=100, edgecolor='black', linewidth=1.5)
    ax2.scatter(thr_x[-1], thr_y[-1], thr_z[-1],
                color="red", marker="^", s=150, edgecolor='black', linewidth=1.5)

    # траектория SA
    sa_x = [p[0] for p in sa_path]
    sa_y = [p[1] for p in sa_path]
    sa_z = [func(p) for p in sa_path]
    ax2.plot(sa_x, sa_y, sa_z, color="blue", label="SA", linewidth=2)
    ax2.scatter(sa_x[0], sa_y[0], sa_z[0],
                color="blue", marker="o", s=100, edgecolor='black', linewidth=1.5)
    ax2.scatter(sa_x[-1], sa_y[-1], sa_z[-1],
                color="blue", marker="^", s=150, edgecolor='black', linewidth=1.5)

    # траектория PSO (глобальный лидер по эпохам)
    pso_x = [p[0] for p in pso_path]
    pso_y = [p[1] for p in pso_path]
    pso_z = [func(p) for p in pso_path]
    ax2.plot(pso_x, pso_y, pso_z, color="green", label="PSO best", linewidth=2)
    ax2.scatter(pso_x[0], pso_y[0], pso_z[0],
                color="green", marker="o", s=100, edgecolor='black', linewidth=1.5)
    ax2.scatter(pso_x[-1], pso_y[-1], pso_z[-1],
                color="green", marker="^", s=150, edgecolor='black', linewidth=1.5)

    # рой в финальной итерации
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
    Создает статический 3D-график траекторий (без анимации).
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
    ax.scatter(thr_x[0], thr_y[0], thr_z[0],
               color="red", marker="o", s=100, edgecolor='black')
    ax.scatter(thr_x[-1], thr_y[-1], thr_z[-1],
               color="red", marker="^", s=150, edgecolor='black')

    # траектория SA
    sa_x = [p[0] for p in sa_path]
    sa_y = [p[1] for p in sa_path]
    sa_z = [func(p) for p in sa_path]
    ax.plot(sa_x, sa_y, sa_z, color="blue", label="SA", linewidth=2)
    ax.scatter(sa_x[0], sa_y[0], sa_z[0],
               color="blue", marker="o", s=100, edgecolor='black')
    ax.scatter(sa_x[-1], sa_y[-1], sa_z[-1],
               color="blue", marker="^", s=150, edgecolor='black')

    # траектория PSO
    pso_x = [p[0] for p in pso_path]
    pso_y = [p[1] for p in pso_path]
    pso_z = [func(p) for p in pso_path]
    ax.plot(pso_x, pso_y, pso_z, color="green", label="PSO best", linewidth=2)
    ax.scatter(pso_x[0], pso_y[0], pso_z[0],
               color="green", marker="o", s=100, edgecolor='black')
    ax.scatter(pso_x[-1], pso_y[-1], pso_z[-1],
               color="green", marker="^", s=150, edgecolor='black')

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