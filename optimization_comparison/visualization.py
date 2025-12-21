import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from typing import List, Optional, Tuple

from .types import Vector, Bounds, Objective


# ----------------------------- Style helpers --------------------------------


def _apply_professional_style() -> None:
    """
    Единый «профессиональный» стиль для всех графиков.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.0,
            "axes.titlepad": 10,
        }
    )


def _guess_known_minimum(func_name: str) -> Optional[Tuple[float, float]]:
    """
    Пытаемся угадать известный глобальный минимум по названию функции.
    (Только для визуальной отметки на графике.)
    """
    name = func_name.lower()
    if "параболоид" in name or "rast" in name or "растриг" in name:
        return (0.0, 0.0)
    if "швеф" in name or "schwefel" in name:
        return (420.9687, 420.9687)
    return None


def _evaluate_on_grid(func: Objective, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычисляем Z = f([x, y]) на сетке X,Y.
    func принимает List[float], поэтому вычисление идёт в питоне.
    """
    Z = np.empty_like(X, dtype=float)
    n0, n1 = X.shape
    for i in range(n0):
        for j in range(n1):
            Z[i, j] = func([float(X[i, j]), float(Y[i, j])])
    return Z


def _downsample_path(path: List[Vector], max_points: int = 250) -> List[Vector]:
    """
    Прореживание траектории для более чистого отображения.
    """
    if len(path) <= max_points:
        return path
    idx = np.linspace(0, len(path) - 1, max_points).astype(int)
    return [path[i] for i in idx]


def _add_summary_box(ax, hist_thr: List[float], hist_sa: List[float], hist_pso: List[float]) -> None:
    """
    Добавляет компактную сводку min f(x*) на график сходимости.
    """
    def _min_safe(h: List[float]) -> float:
        return float(np.min(h)) if len(h) > 0 else float("nan")

    txt = (
        "min f(x*)\n"
        f"TA:  {_min_safe(hist_thr):.3g}\n"
        f"SA:  {_min_safe(hist_sa):.3g}\n"
        f"PSO: {_min_safe(hist_pso):.3g}"
    )
    ax.text(
        0.99,
        0.95,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, linewidth=0.8),
    )


# ------------------------------ Animation -----------------------------------


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
    _apply_professional_style()

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    fig.suptitle(f"Сравнение траекторий: {func_name}", y=1.02, fontsize=14)

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    # Поверхность функции
    n_grid = 32
    xs = np.linspace(x_min, x_max, n_grid)
    ys = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(xs, ys)
    Z = _evaluate_on_grid(func, X, Y)

    # 3D поверхность
    ax3d.plot_surface(X, Y, Z, alpha=0.65, linewidth=0, antialiased=True, cmap="viridis")
    ax3d.set_xlabel("x1")
    ax3d.set_ylabel("x2")
    ax3d.set_zlabel("f(x)")
    ax3d.view_init(elev=28, azim=-55)

    # 2D — карта уровней
    cf = ax2d.contourf(X, Y, Z, levels=40, cmap="viridis")
    ax2d.contour(X, Y, Z, levels=15, colors="k", alpha=0.15, linewidths=0.7)
    ax2d.set_xlabel("x1")
    ax2d.set_ylabel("x2")
    ax2d.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(cf, ax=ax2d, shrink=0.85)
    cbar.set_label("f(x)")

    # Предвычисляем значения f вдоль траекторий (для 3D)
    thr_z_all = [func(p) for p in thr_path]
    sa_z_all = [func(p) for p in sa_path]
    pso_z_all = [func(p) for p in pso_path]

    # Линии и точки
    line_thr_3d, = ax3d.plot([], [], [], color="#d62728", label="Threshold (TA)")
    point_thr_3d, = ax3d.plot([], [], [], "o", color="#d62728", markersize=6)

    line_sa_3d, = ax3d.plot([], [], [], color="#1f77b4", label="Simulated Annealing (SA)")
    point_sa_3d, = ax3d.plot([], [], [], "o", color="#1f77b4", markersize=6)

    line_pso_3d, = ax3d.plot([], [], [], color="#2ca02c", label="PSO (gbest)")
    point_pso_3d, = ax3d.plot([], [], [], "o", color="#2ca02c", markersize=6)

    line_thr_2d, = ax2d.plot([], [], color="#d62728", label="TA")
    point_thr_2d, = ax2d.plot([], [], "o", color="#d62728", markersize=4)

    line_sa_2d, = ax2d.plot([], [], color="#1f77b4", label="SA")
    point_sa_2d, = ax2d.plot([], [], "o", color="#1f77b4", markersize=4)

    line_pso_2d, = ax2d.plot([], [], color="#2ca02c", label="PSO gbest")
    point_pso_2d, = ax2d.plot([], [], "o", color="#2ca02c", markersize=4)

    swarm_scatter_3d = ax3d.scatter([], [], [], c="k", marker=".", alpha=0.45, s=18, label="PSO swarm")
    swarm_scatter_2d = ax2d.scatter([], [], c="k", marker=".", alpha=0.35, s=10)

    # Известный минимум (если угадаем)
    known_min = _guess_known_minimum(func_name)
    if known_min is not None:
        kmx, kmy = known_min
        ax2d.scatter([kmx], [kmy], marker="*", s=180, c="gold", edgecolor="black", linewidth=1.0, zorder=6)
        ax2d.text(kmx, kmy, "  global min", va="center", fontsize=10)

    ax3d.legend(loc="upper left")
    ax2d.legend(loc="upper right")

    text_iter = ax3d.text2D(
        0.04,
        0.95,
        "",
        transform=ax3d.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
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
        for artist in (line_thr_3d, point_thr_3d, line_sa_3d, point_sa_3d, line_pso_3d, point_pso_3d):
            artist.set_data([], [])
            artist.set_3d_properties([])

        for artist in (line_thr_2d, point_thr_2d, line_sa_2d, point_sa_2d, line_pso_2d, point_pso_2d):
            artist.set_data([], [])

        swarm_scatter_3d._offsets3d = ([], [], [])
        swarm_scatter_2d.set_offsets(np.empty((0, 2)))
        text_iter.set_text("")
        return (
            line_thr_3d, point_thr_3d, line_sa_3d, point_sa_3d, line_pso_3d, point_pso_3d,
            swarm_scatter_3d,
            line_thr_2d, point_thr_2d, line_sa_2d, point_sa_2d, line_pso_2d, point_pso_2d,
            swarm_scatter_2d, text_iter
        )

    def update(frame):
        # TA
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

        # SA
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

        # PSO gbest
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

        # Swarm
        k_swarm = sample_idx(frame, len_swarm)
        swarm = swarm_history[k_swarm]
        swarm_x = [p[0] for p in swarm]
        swarm_y = [p[1] for p in swarm]
        swarm_z = [func(p) for p in swarm]
        swarm_scatter_3d._offsets3d = (swarm_x, swarm_y, swarm_z)
        swarm_scatter_2d.set_offsets(np.column_stack([swarm_x, swarm_y]) if swarm_x else np.empty((0, 2)))

        text_iter.set_text(f"Кадр: {frame + 1}/{n_frames}")
        return (
            line_thr_3d, point_thr_3d, line_sa_3d, point_sa_3d, line_pso_3d, point_pso_3d,
            swarm_scatter_3d,
            line_thr_2d, point_thr_2d, line_sa_2d, point_sa_2d, line_pso_2d, point_pso_2d,
            swarm_scatter_2d, text_iter
        )

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=max(1, 1000 // fps),
        blit=False,
    )

    print(f"  Сохраняю GIF: {filename} ({n_frames} кадров, {fps} FPS)")
    anim.save(filename, writer="pillow", fps=fps, dpi=120)
    plt.close(fig)


# ------------------------------ Static plots --------------------------------


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
    Профессиональный статический график сравнения:
      - сверху: сходимость (обычная линейная шкала по y),
      - снизу слева: 2D-карта уровней + траектории,
      - снизу справа: 3D-поверхность + траектории.
    """
    _apply_professional_style()

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 1.25])

    # ---------------- Convergence (LINEAR) ----------------
    ax_conv = fig.add_subplot(gs[0, :])

    ax_conv.plot(hist_thr, label="Threshold Accepting (TA)", color="#d62728")
    ax_conv.plot(hist_sa, label="Simulated Annealing (SA)", color="#1f77b4")
    ax_conv.plot(hist_pso, label="Particle Swarm (PSO, gbest)", color="#2ca02c")

    ax_conv.set_xlabel("Эпоха / итерация")
    ax_conv.set_ylabel("Лучшее значение f(x)")
    ax_conv.set_title(f"Сходимость алгоритмов — {func_name}")
    ax_conv.legend(loc="best")

    # линия уровня «лучшего из лучших»
    try:
        min_val = float(np.min([np.min(hist_thr), np.min(hist_sa), np.min(hist_pso)]))
        ax_conv.axhline(y=min_val, color="gray", linestyle="--", alpha=0.5, linewidth=1.0)
    except ValueError:
        pass

    _add_summary_box(ax_conv, hist_thr, hist_sa, hist_pso)

    # ---------------- 2D contour + paths ----------------
    ax2d = fig.add_subplot(gs[1, 0])

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    n_grid = 140
    xs = np.linspace(x_min, x_max, n_grid)
    ys = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(xs, ys)
    Z = _evaluate_on_grid(func, X, Y)

    cf = ax2d.contourf(X, Y, Z, levels=55, cmap="viridis")
    ax2d.contour(X, Y, Z, levels=18, colors="k", alpha=0.18, linewidths=0.8)
    cbar = fig.colorbar(cf, ax=ax2d, shrink=0.9)
    cbar.set_label("f(x)")

    thr_xy = np.array(thr_path, dtype=float)
    sa_xy = np.array(sa_path, dtype=float)
    pso_xy = np.array(pso_path, dtype=float)

    ax2d.plot(thr_xy[:, 0], thr_xy[:, 1], color="#d62728", label="TA path", alpha=0.95)
    ax2d.plot(sa_xy[:, 0], sa_xy[:, 1], color="#1f77b4", label="SA path", alpha=0.95)
    ax2d.plot(pso_xy[:, 0], pso_xy[:, 1], color="#2ca02c", label="PSO gbest path", alpha=0.95)

    # Start/end markers
    ax2d.scatter(thr_xy[0, 0], thr_xy[0, 1], s=70, marker="o", color="#d62728", edgecolor="black", linewidth=0.8)
    ax2d.scatter(thr_xy[-1, 0], thr_xy[-1, 1], s=110, marker="^", color="#d62728", edgecolor="black", linewidth=0.8)

    ax2d.scatter(sa_xy[0, 0], sa_xy[0, 1], s=70, marker="o", color="#1f77b4", edgecolor="black", linewidth=0.8)
    ax2d.scatter(sa_xy[-1, 0], sa_xy[-1, 1], s=110, marker="^", color="#1f77b4", edgecolor="black", linewidth=0.8)

    ax2d.scatter(pso_xy[0, 0], pso_xy[0, 1], s=70, marker="o", color="#2ca02c", edgecolor="black", linewidth=0.8)
    ax2d.scatter(pso_xy[-1, 0], pso_xy[-1, 1], s=110, marker="^", color="#2ca02c", edgecolor="black", linewidth=0.8)

    # Swarm (last)
    if swarm_history:
        swarm_last = np.array(swarm_history[-1], dtype=float)
        ax2d.scatter(
            swarm_last[:, 0],
            swarm_last[:, 1],
            s=14,
            marker=".",
            c="black",
            alpha=0.25,
            label="PSO swarm (last)",
        )

    # Known global min (if guessed)
    known_min = _guess_known_minimum(func_name)
    if known_min is not None:
        kmx, kmy = known_min
        ax2d.scatter([kmx], [kmy], marker="*", s=220, c="gold", edgecolor="black", linewidth=1.0, zorder=7)
        ax2d.text(kmx, kmy, "  global min", va="center", fontsize=10)

    ax2d.set_title("Карта уровней + траектории (вид сверху)")
    ax2d.set_xlabel("x1")
    ax2d.set_ylabel("x2")
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.legend(loc="upper right")

    # ---------------- 3D surface + (downsampled) paths ----------------
    ax3d = fig.add_subplot(gs[1, 1], projection="3d")

    n_grid3 = 70
    xs3 = np.linspace(x_min, x_max, n_grid3)
    ys3 = np.linspace(y_min, y_max, n_grid3)
    X3, Y3 = np.meshgrid(xs3, ys3)
    Z3 = _evaluate_on_grid(func, X3, Y3)

    ax3d.plot_surface(X3, Y3, Z3, alpha=0.62, linewidth=0, antialiased=True, cmap="viridis")

    thr_ds = _downsample_path(thr_path, max_points=220)
    sa_ds = _downsample_path(sa_path, max_points=220)
    pso_ds = _downsample_path(pso_path, max_points=220)

    thr_x = [p[0] for p in thr_ds]
    thr_y = [p[1] for p in thr_ds]
    thr_z = [func(p) for p in thr_ds]
    ax3d.plot(thr_x, thr_y, thr_z, color="#d62728", label="TA", alpha=0.95)

    sa_x = [p[0] for p in sa_ds]
    sa_y = [p[1] for p in sa_ds]
    sa_z = [func(p) for p in sa_ds]
    ax3d.plot(sa_x, sa_y, sa_z, color="#1f77b4", label="SA", alpha=0.95)

    pso_x = [p[0] for p in pso_ds]
    pso_y = [p[1] for p in pso_ds]
    pso_z = [func(p) for p in pso_ds]
    ax3d.plot(pso_x, pso_y, pso_z, color="#2ca02c", label="PSO gbest", alpha=0.95)

    ax3d.set_title("3D-поверхность + траектории")
    ax3d.set_xlabel("x1")
    ax3d.set_ylabel("x2")
    ax3d.set_zlabel("f(x)")
    ax3d.view_init(elev=28, azim=-55)
    ax3d.legend(loc="upper right")

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
    Компактный статический график (в основном для 3D),
    если не хочется полный comparison-лейаут.
    """
    _apply_professional_style()

    fig = plt.figure(figsize=(10.5, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    n_grid = 75
    xs = np.linspace(x_min, x_max, n_grid)
    ys = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(xs, ys)
    Z = _evaluate_on_grid(func, X, Y)

    ax.plot_surface(X, Y, Z, alpha=0.62, linewidth=0, antialiased=True, cmap="viridis")

    thr_ds = _downsample_path(thr_path, max_points=260)
    sa_ds = _downsample_path(sa_path, max_points=260)
    pso_ds = _downsample_path(pso_path, max_points=260)

    thr_x = [p[0] for p in thr_ds]
    thr_y = [p[1] for p in thr_ds]
    thr_z = [func(p) for p in thr_ds]
    ax.plot(thr_x, thr_y, thr_z, color="#d62728", label="TA")
    ax.scatter(thr_x[0], thr_y[0], thr_z[0], color="#d62728", marker="o", s=70, edgecolor="black", linewidth=0.8)
    ax.scatter(thr_x[-1], thr_y[-1], thr_z[-1], color="#d62728", marker="^", s=110, edgecolor="black", linewidth=0.8)

    sa_x = [p[0] for p in sa_ds]
    sa_y = [p[1] for p in sa_ds]
    sa_z = [func(p) for p in sa_ds]
    ax.plot(sa_x, sa_y, sa_z, color="#1f77b4", label="SA")
    ax.scatter(sa_x[0], sa_y[0], sa_z[0], color="#1f77b4", marker="o", s=70, edgecolor="black", linewidth=0.8)
    ax.scatter(sa_x[-1], sa_y[-1], sa_z[-1], color="#1f77b4", marker="^", s=110, edgecolor="black", linewidth=0.8)

    pso_x = [p[0] for p in pso_ds]
    pso_y = [p[1] for p in pso_ds]
    pso_z = [func(p) for p in pso_ds]
    ax.plot(pso_x, pso_y, pso_z, color="#2ca02c", label="PSO gbest")
    ax.scatter(pso_x[0], pso_y[0], pso_z[0], color="#2ca02c", marker="o", s=70, edgecolor="black", linewidth=0.8)
    ax.scatter(pso_x[-1], pso_y[-1], pso_z[-1], color="#2ca02c", marker="^", s=110, edgecolor="black", linewidth=0.8)

    if swarm_history:
        swarm_last = swarm_history[-1]
        swarm_x = [p[0] for p in swarm_last]
        swarm_y = [p[1] for p in swarm_last]
        swarm_z = [func(p) for p in swarm_last]
        ax.scatter(swarm_x, swarm_y, swarm_z, color="black", marker=".", alpha=0.25, s=18, label="PSO swarm (last)")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")
    ax.set_title(f"{func_name} — траектории алгоритмов")
    ax.view_init(elev=28, azim=-55)
    ax.legend(loc="best")

    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Статический график сохранен: {filename}")