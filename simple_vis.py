"""
Построение 3D-поверхностей для 5 тестовых функций (Paraboloid, Rastrigin,
Schwefel, Matyas, Booth) с помощью matplotlib.

Как запускать:
1) положи этот файл рядом с benchmark_functions.py
2) установи зависимости: pip install numpy matplotlib
3) запусти: python plot_surfaces.py

Скрипт сохранит 5 PNG-файлов и (по желанию) покажет окна с графиками.
"""

import numpy as np
import matplotlib.pyplot as plt


# --- Пытаемся импортировать функции/границы из твоего файла -----------------
try:
    import benchmark_functions as bf  # если файл рядом

    FUNCS = {
        "paraboloid": (bf.paraboloid, bf.FUNCTION_BOUNDS["paraboloid"], "Paraboloid: f(x,y)=x^2+y^2"),
        "rastrigin": (bf.rastrigin, bf.FUNCTION_BOUNDS["rastrigin"], "Rastrigin"),
        "schwefel": (bf.schwefel, bf.FUNCTION_BOUNDS["schwefel"], "Schwefel"),
        "matyas": (bf.matyas, bf.FUNCTION_BOUNDS["matyas"], "Matyas"),
        "booth": (bf.booth, bf.FUNCTION_BOUNDS["booth"], "Booth"),
    }

except Exception:
    # --- Фоллбек: если import не сработал, определяем функции прямо тут -------
    demonstrates = True

    def paraboloid(x):
        return x[0] ** 2 + x[1] ** 2

    def rastrigin(x):
        A = 10
        return 2 * A + (x[0] ** 2 - A * np.cos(2 * np.pi * x[0])) + (x[1] ** 2 - A * np.cos(2 * np.pi * x[1]))

    def schwefel(x):
        # 2D вариант из формулы: 418.982887*n - sum(x_i*sin(sqrt(|x_i|)))
        n = 2
        return 418.982887 * n - (x[0] * np.sin(np.sqrt(abs(x[0]))) + x[1] * np.sin(np.sqrt(abs(x[1]))))

    def matyas(x):
        x1, x2 = x
        return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2

    def booth(x):
        x1, x2 = x
        return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2

    FUNCS = {
        "paraboloid": (paraboloid, [(-5, 5), (-5, 5)], "Paraboloid: f(x,y)=x^2+y^2"),
        "rastrigin": (rastrigin, [(-5.12, 5.12), (-5.12, 5.12)], "Rastrigin"),
        "schwefel": (schwefel, [(-500, 500), (-500, 500)], "Schwefel"),
        "matyas": (matyas, [(-10, 10), (-10, 10)], "Matyas"),
        "booth": (booth, [(-10, 10), (-10, 10)], "Booth"),
    }


def evaluate_on_grid(func, X, Y):
    """func принимает List[float] => считаем Z через двойной цикл (надёжно и совместимо с твоим кодом)."""
    Z = np.empty_like(X, dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([float(X[i, j]), float(Y[i, j])])
    return Z


def plot_surface(func, bounds, title, filename, n_grid=150, z_clip=None, show=False):
    (x_min, x_max), (y_min, y_max) = bounds

    xs = np.linspace(x_min, x_max, n_grid)
    ys = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(xs, ys)

    Z = evaluate_on_grid(func, X, Y)

    # Иногда у Швеффеля/Растригина бывают огромные значения — можно подрезать Z для читаемости
    if z_clip is not None:
        Z = np.clip(Z, z_clip[0], z_clip[1])

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)  # без задания цвета/стиля

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")

    plt.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(show=False):
    # Для Швеффеля сетку делаем реже, иначе будет тяжеловато (и график не очень информативный на огромном диапазоне)
    for key, (func, bounds, title) in FUNCS.items():
        if key == "schwefel":
            plot_surface(
                func=func,
                bounds=bounds,
                title=title,
                filename=f"{key}_surface.png",
                n_grid=120,          # меньше точек
                z_clip=None,         # можешь поставить, например: z_clip=(0, 2000)
                show=show,
            )
        else:
            plot_surface(
                func=func,
                bounds=bounds,
                title=title,
                filename=f"{key}_surface.png",
                n_grid=160,
                z_clip=None,         # для Rastrigin можно поставить z_clip=(0, 200) если "плоско"
                show=show,
            )

    print("Готово! Сохранены файлы:")
    for key in FUNCS.keys():
        print(f" - {key}_surface.png")


if __name__ == "__main__":
    main(show=False)  # поставь True, если хочешь, чтобы окна с графиками открывались