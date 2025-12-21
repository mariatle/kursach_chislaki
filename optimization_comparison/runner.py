import random
import json
from typing import Optional, Dict, Any

from . import algorithms
from . import benchmark_functions
from . import visualization
from .types import Bounds, Objective


# --- Конфигурация параметров алгоритмов -------------------------------------


DEFAULT_ALGO_PARAMS: Dict[str, Any] = {
    "n_outer": 50,
    "n_inner": 20,
    "n_particles": 30,
    "ta_eps0": 1.0,
    "sa_T0": 1.0,
    "alpha": 0.95,
    "step0": 1.0,
    "pso_w": 0.5,
    "pso_c1": 1.5,
    "pso_c2": 1.5,
}


# Профили параметров для разных тестовых функций.
FUNCTION_ALGO_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # Простой выпуклый параболоид
    "paraboloid": {
        "n_outer": 50,
        "n_inner": 20,
        "n_particles": 25,
        "ta_eps0": 0.5,
        "sa_T0": 1.0,
        "alpha": 0.9,
        "step0": 0.7,
        "pso_w": 0.6,
        "pso_c1": 1.5,
        "pso_c2": 1.5,
    },
    # Многомодальная функция Растригина
    "rastrigin": {
        "n_outer": 80,
        "n_inner": 25,
        "n_particles": 40,
        "ta_eps0": 1.5,
        "sa_T0": 5.0,
        "alpha": 0.97,
        "step0": 0.6,
        "pso_w": 0.7,
        "pso_c1": 1.6,
        "pso_c2": 1.6,
    },
    # Очень «ломаный» ландшафт Швеффеля
    "schwefel": {
        "n_outer": 100,
        "n_inner": 30,
        "n_particles": 50,
        "ta_eps0": 50.0,
        "sa_T0": 200.0,
        "alpha": 0.98,
        "step0": 40.0,
        "pso_w": 0.7,
        "pso_c1": 1.7,
        "pso_c2": 1.7,
    },
    # Новые «простые» функции (красиво сходятся)
    "matyas": {
        "n_outer": 50,
        "n_inner": 20,
        "n_particles": 25,
        "ta_eps0": 0.5,
        "sa_T0": 1.0,
        "alpha": 0.9,
        "step0": 0.7,
        "pso_w": 0.6,
        "pso_c1": 1.5,
        "pso_c2": 1.5,
    },
    "booth": {
        "n_outer": 60,
        "n_inner": 20,
        "n_particles": 30,
        "ta_eps0": 0.8,
        "sa_T0": 1.5,
        "alpha": 0.9,
        "step0": 0.8,
        "pso_w": 0.6,
        "pso_c1": 1.5,
        "pso_c2": 1.5,
    },
}


def get_algorithm_params(
    func_key: str,
    global_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Возвращает словарь параметров алгоритмов для заданной тестовой функции.
    """
    params: Dict[str, Any] = dict(DEFAULT_ALGO_PARAMS)
    params.update(FUNCTION_ALGO_OVERRIDES.get(func_key, {}))
    if global_overrides:
        params.update(global_overrides)
    return params


def _format_profile_line(label: str, items: Dict[str, Any]) -> str:
    return f"  {label}: " + ", ".join(f"{k}={v}" for k, v in items.items())


def print_params_profile(func_key: str, params: Dict[str, Any]) -> None:
    """
    Красивый вывод профиля параметров для выбранной функции.
    """
    n_outer = params.get("n_outer", DEFAULT_ALGO_PARAMS["n_outer"])
    n_inner = params.get("n_inner", DEFAULT_ALGO_PARAMS["n_inner"])
    n_particles = params.get("n_particles", DEFAULT_ALGO_PARAMS["n_particles"])
    n_iters = max(1, (n_outer * n_inner) // max(1, n_particles))

    print(f"Профиль параметров для {func_key}:")
    print(
        _format_profile_line(
            "TA",
            {
                "eps0": params.get("ta_eps0"),
                "alpha": params.get("alpha"),
                "n_outer": n_outer,
                "n_inner": n_inner,
                "step0": params.get("step0"),
            },
        )
    )
    print(
        _format_profile_line(
            "SA",
            {
                "T0": params.get("sa_T0"),
                "alpha": params.get("alpha"),
                "n_outer": n_outer,
                "n_inner": n_inner,
                "step0": params.get("step0"),
            },
        )
    )
    print(
        _format_profile_line(
            "PSO",
            {
                "n_particles": n_particles,
                "n_iters": n_iters,
                "w": params.get("pso_w"),
                "c1": params.get("pso_c1"),
                "c2": params.get("pso_c2"),
            },
        )
    )
    print()


def _merge_params(algorithm_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    params = dict(DEFAULT_ALGO_PARAMS)
    if algorithm_params:
        params.update(algorithm_params)
    return params


def _print_single_run_summary(val_thr: float, val_sa: float, val_pso: float) -> None:
    summary = [
        ("Пороговый алгоритм", val_thr),
        ("Имитация отжига", val_sa),
        ("Рой частиц (PSO)", val_pso),
    ]
    summary_sorted = sorted(summary, key=lambda x: x[1])

    print("Сравнение по итоговым значениям f(x*):")
    for rank, (name, val) in enumerate(summary_sorted, start=1):
        print(f"  {rank}. {name}: f(x*) = {val:.6f}")
    best_name, best_val = summary_sorted[0]
    print(f"--> Лучший результат на этой функции дал: {best_name} (f(x*) = {best_val:.6f})")
    print("-" * 40)


def run_single_comparison(
    func: Objective,
    func_name: str,
    bounds: Bounds,
    seed: int = 0,
    create_gif: bool = False,
    gif_filename: Optional[str] = None,
    max_frames: int = 200,
    algorithm_params: Optional[Dict[str, Any]] = None,
) -> tuple:
    """
    Запускает сравнение трех алгоритмов на одной функции.
    Возвращает (fig, results_dict).
    """
    random.seed(seed)
    params = _merge_params(algorithm_params)

    # одинаковая стартовая точка для SA и TA
    x0 = [random.uniform(low, high) for (low, high) in bounds]

    # --- Threshold Accepting ---
    best_thr, val_thr, hist_thr, thr_path = algorithms.threshold_algorithm(
        func=func,
        bounds=bounds,
        x0=x0,
        eps0=params["ta_eps0"],
        alpha=params["alpha"],
        n_outer=params["n_outer"],
        n_inner=params["n_inner"],
        step0=params["step0"],
    )

    # --- Simulated Annealing ---
    best_sa, val_sa, hist_sa, sa_path = algorithms.simulated_annealing(
        func=func,
        bounds=bounds,
        x0=x0,
        T0=params["sa_T0"],
        alpha=params["alpha"],
        n_outer=params["n_outer"],
        n_inner=params["n_inner"],
        step0=params["step0"],
    )

    # --- PSO ---
    n_iters = max(1, (params["n_outer"] * params["n_inner"]) // params["n_particles"])

    best_pso, val_pso, hist_pso, pso_path, swarm_history = algorithms.particle_swarm_optimization(
        func=func,
        bounds=bounds,
        n_particles=params["n_particles"],
        n_iters=n_iters,
        w=params["pso_w"],
        c1=params["pso_c1"],
        c2=params["pso_c2"],
    )

    # Вывод результатов в консоль
    print(f"\nРезультаты для {func_name}:")
    print(f"{'-'*40}")
    print(f"Начальная точка: {x0}")
    print(f"{'-'*40}")
    print("Пороговый алгоритм:")
    print(f"  x* = {[round(v, 4) for v in best_thr]}")
    print(f"  f(x*) = {val_thr:.6f}")
    print(f"{'-'*40}")
    print("Имитация отжига:")
    print(f"  x* = {[round(v, 4) for v in best_sa]}")
    print(f"  f(x*) = {val_sa:.6f}")
    print(f"{'-'*40}")
    print("Рой частиц (PSO):")
    print(f"  x* = {[round(v, 4) for v in best_pso]}")
    print(f"  f(x*) = {val_pso:.6f}")
    print(f"{'-'*40}")

    _print_single_run_summary(val_thr, val_sa, val_pso)

    fig = visualization.plot_comparison(
        func=func,
        func_name=func_name,
        bounds=bounds,
        hist_thr=hist_thr,
        hist_sa=hist_sa,
        hist_pso=hist_pso,
        thr_path=thr_path,
        sa_path=sa_path,
        pso_path=pso_path,
        swarm_history=swarm_history,
    )

    if create_gif and gif_filename:
        print(f"Создание GIF-анимации: {gif_filename}")
        visualization.create_animation(
            func=func,
            func_name=func_name,
            bounds=bounds,
            thr_path=thr_path,
            sa_path=sa_path,
            pso_path=pso_path,
            swarm_history=swarm_history,
            filename=gif_filename,
            max_frames=max_frames,
        )

    results = {
        "threshold": {"best": best_thr, "value": val_thr, "history": hist_thr, "path": thr_path},
        "simulated_annealing": {"best": best_sa, "value": val_sa, "history": hist_sa, "path": sa_path},
        "particle_swarm": {
            "best": best_pso,
            "value": val_pso,
            "history": hist_pso,
            "path": pso_path,
            "swarm_history": swarm_history,
        },
    }
    return fig, results


def _compute_stats(all_values: Dict[str, list], best_results: Dict[str, Any]):
    stats = {}
    for algo, values in all_values.items():
        if not values:
            continue
        v_min = min(values)
        v_max = max(values)
        v_mean = sum(values) / len(values)
        best_info = best_results.get(algo, {})
        stats[algo] = {
            "values": values,
            "min": v_min,
            "max": v_max,
            "mean": v_mean,
            "best_value": best_info.get("value", v_min),
            "best_point": best_info.get("best"),
            "best_seed": best_info.get("seed"),
            "best_run_index": best_info.get("run_index"),
        }
    return stats


def run_repeated_single_comparison(
    func: Objective,
    func_name: str,
    bounds: Bounds,
    n_runs: int = 10,
    base_seed: int = 0,
    create_gif: bool = False,
    gif_basename: Optional[str] = None,
    max_frames: int = 200,
    algorithm_params: Optional[Dict[str, Any]] = None,
    json_filename: Optional[str] = None,
):
    if n_runs <= 0:
        raise ValueError("n_runs должно быть положительным")

    best_results: Dict[str, Dict[str, Any]] = {}
    best_values = {"threshold": float("inf"), "simulated_annealing": float("inf"), "particle_swarm": float("inf")}
    all_values = {"threshold": [], "simulated_annealing": [], "particle_swarm": []}

    import matplotlib.pyplot as plt

    for i in range(n_runs):
        seed = base_seed + i
        print(f"\n=== Повтор {i+1}/{n_runs}, seed={seed} ===")

        fig_tmp, res = run_single_comparison(
            func=func,
            func_name=func_name,
            bounds=bounds,
            seed=seed,
            create_gif=False,
            gif_filename=None,
            max_frames=max_frames,
            algorithm_params=algorithm_params,
        )

        for algo_key in best_values.keys():
            val = res[algo_key]["value"]
            all_values[algo_key].append(val)

            if val < best_values[algo_key]:
                best_values[algo_key] = val
                best_results[algo_key] = dict(res[algo_key])
                best_results[algo_key]["seed"] = seed
                best_results[algo_key]["run_index"] = i

        plt.close(fig_tmp)

    print("\n===== Результаты по лучшим запускам (по каждому алгоритму) =====")
    for algo_key, val in best_values.items():
        info = best_results[algo_key]
        print(f"\nАлгоритм: {algo_key}")
        print(f"  лучший f(x*) = {val:.6f}")
        print(f"  точка x* = {[round(v, 4) for v in info['best']]}")
        print(f"  получен в запуске #{info['run_index']+1} (seed={info['seed']})")

    stats = _compute_stats(all_values, best_results)

    print("\n===== Сводная статистика (min / mean / max по f(x*)) =====")
    for algo_key, s in stats.items():
        print(f"\nАлгоритм: {algo_key}")
        print(f"  min  = {s['min']:.6f}")
        print(f"  mean = {s['mean']:.6f}")
        print(f"  max  = {s['max']:.6f}")

    if json_filename:
        data = {"function": func_name, "n_runs": n_runs, "base_seed": base_seed, "algo_stats": stats}
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Статистика сохранена в JSON: {json_filename}")

    fig = visualization.plot_comparison(
        func=func,
        func_name=func_name,
        bounds=bounds,
        hist_thr=best_results["threshold"]["history"],
        hist_sa=best_results["simulated_annealing"]["history"],
        hist_pso=best_results["particle_swarm"]["history"],
        thr_path=best_results["threshold"]["path"],
        sa_path=best_results["simulated_annealing"]["path"],
        pso_path=best_results["particle_swarm"]["path"],
        swarm_history=best_results["particle_swarm"]["swarm_history"],
    )

    if create_gif and gif_basename is not None:
        gif_filename = f"{gif_basename}.gif"
        print(f"Создание GIF-анимации (по лучшим траекториям): {gif_filename}")
        visualization.create_animation(
            func=func,
            func_name=func_name,
            bounds=bounds,
            thr_path=best_results["threshold"]["path"],
            sa_path=best_results["simulated_annealing"]["path"],
            pso_path=best_results["particle_swarm"]["path"],
            swarm_history=best_results["particle_swarm"]["swarm_history"],
            filename=gif_filename,
            max_frames=max_frames,
        )

    return fig, best_results, stats


def run_all_comparisons(
    create_gifs: bool = False,
    seed: int = 0,
    max_frames: int = 200,
    algorithm_params: Optional[Dict[str, Any]] = None,
    n_runs: int = 1,
):
    """
    Запускает сравнение на всех тестовых функциях.
    """
    results = {}
    all_stats = {}
    current_seed = seed

    def _run_one(func_key: str, func: Objective, func_name: str, png_name: str, stats_name: str):
        params = get_algorithm_params(func_key, algorithm_params)
        print_params_profile(func_key, params)

        if n_runs <= 1:
            fig, res = run_single_comparison(
                func=func,
                func_name=func_name,
                bounds=benchmark_functions.FUNCTION_BOUNDS[func_key],
                seed=current_seed,
                create_gif=create_gifs,
                gif_filename=f"{func_key}.gif" if create_gifs else None,
                max_frames=max_frames,
                algorithm_params=params,
            )
            stats = None
        else:
            fig, res, stats = run_repeated_single_comparison(
                func=func,
                func_name=func_name,
                bounds=benchmark_functions.FUNCTION_BOUNDS[func_key],
                n_runs=n_runs,
                base_seed=current_seed,
                create_gif=create_gifs,
                gif_basename=func_key,
                max_frames=max_frames,
                algorithm_params=params,
                json_filename=stats_name,
            )

        results[func_key] = res
        if stats is not None:
            all_stats[func_key] = stats
        fig.savefig(png_name, dpi=150, bbox_inches="tight")

    # Порядок: простые -> сложные
    _run_one("paraboloid", benchmark_functions.paraboloid, "Параболоид f(x)=x1^2+x2^2",
             "paraboloid_comparison.png", "paraboloid_stats.json")
    _run_one("matyas", benchmark_functions.matyas, "Функция Матияса",
             "matyas_comparison.png", "matyas_stats.json")
    _run_one("booth", benchmark_functions.booth, "Функция Бута",
             "booth_comparison.png", "booth_stats.json")
    _run_one("rastrigin", benchmark_functions.rastrigin, "Функция Растригина",
             "rastrigin_comparison.png", "rastrigin_stats.json")
    _run_one("schwefel", benchmark_functions.schwefel, "Функция Швеффеля",
             "schwefel_comparison.png", "schwefel_stats.json")

    if all_stats:
        with open("all_functions_stats.json", "w", encoding="utf-8") as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=2)
        print("Сводная статистика по всем функциям сохранена в all_functions_stats.json")

    return results, all_stats