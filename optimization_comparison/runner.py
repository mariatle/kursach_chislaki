import random
import json
from typing import Optional, Dict, Any

from . import algorithms
from . import benchmark_functions
from . import visualization
from .types import Bounds, Objective


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

    # Параметры по умолчанию
    default_params = {
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

    # Объединяем с пользовательскими параметрами
    if algorithm_params:
        default_params.update(algorithm_params)

    params = default_params

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
    # подбираем число итераций для сопоставимости
    n_iters = max(1, (params["n_outer"] * params["n_inner"]) // params["n_particles"])

    best_pso, val_pso, hist_pso, pso_path, swarm_history = (
        algorithms.particle_swarm_optimization(
            func=func,
            bounds=bounds,
            n_particles=params["n_particles"],
            n_iters=n_iters,
            w=params["pso_w"],
            c1=params["pso_c1"],
            c2=params["pso_c2"],
        )
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

    # Создаем фигуру сравнения
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

    # Создаем GIF если нужно
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
        "threshold": {
            "best": best_thr,
            "value": val_thr,
            "history": hist_thr,
            "path": thr_path,
        },
        "simulated_annealing": {
            "best": best_sa,
            "value": val_sa,
            "history": hist_sa,
            "path": sa_path,
        },
        "particle_swarm": {
            "best": best_pso,
            "value": val_pso,
            "history": hist_pso,
            "path": pso_path,
            "swarm_history": swarm_history,
        },
    }

    return fig, results


def _compute_stats(all_values: Dict[str, list], best_results: Dict[str, Dict[str, Any]]):
    """
    Считает min/mean/max для каждого алгоритма + инфо о лучшем запуске.
    """
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
    """
    Многократный запуск run_single_comparison на одной функции.

    Делает:
      - несколько запусков с разными seed,
      - собирает значения f(x*) для каждого алгоритма,
      - выбирает лучший запуск по каждому алгоритму,
      - считает min/mean/max по f(x*),
      - по желанию сохраняет статистику в JSON.

    Возвращает:
        fig: фигура с траекториями из лучших запусков для каждого алгоритма
        best_results: словарь лучших результатов по каждому алгоритму
        stats: словарь со статистикой (min/mean/max)
    """
    if n_runs <= 0:
        raise ValueError("n_runs должно быть положительным")

    best_results: Dict[str, Dict[str, Any]] = {}
    best_values = {
        "threshold": float("inf"),
        "simulated_annealing": float("inf"),
        "particle_swarm": float("inf"),
    }
    all_values = {
        "threshold": [],
        "simulated_annealing": [],
        "particle_swarm": [],
    }

    import matplotlib.pyplot as plt

    for i in range(n_runs):
        seed = base_seed + i
        print(f"\n=== Повтор {i+1}/{n_runs}, seed={seed} ===")

        # Внутри повторов GIF не создаем, чтобы не плодить файлы
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

        # Обновляем лучшие результаты по каждому алгоритму и копим все значения
        for algo_key in best_values.keys():
            val = res[algo_key]["value"]
            all_values[algo_key].append(val)

            if val < best_values[algo_key]:
                best_values[algo_key] = val
                best_results[algo_key] = dict(res[algo_key])
                best_results[algo_key]["seed"] = seed
                best_results[algo_key]["run_index"] = i

        # Закрываем временную фигуру, чтобы не захламлять окна
        plt.close(fig_tmp)

    # Краткая сводка по лучшим запускам
    print("\n===== Результаты по лучшим запускам (по каждому алгоритму) =====")
    for algo_key, val in best_values.items():
        info = best_results[algo_key]
        print(f"\nАлгоритм: {algo_key}")
        print(f"  лучший f(x*) = {val:.6f}")
        print(f"  точка x* = {[round(v, 4) for v in info['best']]}")
        print(f"  получен в запуске #{info['run_index']+1} (seed={info['seed']})")

    # Статистика min/mean/max
    stats = _compute_stats(all_values, best_results)

    print("\n===== Сводная статистика (min / mean / max по f(x*)) =====")
    for algo_key, s in stats.items():
        print(f"\nАлгоритм: {algo_key}")
        print(f"  min  = {s['min']:.6f}")
        print(f"  mean = {s['mean']:.6f}")
        print(f"  max  = {s['max']:.6f}")

    # Сохраняем статистику в JSON, если нужно
    if json_filename:
        data = {
            "function": func_name,
            "n_runs": n_runs,
            "base_seed": base_seed,
            "algo_stats": stats,
        }
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Статистика сохранена в JSON: {json_filename}")

    # Финальный график — по лучшим траекториям
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

    # Один GIF по лучшим траекториям
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
    Если n_runs > 1, для каждой функции используется режим многократных запусков.
    """
    results = {}
    all_stats = {}

    # 1) Параболоид
    if n_runs <= 1:
        fig1, res1 = run_single_comparison(
            func=benchmark_functions.paraboloid,
            func_name="Параболоид f(x)=x1^2+x2^2",
            bounds=benchmark_functions.FUNCTION_BOUNDS["paraboloid"],
            seed=seed,
            create_gif=create_gifs,
            gif_filename="paraboloid.gif" if create_gifs else None,
            max_frames=max_frames,
            algorithm_params=algorithm_params,
        )
        stats1 = None
    else:
        fig1, res1, stats1 = run_repeated_single_comparison(
            func=benchmark_functions.paraboloid,
            func_name="Параболоид f(x)=x1^2+x2^2",
            bounds=benchmark_functions.FUNCTION_BOUNDS["paraboloid"],
            n_runs=n_runs,
            base_seed=seed,
            create_gif=create_gifs,
            gif_basename="paraboloid",
            max_frames=max_frames,
            algorithm_params=algorithm_params,
            json_filename="paraboloid_stats.json",
        )
    results["paraboloid"] = res1
    if stats1 is not None:
        all_stats["paraboloid"] = stats1
    fig1.savefig("paraboloid_comparison.png", dpi=150, bbox_inches="tight")

    # 2) Растригин
    if n_runs <= 1:
        fig2, res2 = run_single_comparison(
            func=benchmark_functions.rastrigin,
            func_name="Функция Растригина",
            bounds=benchmark_functions.FUNCTION_BOUNDS["rastrigin"],
            seed=seed,
            create_gif=create_gifs,
            gif_filename="rastrigin.gif" if create_gifs else None,
            max_frames=max_frames,
            algorithm_params=algorithm_params,
        )
        stats2 = None
    else:
        fig2, res2, stats2 = run_repeated_single_comparison(
            func=benchmark_functions.rastrigin,
            func_name="Функция Растригина",
            bounds=benchmark_functions.FUNCTION_BOUNDS["rastrigin"],
            n_runs=n_runs,
            base_seed=seed,
            create_gif=create_gifs,
            gif_basename="rastrigin",
            max_frames=max_frames,
            algorithm_params=algorithm_params,
            json_filename="rastrigin_stats.json",
        )
    results["rastrigin"] = res2
    if stats2 is not None:
        all_stats["rastrigin"] = stats2
    fig2.savefig("rastrigin_comparison.png", dpi=150, bbox_inches="tight")

    # 3) Швеффеля
    if n_runs <= 1:
        fig3, res3 = run_single_comparison(
            func=benchmark_functions.schwefel,
            func_name="Функция Швеффеля",
            bounds=benchmark_functions.FUNCTION_BOUNDS["schwefel"],
            seed=seed,
            create_gif=create_gifs,
            gif_filename="schwefel.gif" if create_gifs else None,
            max_frames=max_frames,
            algorithm_params=algorithm_params,
        )
        stats3 = None
    else:
        fig3, res3, stats3 = run_repeated_single_comparison(
            func=benchmark_functions.schwefel,
            func_name="Функция Швеффеля",
            bounds=benchmark_functions.FUNCTION_BOUNDS["schwefel"],
            n_runs=n_runs,
            base_seed=seed,
            create_gif=create_gifs,
            gif_basename="schwefel",
            max_frames=max_frames,
            algorithm_params=algorithm_params,
            json_filename="schwefel_stats.json",
        )
    results["schwefel"] = res3
    if stats3 is not None:
        all_stats["schwefel"] = stats3
    fig3.savefig("schwefel_comparison.png", dpi=150, bbox_inches="tight")

    # Если собирали статистику по нескольким функциям, сохранить общий JSON
    if all_stats:
        with open("all_functions_stats.json", "w", encoding="utf-8") as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=2)
        print("Сводная статистика по всем функциям сохранена в all_functions_stats.json")

    return results, all_stats