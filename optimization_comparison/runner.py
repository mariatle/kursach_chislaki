# optimization_comparison/runner.py
import random
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
    
    Args:
        func: Целевая функция
        func_name: Название функции для отображения
        bounds: Границы поиска
        seed: Seed для генератора случайных чисел
        create_gif: Создавать ли GIF-анимацию
        gif_filename: Имя файла для GIF
        max_frames: Максимальное количество кадров в GIF
        algorithm_params: Дополнительные параметры алгоритмов
    
    Returns:
        tuple: (fig, results_dict)
    """
    random.seed(seed)
    
    # Параметры по умолчанию
    default_params = {
        'n_outer': 50,
        'n_inner': 20,
        'n_particles': 30,
        'ta_eps0': 1.0,
        'sa_T0': 1.0,
        'alpha': 0.95,
        'step0': 1.0,
        'pso_w': 0.5,
        'pso_c1': 1.5,
        'pso_c2': 1.5,
    }
    
    # Объединяем с пользовательскими параметрами
    if algorithm_params:
        default_params.update(algorithm_params)
    
    params = default_params
    
    # одинаковая стартовая точка для SA и TA
    x0 = [random.uniform(low, high) for (low, high) in bounds]

    # Запускаем алгоритмы
    best_thr, val_thr, hist_thr, thr_path = algorithms.threshold_algorithm(
        func=func,
        bounds=bounds,
        x0=x0,
        eps0=params['ta_eps0'],
        alpha=params['alpha'],
        n_outer=params['n_outer'],
        n_inner=params['n_inner'],
        step0=params['step0'],
    )

    best_sa, val_sa, hist_sa, sa_path = algorithms.simulated_annealing(
        func=func,
        bounds=bounds,
        x0=x0,
        T0=params['sa_T0'],
        alpha=params['alpha'],
        n_outer=params['n_outer'],
        n_inner=params['n_inner'],
        step0=params['step0'],
    )

    # PSO: подбираем число итераций для сопоставимости
    n_iters = max(1, (params['n_outer'] * params['n_inner']) // params['n_particles'])

    best_pso, val_pso, hist_pso, pso_path, swarm_history = algorithms.particle_swarm_optimization(
        func=func,
        bounds=bounds,
        n_particles=params['n_particles'],
        n_iters=n_iters,
        w=params['pso_w'],
        c1=params['pso_c1'],
        c2=params['pso_c2'],
    )

    # Вывод результатов
    print(f"\nРезультаты для {func_name}:")
    print(f"{'-'*40}")
    print(f"Начальная точка: {x0}")
    print(f"{'-'*40}")
    print(f"Пороговый алгоритм:")
    print(f"  x* = {[round(v, 4) for v in best_thr]}")
    print(f"  f(x*) = {val_thr:.6f}")
    print(f"{'-'*40}")
    print(f"Имитация отжига:")
    print(f"  x* = {[round(v, 4) for v in best_sa]}")
    print(f"  f(x*) = {val_sa:.6f}")
    print(f"{'-'*40}")
    print(f"Рой частиц (PSO):")
    print(f"  x* = {[round(v, 4) for v in best_pso]}")
    print(f"  f(x*) = {val_pso:.6f}")
    print(f"{'-'*40}")

    # Создаем график
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
        'threshold': {
            'best': best_thr,
            'value': val_thr,
            'history': hist_thr,
            'path': thr_path,
        },
        'simulated_annealing': {
            'best': best_sa,
            'value': val_sa,
            'history': hist_sa,
            'path': sa_path,
        },
        'particle_swarm': {
            'best': best_pso,
            'value': val_pso,
            'history': hist_pso,
            'path': pso_path,
            'swarm_history': swarm_history,
        }
    }
    
    return fig, results


def run_all_comparisons(
    create_gifs: bool = False,
    seed: int = 0,
    max_frames: int = 200,
    algorithm_params: Optional[Dict[str, Any]] = None,
):
    """
    Запускает сравнение на всех тестовых функциях.
    """
    results = {}
    
    # 1) Параболоид
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
    results["paraboloid"] = res1
    fig1.savefig("paraboloid_comparison.png", dpi=150, bbox_inches='tight')

    # 2) Растригин
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
    results["rastrigin"] = res2
    fig2.savefig("rastrigin_comparison.png", dpi=150, bbox_inches='tight')

    # 3) Швеффеля
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
    results["schwefel"] = res3
    fig3.savefig("schwefel_comparison.png", dpi=150, bbox_inches='tight')

    return results