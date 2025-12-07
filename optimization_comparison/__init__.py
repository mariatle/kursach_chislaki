# optimization_comparison/__init__.py
"""
Пакет для сравнения алгоритмов оптимизации.
Содержит реализации Threshold Accepting, Simulated Annealing и Particle Swarm Optimization.
"""

from .types import Vector, Bounds, Objective, random_neighbor
from .benchmark_functions import paraboloid, rastrigin, schwefel, BENCHMARK_FUNCTIONS, FUNCTION_BOUNDS
from .algorithms import threshold_algorithm, simulated_annealing, particle_swarm_optimization
from .visualization import create_animation, plot_comparison, create_static_plot
from .runner import run_single_comparison, run_all_comparisons

__version__ = "1.0.0"
__all__ = [
    # Типы
    'Vector', 'Bounds', 'Objective', 'random_neighbor',
    
    # Тестовые функции
    'paraboloid', 'rastrigin', 'schwefel', 'BENCHMARK_FUNCTIONS', 'FUNCTION_BOUNDS',
    
    # Алгоритмы
    'threshold_algorithm', 'simulated_annealing', 'particle_swarm_optimization',
    
    # Визуализация
    'create_animation', 'plot_comparison', 'create_static_plot',
    
    # Запуск
    'run_single_comparison', 'run_all_comparisons',
]