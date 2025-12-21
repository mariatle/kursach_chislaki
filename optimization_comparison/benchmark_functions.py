# optimization_comparison/benchmark_functions.py
import math
from typing import List
from .types import Vector


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
def matyas(x):
    x1, x2 = x[0], x[1]
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2

def booth(x):
    x1, x2 = x[0], x[1]
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2




    # новые:



# Словарь доступных функций
BENCHMARK_FUNCTIONS = {
    "paraboloid": paraboloid,
    "rastrigin": rastrigin,
    "schwefel": schwefel,
    "matyas": matyas,
    "booth": booth,
    
}

# Соответствующие области определения
FUNCTION_BOUNDS = {
    "paraboloid": [(-5.0, 5.0), (-5.0, 5.0)],
    "rastrigin": [(-5.12, 5.12), (-5.12, 5.12)],
    "schwefel": [(-500.0, 500.0), (-500.0, 500.0)],
    "matyas": [(-10, 10), (-10, 10)],
    "booth":  [(-10, 10), (-10, 10)],
}