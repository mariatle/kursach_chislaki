import math
import random
from typing import List, Tuple, Callable, TypeAlias


Vector: TypeAlias = List[float]
Bounds: TypeAlias = List[Tuple[float, float]]
Objective: TypeAlias = Callable[[Vector], float]


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