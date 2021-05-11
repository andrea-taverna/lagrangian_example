from typing import Callable, Tuple
import numpy as np

DeflectionRule = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, float]]


def make_CFMDeflection (coefficient: float = 1.5, tolerance=1e-5) -> DeflectionRule:
    def deflector(subgradient: np.ndarray, previous_direction: np.ndarray) -> [np.ndarray, float]:
        product = np.dot(subgradient, previous_direction)
        if product > -tolerance:
            defl_coefficient = 0
        else:
            defl_coefficient = -coefficient * product / np.linalg.norm(previous_direction) ** 2

        direction = subgradient + defl_coefficient * previous_direction
        combination_coefficient = 1 / (1 + defl_coefficient)
        return direction, combination_coefficient

    return deflector
