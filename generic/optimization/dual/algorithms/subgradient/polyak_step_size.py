from typing import Callable
import numpy as np

from generic.optimization.model import OptimizationSense


class PolyakStepSizeRule:
    size_coefficient: float
    target_tracker: Callable[[float], float]
    sense: OptimizationSense

    def __init__(
        self,
        sense: OptimizationSense,
        target_tracker: Callable[[float], float],
        size_coefficient: float = 1,
    ):
        assert (
            0 < size_coefficient < 2
        ), "Size coefficient must be within (0,2), extremes excluded."
        self.sense = sense
        self.size_coefficient = size_coefficient
        self.target_tracker = target_tracker

    def __call__(
        self,
        current_value: float,
        multipliers: np.ndarray,
        direction: np.ndarray,
        max_size_coefficient=2.0,
    ) -> float:
        target_value = self.target_tracker(current_value)
        actual_size_coef = min(max_size_coefficient, self.size_coefficient)
        return (
            actual_size_coef
            * self.sense.value
            * (target_value - current_value)
            / max(1e-3, np.linalg.norm(direction) ** 2)
        )
