from typing import Tuple, Optional, Protocol

import numpy as np

from generic.optimization.model import OptimizationSense

_DEFAULT_FEASIBILITY_TOLERANCE = 1e-6


class StepSizeRule(Protocol):
    def __call__(
        self,
        current_value: float,
        current_solution: np.ndarray,
        direction: np.ndarray,
        combination_coefficient: float,
    ) -> float:
        ...


class DeflectionRule(Protocol):
    def __call__(self, subgradient: np.ndarray, previous_direction: np.ndarray) -> Tuple[np.ndarray, float]:
        ...


class SubgradientMethod:
    sense: OptimizationSense
    var_lb: np.ndarray
    var_ub: np.ndarray
    _previous_direction: Optional[np.ndarray]
    feasibility_tolerance: float
    current_solution: np.ndarray
    step_size_fun: StepSizeRule
    deflection_fun: DeflectionRule

    def __init__(
        self,
        sense: OptimizationSense,
        var_lb: np.ndarray,
        var_ub: np.ndarray,
        initial_solution: np.ndarray,
        step_size_fun: StepSizeRule,
        deflection_fun: DeflectionRule,
        feasibility_tolerance=_DEFAULT_FEASIBILITY_TOLERANCE,
    ):
        self.sense = sense
        self.var_lb, self.var_ub = var_lb, var_ub
        self.feasibility_tolerance = feasibility_tolerance
        self._previous_direction = None
        self.current_solution = initial_solution.copy()
        self.step_size_fun = step_size_fun
        self.deflection_fun = deflection_fun

    def __call__(self, value: float, subgradient: np.ndarray) -> np.ndarray:
        # compute direction
        # deflect subgradient if needed
        if self._previous_direction is None:
            direction, combination_coef = subgradient.copy(), 1
        else:
            direction, combination_coef = self._deflection_rule(subgradient, self._previous_direction)

        # project direction
        direction = self._project_direction(direction, self.current_solution)

        # direction computation finished. update `_previous_direction` accordingly
        self._previous_direction = direction

        # compute step size
        new_step_size = self._step_size_rule(
            current_value=value,
            current_solution=self.current_solution,
            direction=direction,
            combination_coefficient=combination_coef,
        )

        # compute and project new current_solution
        self.current_solution = np.clip(
            self.current_solution + self.sense * new_step_size * direction,
            self.var_lb,
            self.var_ub,
        )

        return self.current_solution

    def _project_direction(self, direction: np.ndarray, current_solution: np.ndarray) -> np.ndarray:
        # find direction components that need clipping
        clip_lower = np.where((current_solution < self.var_lb + self.feasibility_tolerance) & (direction < 0))
        clip_upper = np.where((current_solution > self.var_ub - self.feasibility_tolerance) & (direction > 0))

        # for the non-clipped components set dummy boundaries, the clipped ones are set to zero instead
        a_min = direction.copy() - 1e-4
        a_min[clip_lower] = 0

        a_max = direction.copy() + 1e-4
        a_max[clip_upper] = 0

        return np.clip(direction, a_min, a_max)

    def _step_size_rule(
        self, current_value: float, current_solution: np.ndarray, direction: np.ndarray, combination_coefficient: float
    ) -> float:
        return self.step_size_fun(current_value, current_solution, direction, combination_coefficient)

    def _deflection_rule(self, subgradient: np.ndarray, previous_direction: np.ndarray) -> Tuple[np.ndarray, float]:
        return self.deflection_fun(subgradient, previous_direction)

    def restart(self):
        self._previous_direction = None
