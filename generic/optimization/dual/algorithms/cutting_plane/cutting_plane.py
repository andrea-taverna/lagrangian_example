from dataclasses import dataclass
from typing import Tuple

import numpy as np

from generic.optimization.dual.algorithms.cutting_plane.lp_model import CPLPModel
from generic.optimization.model import OptimizationSense


@dataclass
class CuttingPlane:
    sense: OptimizationSense
    num_vars: int
    var_lb: np.ndarray
    var_ub: np.ndarray
    optimality_tolerance: float
    model: CPLPModel

    def __init__(
        self,
        sense: OptimizationSense,
        var_lb: np.array,
        var_ub: np.ndarray,
        optimality_tolerance=1e-6,
        **kwargs,
    ):
        assert var_ub.size == var_lb.size
        self.sense = sense
        self.var_ub = var_ub
        self.var_lb = var_lb
        self.optimality_tolerance = optimality_tolerance

        self.model = CPLPModel(self.sense, self.var_lb, self.var_ub)

    def add_cut(self, intercept: float, subgradient: np.ndarray):
        self.model.add_cut(intercept, subgradient)

    def __call__(
        self, value: float, intercept: float, subgradient: np.ndarray, **solver_options
    ) -> Tuple[float, np.ndarray]:
        self.add_cut(intercept, subgradient)
        return self.model.solve(**solver_options)
