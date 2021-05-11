import numpy as np

from generic.optimization.dual.combined_algorithm.configuration import LARGE_VALUE
from generic.optimization.dual.lagrangian_decomposition import (
    PrimalInformation,
    DualInformation,
)
from generic.optimization.model import OptimizationSense


class BoundsTracker:
    best_primal_solution: PrimalInformation
    best_dual_solution: DualInformation
    best_master_bound: float

    optimality_gap: float
    cutting_plane_gap: float

    sense: OptimizationSense
    primal_feasibility_tolerance: float

    def __init__(self, sense: OptimizationSense, primal_feasibility_tolerance: float = 1e-6):
        self.sense = sense
        self.primal_feasibility_tolerance = primal_feasibility_tolerance
        self.reset()

    def update_primal_bound(self, primal_solution: PrimalInformation):
        max_violation = max(v.max() for v in primal_solution.infeasibilities.values())
        if max_violation <= self.primal_feasibility_tolerance:
            # when maximizing, you want to maximize the primal bound
            if self.sense * (primal_solution.objective - self.best_primal_solution.objective) > 0:
                self.best_primal_solution = primal_solution
                self._update_optimality_gap()

    def update_dual_bound(self, dual_solution: DualInformation):
        # when maximizing, you want to minimize the dual bound
        if self.sense * (self.best_dual_solution.objective - dual_solution.objective) > 0:
            self.best_dual_solution = dual_solution
            self._update_optimality_gap()
            self._update_cp_gap()

    def update_master_bound(self, new_bound: float):
        if self.sense * (new_bound - self.best_master_bound) > 0:
            self.best_master_bound = new_bound
            self._update_cp_gap()

    def _update_optimality_gap(self):
        self.optimality_gap = _compute_gap(self.best_dual_solution.objective, self.best_primal_solution.objective)

    def _update_cp_gap(self):
        self.cutting_plane_gap = _compute_gap(self.best_dual_solution.objective, self.best_master_bound)

    def reset(self):
        self.best_dual_solution = DualInformation.dummy(self.sense)
        self.best_primal_solution = PrimalInformation.dummy(self.sense)

        self.best_master_bound = -self.sense * np.infty

        self.optimality_gap = 1
        self.cutting_plane_gap = 1


def _compute_gap(dual, primal):
    return min(LARGE_VALUE, abs(dual - primal)) / max(min(LARGE_VALUE, abs(primal)), 1e-3)
