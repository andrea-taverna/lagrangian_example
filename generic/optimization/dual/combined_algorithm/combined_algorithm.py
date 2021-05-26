from typing import List, OrderedDict, Any, Optional, Tuple, Dict
import pandas as pd
import numpy as np

from generic.optimization.dual.combined_algorithm.bounds_tracker import BoundsTracker
from generic.optimization.dual.combined_algorithm.configuration import (
    AlgorithmConfiguration,
    PrimalHeuristicAlgorithm,
)

from generic.optimization.dual.combined_algorithm.dual_runner import DualAlgorithmRunner

from generic.optimization.dual.lagrangian_decomposition import (
    LagrangianDecomposition,
    PrimalInformation,
    DualInformation,
    SeriesBundle,
)

from generic.optimization.model import Solution
from generic.series_dict import series_dict_to_array


class CombinedAlgorithm:
    configuration: AlgorithmConfiguration

    relaxation: LagrangianDecomposition
    dual_runner: DualAlgorithmRunner
    heuristic_algorithm: PrimalHeuristicAlgorithm

    relaxation_primal: Optional[PrimalInformation]
    relaxation_dual: Optional[DualInformation]
    heuristic_solution: Optional[PrimalInformation]

    master_bound: float
    bounds: BoundsTracker

    stop: bool
    converged: bool

    iteration: int

    current_multipliers: OrderedDict[str, pd.Series]

    dual_algorithm: str

    primal_solutions: List[Solution]
    dual_solutions: List[OrderedDict[str, pd.Series]]

    def __init__(
        self,
        configuration: AlgorithmConfiguration,
        relaxation: LagrangianDecomposition,
        initial_multipliers: OrderedDict[str, pd.Series],
        cp_algorithm_configuration: Dict[str, Any],
        sgd_algorithm_configuration: Dict[str, Any],
        heuristic_algorithm: PrimalHeuristicAlgorithm,
        bounds_tracker: BoundsTracker = None,
    ):
        self.configuration = configuration
        self.relaxation = relaxation
        self.dual_runner = DualAlgorithmRunner(
            self.configuration,
            cp_algorithm_configuration,
            sgd_algorithm_configuration,
            initial_multipliers,
        )
        self.heuristic_algorithm = heuristic_algorithm

        if bounds_tracker is not None:
            self.bounds = bounds_tracker
        else:
            self.bounds = BoundsTracker(sense=relaxation.sense)

        self.primal_solutions = []
        self.dual_solutions = []
        self.relaxation_dual = None
        self.relaxation_primal = None
        self.heuristic_solution = None
        self.iteration = -1
        self.converged = False
        self.stop = False

        self.dual_algorithm = "None"

        self.current_multipliers = initial_multipliers

        self.master_bound = -relaxation.sense * np.infty

    def __call__(self) -> bool:
        if not self.stop:
            self.iteration += 1

            ## Compute multipliers
            # except for first iteration, where multipliers are given
            if self.iteration > 0:
                (
                    self.master_bound,
                    self.current_multipliers,
                    self.dual_algorithm,
                ) = self.dual_runner.new_multipliers(self.relaxation_dual)

            ## Evaluate multipliers by solving the lagrangian subproblem and collect primal and dual solutions
            self.relaxation_primal, self.relaxation_dual = self.relaxation.evaluate(
                self.current_multipliers, **self.configuration.relaxation_solver_options
            )
            self.primal_solutions.append(self.relaxation_primal.solution)
            self.dual_solutions.append(self.current_multipliers)

            ## Update global bounds
            self.bounds.update_dual_bound(self.relaxation_dual)
            self.bounds.update_primal_bound(self.relaxation_primal)
            self.bounds.update_master_bound(self.master_bound)

            ## Maybe run primal heuristic
            if self.iteration > 0 and self.iteration % self.configuration.heuristic_frequency == 0:
                self.heuristic_solution, bundle = self.heuristic_algorithm(self.primal_solutions)
                if self.heuristic_solution is not None:
                    # exploit the solution
                    self.bounds.update_primal_bound(self.heuristic_solution)
                    self.primal_solutions.append(self.heuristic_solution.solution)
                    self.dual_runner.add_cut(bundle.intercept, bundle.subgradient)
            else:
                self.heuristic_solution = None

            ## Check convergence
            self.converged = self.check_convergence()
            self.stop = self.converged or self.iteration >= self.configuration.max_iterations

        return self.stop

    def check_convergence(self) -> bool:
        are_gaps_below_threshold = (
            self.bounds.optimality_gap < self.configuration.max_gap
            or self.bounds.cutting_plane_gap < self.configuration.max_cp_gap
        )
        vect_subgradient = series_dict_to_array(**self.relaxation_dual.bundle.subgradient)
        sgd_norm = np.linalg.norm(vect_subgradient)
        is_subgradient_small = sgd_norm <= self.configuration.subgradient_tolerance
        return are_gaps_below_threshold and is_subgradient_small
