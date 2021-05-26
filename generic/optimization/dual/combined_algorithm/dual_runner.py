from typing import Tuple, OrderedDict, Dict, Any
from numpy import ndarray, infty
from pandas import Series, Index

from generic.optimization.dual.algorithms.cutting_plane import CuttingPlane
from generic.optimization.dual.algorithms.subgradient import SubgradientMethod

from generic.optimization.dual.lagrangian_decomposition import DualInformation
from generic.optimization.dual.combined_algorithm.configuration import AlgorithmConfiguration
from generic.series_dict import series_dict_to_array, series_dict_indexes, full_series_dict, array_into_series_dict


class DualAlgorithmRunner:

    configuration: AlgorithmConfiguration

    cp_configuration: Dict[str, Any]
    sgd_configuration: Dict[str, Any]

    cp_alg: CuttingPlane
    sgd_alg: SubgradientMethod
    vect_multipliers: ndarray
    master_bound: float
    multipliers_dict_template: OrderedDict[str, Index]

    sgd_counter: int
    cp_counter: int

    def __init__(
        self,
        configuration: AlgorithmConfiguration,
        cp_configuration: Dict[str, Any],
        sgd_configuration: Dict[str, Any],
        initial_multipliers: OrderedDict[str, Series],
    ):
        self.configuration = configuration
        self.sgd_configuration = sgd_configuration

        self.sgd_flag = True
        self.sgd_counter = 0
        self.cp_counter = 0
        self.vect_multipliers = series_dict_to_array(**initial_multipliers)
        self.master_bound = infty * cp_configuration["sense"]
        self.multipliers_dict_template = series_dict_indexes(**initial_multipliers)

        self.sgd_configuration, self.cp_configuration = sgd_configuration, cp_configuration

        self.sgd_alg = SubgradientMethod(initial_solution=self.vect_multipliers.copy(), **self.sgd_configuration)
        self.cp_alg = CuttingPlane(**self.cp_configuration)

    def add_cut(self, intercept: float, subgradient: OrderedDict[str, Series]):
        vect_subgradient = series_dict_to_array(**subgradient)
        self.cp_alg.add_cut(intercept, vect_subgradient)

    def new_multipliers(self, dual_solution: DualInformation) -> Tuple[float, OrderedDict[str, Series], str]:
        vect_subgradient = series_dict_to_array(**dual_solution.bundle.subgradient)

        self.vect_multipliers, algorithm = self._compute_new_multipliers(
            dual_solution.objective, dual_solution.bundle.intercept, vect_subgradient
        )
        new_multipliers = full_series_dict(0.0, **self.multipliers_dict_template)
        array_into_series_dict(self.vect_multipliers, **new_multipliers)

        return self.master_bound, new_multipliers, algorithm

    def _compute_new_multipliers(
        self, value: float, intercept: float, vect_subgradient: ndarray
    ) -> Tuple[ndarray, str]:
        self._choose_dual_algorithm()
        if self.sgd_flag:
            new_vect_multipliers = self.sgd_alg(value, vect_subgradient)
            self.sgd_counter += 1
            algorithm = self.configuration.sgd_name
        else:
            self.master_bound, new_vect_multipliers = self.cp_alg(
                value, intercept, vect_subgradient, **self.configuration.cp_solver_options
            )
            self.cp_counter += 1
            algorithm = self.configuration.cp_name

        return new_vect_multipliers, algorithm

    def _choose_dual_algorithm(self):
        if self.sgd_counter >= self.configuration.sgd_iterations:
            # choose cutting plane
            self.sgd_counter = 0
            self.sgd_flag = False
        elif self.cp_counter >= self.configuration.cp_iterations:
            # choose SGD
            self.cp_counter = 0
            self.sgd_flag = True
            self.sgd_alg = SubgradientMethod(initial_solution=self.vect_multipliers, **self.sgd_configuration)
