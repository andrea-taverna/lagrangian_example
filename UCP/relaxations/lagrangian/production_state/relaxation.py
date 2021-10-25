import collections
from typing import Dict, Any, OrderedDict, Tuple

import numpy as np
import pandas as pd
from pulp import PULP_CBC_CMD, lpSum, LpStatusOptimal

from UCP.model import UCPData
from UCP.relaxations.lagrangian.production_state.subproblems import (
    make_single_UCP,
    make_economic_dispatch,
)
from generic.optimization.dual.lagrangian_decomposition import (
    LagrangianDecomposition,
    PrimalInformation,
    SeriesBundle,
    DualInformation,
)
from generic.optimization.model import MathematicalProgram, Solution, OptimizationSense
from generic.optimization.solution_aggregation import extract_and_aggregate_solutions
from generic.optimization.solution_extraction import extract_solution


class ProductionStateRelaxation(LagrangianDecomposition):
    """ "
    Implements a production/state lagrangian relaxation for the UCP
    """

    single_ucps: Dict[Any, MathematicalProgram]
    """ UCP subproblem for each single power plant"""

    econ_dispatch: MathematicalProgram
    """ Economic dispatch problem. Computes optimal production levels for power plants."""

    data: UCPData
    """ data for the UCP problem."""

    penalty_increase_factor: float
    """ How much should the multipliers' (absolute) bounds are larger than the maximum between per-unit cost 
        of ENP and EIE."""

    def __init__(self, data: UCPData, penalty_increase_factor: float = 1.5):
        """

        Args:
            data: data for UCP problem
            penalty_increase_factor: penalty increase factor for multipliers' range
        """
        super().__init__(sense=OptimizationSense.MIN)

        self.data = data
        self.single_ucps = {plant: make_single_UCP(data, plant) for plant in data.thermal_plants["plant"]}
        self.econ_dispatch = make_economic_dispatch(data)
        self.penalty_increase_factor = penalty_increase_factor

    def fill_multipliers(self, value: float = 0.0) -> OrderedDict[str, pd.Series]:
        """
        Creates multipliers data for the problem with the fixed value (default zero)
        Args:
            value: initial value for multipliers

        Returns:
            a series_dict for the multipliers with the provided value.
        """
        index = _multipliers_index(self.data)
        return collections.OrderedDict(
            [
                ("min_production", pd.Series(np.full(len(index), value), index=index)),
                ("max_production", pd.Series(np.full(len(index), value), index=index)),
            ]
        )

    def multipliers_range(
        self,
    ) -> Tuple[OrderedDict[str, pd.Series], OrderedDict[str, pd.Series]]:
        """
        Returns the range for the multipliers.
        Returns:
            a tuple (lower,upper) of series_dict containing for each multiplier the lower and upper value resp.
        """
        index = _multipliers_index(self.data)

        max_obj_cost = max(self.data.c_EIE, self.data.c_ENP)

        multipliers_lb = pd.Series(np.zeros(len(index)), index=index)
        multipliers_ub = pd.Series(
            np.full(len(index), max_obj_cost * self.penalty_increase_factor),
            index=index,
        )

        return (
            collections.OrderedDict([("min_production", multipliers_lb), ("max_production", multipliers_lb)]),
            collections.OrderedDict([("min_production", multipliers_ub), ("max_production", multipliers_ub)]),
        )

    def violations(self, solution: Solution) -> OrderedDict[str, pd.Series]:
        """
        Returns the violations for the provided solution.
        Args:
            solution: current solution to compute violations from

        Returns:
            a series_dict containing the violations for each dualized constraint
        """
        production = solution["p"]
        state = solution["s"]
        tpp = self.data.thermal_plants.set_index(["plant"])

        return collections.OrderedDict(
            [
                ("min_production", state * tpp["min_power"] - production),
                ("max_production", production - state * tpp["max_power"]),
            ]
        )

    def infeasibilities(self, solution) -> OrderedDict[str, pd.Series]:
        """
        Returns the infeasibilities for the provided solution.
        Args:
            solution: solution to compute infeasibilities for

        Returns:
            a series_dict with infeasibilities for each dualized constraint
        """
        violations = self.violations(solution)
        infeasibilities = collections.OrderedDict(
            [(k, v.transform(lambda x: max(0.0, x))) for k, v in violations.items()]
        )
        return infeasibilities

    def evaluate(self, multipliers: OrderedDict[str, pd.Series], **kwargs) -> Tuple[PrimalInformation, DualInformation]:
        """
        Returns the value of the Lagrangian Function (LF) for given multipliers
        Args:
            multipliers: multipliers to evaluate
            **kwargs: solver options

        Returns:
            a tuple containing primal and dual information for the current LF evaluation
        """
        self._set_multipliers(multipliers)

        for problem in self.single_ucps.values():
            status = problem.model.solve(PULP_CBC_CMD(msg=0,**kwargs))
            assert status == LpStatusOptimal

        status = self.econ_dispatch.solve(PULP_CBC_CMD(msg=0,**kwargs))
        assert status == LpStatusOptimal

        solution, value = self._extract_solution_kpis()

        original_objective = self._original_objective(solution)
        intercept = self.linearization_intercept(solution)

        infeasibilities = self.infeasibilities(solution)
        subgradient = self.subgradient(solution)

        primal_information = PrimalInformation(
            solution=solution,
            objective=original_objective,
            infeasibilities=infeasibilities,
        )
        dual_information = DualInformation(
            solution=multipliers,
            objective=value,
            bundle=SeriesBundle(intercept=intercept, subgradient=subgradient),
        )
        return primal_information, dual_information

    def _set_multipliers(self, multipliers: Dict[str, pd.Series]):
        """
        Set multipliers in each subproblem's objective function.
        Args:
            multipliers: multipliers to set
        """
        production_up = multipliers["max_production"]
        production_low = multipliers["min_production"]
        TPP = self.data.thermal_plants.set_index("plant")

        dual_coef = {
            (plant, t): production_low[plant, t] * TPP.loc[plant, "min_power"]
            - production_up[plant, t] * TPP.loc[plant, "max_power"]
            for plant, t in production_low.keys()
        }
        for plant, problem in self.single_ucps.items():
            s = problem.vars["s"]
            commitment_cost = problem.exprs["commitment_cost"]
            problem.model.setObjective(
                commitment_cost + lpSum(dual_coef[plant, t] * s[t] for t in self.data.loads["period"])
            )

        p = self.econ_dispatch.vars["p"]
        dispatch_cost = self.econ_dispatch.exprs["dispatch_cost"]
        self.econ_dispatch.model.setObjective(
            dispatch_cost
            + lpSum(
                (production_up[plant, t] - production_low[plant, t]) * p[plant, t]
                for (plant, t) in production_low.keys()
            )
        )

    def _extract_solution_kpis(self) -> Tuple[Solution, float]:
        """
        Extract the solution KPIs (value of variables + LF value
        Returns:
            a tuple (Solution, LF_value)
        """
        ucp_solution = extract_and_aggregate_solutions(self.single_ucps, id_name=["plant"])

        edp_solution = extract_solution(self.econ_dispatch)

        assert len(set(ucp_solution.keys()).intersection(edp_solution.keys())) == 0
        solution = {**ucp_solution, **edp_solution}

        single_ucps_obj = sum(problem.model.objective.value() for problem in self.single_ucps.values())
        LF_value = single_ucps_obj + self.econ_dispatch.model.objective.value()

        return solution, LF_value

    def _original_objective(self, solution: Solution) -> float:
        """
        Returns the "original objective", i.e. the objective value of the original UCP problem, from the solution of
        the lagrangian relaxation.
        Args:
            solution: solution to compute the original objective for

        Returns:
            value of the original objective
        """
        return solution["commitment_cost"].sum() + solution["dispatch_cost"]

    def linearization_intercept(self, solution: Solution) -> float:
        """
        Returns the intercept of the linear approximation/cutting plane/bundle of the lagrangian function at the solution.
        Usually equal to the `original_objective`. There may be corner cases where it is computed differently.

        Args:
            solution: solution to compute the linearization intercept

        Returns:
            linearization intercept.
        """
        return self._original_objective(solution)

    def information_from_primal_solution(self, solution: Solution, **kwargs) -> Tuple[PrimalInformation, SeriesBundle]:
        """
        Computes primal and dual information from a solution **to the original problem**

        Args:
            solution: solution to the original UCP formulation to compute information from

        Returns:
        Primal and Dual information as in `evaluate`.
        """
        primal_info = PrimalInformation(
            solution=solution,
            objective=solution["total_production_cost"],
            infeasibilities=self.infeasibilities(solution),
        )
        bundle = SeriesBundle(
            intercept=solution["total_production_cost"],
            subgradient=self.subgradient(solution),
        )
        return primal_info, bundle


def _multipliers_index(data: UCPData) -> pd.Index:
    periods = data.loads["period"].to_list()
    plants = data.thermal_plants["plant"].to_list()

    return pd.MultiIndex.from_product([plants, periods], names=["period", "plant"])
