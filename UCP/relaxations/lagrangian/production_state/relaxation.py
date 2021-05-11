import collections
from typing import Dict, Any, OrderedDict, Tuple

import numpy as np
import pandas as pd
from pulp import PULP_CBC_CMD, lpSum, LpStatus, LpStatusOptimal

from UCP.model.ucp import UCPData, create_model
from UCP.relaxations.lagrangian.production_state.subproblems import make_single_UCP, make_economic_dispatch
from generic.optimization.dual.dualized_constraints import DualizedConstraint, ConstraintSense
from generic.optimization.dual.lagrangian_decomposition import (
    LagrangianDecomposition,
    PrimalInformation,
    SeriesBundle,
    DualInformation,
)
from generic.optimization.model import MathematicalProgram, Solution, OptimizationSense
from generic.optimization.solution_aggregation import extract_and_aggregate_solutions
from generic.optimization.solution_extraction import extract_solution, compute_multipliers


class ProductionStateRelaxation(LagrangianDecomposition):
    single_ucps: Dict[Any, MathematicalProgram]
    econ_dispatch: MathematicalProgram
    data: UCPData
    dualized_constraints: OrderedDict[str, DualizedConstraint]

    def __init__(self, data: UCPData, penalty_increase_factor: float = 1.5):
        dualized_cons = _make_dualized_constraints(data, penalty_increase_factor=penalty_increase_factor)
        super().__init__(sense=OptimizationSense.MIN, dualized_constraints=dualized_cons)

        self.data = data
        self.single_ucps = {plant: make_single_UCP(data, plant) for plant in data.thermal_plants["plant"]}
        self.econ_dispatch = make_economic_dispatch(data)

    def lhs(self, solution: Solution) -> OrderedDict[str, pd.Series]:
        production = solution["p"]
        state = solution["s"]
        tpp = self.data.thermal_plants.set_index(["plant"])

        production_low = production - state * tpp["min_power"]
        production_up = production - state * tpp["max_power"]

        return collections.OrderedDict({"min_production": production_low, "max_production": production_up})

    def rhs(self) -> OrderedDict[str, pd.Series]:
        index = self.dualized_constraints["min_production"].index
        rhs_value = pd.Series(np.zeros(len(index)), index=index)
        return collections.OrderedDict(min_production=rhs_value, max_production=rhs_value.copy())

    def evaluate(self, multipliers: OrderedDict[str, pd.Series], **kwargs) -> Tuple[PrimalInformation, DualInformation]:
        self._set_multipliers(multipliers)

        for plant, problem in self.single_ucps.items():
            status = problem.model.solve(PULP_CBC_CMD(**kwargs))
            assert status == LpStatusOptimal

        status = self.econ_dispatch.solve(PULP_CBC_CMD(**kwargs))
        assert status == LpStatusOptimal

        solution, value = self._extract_solution_kpis()

        original_objective = self._original_objective(solution)
        intercept = self.linearization_intercept(solution)

        infeasibilities = self.infeasibilities(solution)
        subgradient = self.subgradient(solution)

        primal_information = PrimalInformation(
            solution=solution, objective=original_objective, infeasibilities=infeasibilities
        )
        dual_information = DualInformation(
            solution=multipliers, objective=value, bundle=SeriesBundle(intercept=intercept, subgradient=subgradient)
        )
        return primal_information, dual_information

    def _set_multipliers(self, multipliers: Dict[str, pd.Series]):
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
        ucp_solution = extract_and_aggregate_solutions(self.single_ucps, id_name=["plant"])

        edp_solution = extract_solution(self.econ_dispatch)

        assert len(set(ucp_solution.keys()).intersection(edp_solution.keys())) == 0
        solution = {**ucp_solution, **edp_solution}

        single_ucps_obj = sum(problem.model.objective.value() for problem in self.single_ucps.values())
        objective = single_ucps_obj + self.econ_dispatch.model.objective.value()

        return solution, objective

    def _original_objective(self, solution: Solution) -> float:
        return solution["commitment_cost"].sum() + solution["dispatch_cost"]

    def linearization_intercept(self, solution: Solution) -> float:
        return self._original_objective(solution)

    def information_from_primal_solution(self, solution:Solution, **kwargs) -> Tuple[PrimalInformation, SeriesBundle]:
        primal_info = PrimalInformation(
            solution=solution,
            objective=solution["total_production_cost"],
            infeasibilities=self.infeasibilities(solution),
        )
        bundle = SeriesBundle(intercept=solution["total_production_cost"], subgradient=self.subgradient(solution))
        return primal_info, bundle

    def multipliers_from_primal_solution(self, solution, **kwargs) -> OrderedDict[str, pd.Series]:
        model = create_model(self.data)
        multipliers = compute_multipliers(model, solution)
        return collections.OrderedDict([(k, multipliers[k]) for k in self.dualized_constraints.keys()])


def _make_dualized_constraints(data: UCPData, penalty_increase_factor: float) -> OrderedDict[str, DualizedConstraint]:
    periods = data.loads["period"].to_list()
    plants = data.thermal_plants["plant"].to_list()

    index = pd.MultiIndex.from_product([plants, periods], names=["period", "plant"])

    max_obj_cost = max(data.c_EIE, data.c_ENP)

    multipliers_lb = pd.Series(np.zeros(len(index)), index=index)
    multipliers_ub = pd.Series(np.full(len(index), max_obj_cost * penalty_increase_factor), index=index)

    return collections.OrderedDict(
        [
            (
                "min_production",
                DualizedConstraint("min_production", ConstraintSense.GREATER_THAN, multipliers_lb, multipliers_ub),
            ),
            (
                "max_production",
                DualizedConstraint("max_production", ConstraintSense.LESS_THAN, multipliers_lb, multipliers_ub),
            ),
        ]
    )
