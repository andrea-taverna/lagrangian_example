import collections
from dataclasses import dataclass
from typing import Dict, Any, OrderedDict, Tuple

import numpy as np
import pandas as pd
from pulp import lpSum, PULP_CBC_CMD, LpStatus

from UCP.input.data import UCPData
from UCP.relaxations.lagrangian.demand_satisfaction.single_ucp import make_single_UCP
from UCP.model.ucp import create_model
from generic.optimization.dual.dualized_constraints import DualizedConstraint, ConstraintSense
from generic.optimization.dual.lagrangian_decomposition import (
    LagrangianDecomposition,
    PrimalInformation,
    SeriesBundle,
    DualInformation,
)
from generic.optimization.model import MathematicalProgram, OptimizationSense, Solution
from generic.optimization.solution_aggregation import extract_and_aggregate_solutions
from generic.optimization.solution_extraction import compute_multipliers
from generic.series_dict import full_series_dict


@dataclass
class DemandSatRelaxation(LagrangianDecomposition):
    subproblems: Dict[Any, MathematicalProgram]
    data: UCPData

    def __init__(self, data: UCPData, penalty_increase_factor: float = 3):
        dualized_cons = _make_dualized_constraints(data, penalty_increase_factor=penalty_increase_factor)
        super().__init__(sense=OptimizationSense.MIN, dualized_constraints=dualized_cons)
        self.data = data
        self.subproblems = {plant: make_single_UCP(data, plant) for plant in data.thermal_plants["plant"]}

    def lhs(self, solution: Solution) -> OrderedDict[str, pd.Series]:
        return collections.OrderedDict({"demand_satisfaction": solution["p"].groupby(level=["period"]).sum()})

    def rhs(self) -> OrderedDict[str, pd.Series]:
        return collections.OrderedDict(
            {"demand_satisfaction": self.data.loads[["period", "value"]].set_index("period")["value"]}
        )

    def evaluate(self, multipliers: OrderedDict[str, pd.Series], **kwargs) -> Tuple[PrimalInformation, DualInformation]:
        self._read_multipliers(multipliers)
        for name, problem in self.subproblems.items():
            problem.model.writeLP(f"problem_{name}.lp")
            problem.solve(PULP_CBC_CMD(**kwargs))

        return self._extract_solution_kpis(multipliers)

    def _read_multipliers(self, multipliers: Dict[str, pd.Series]):
        dem_sat_multipliers = multipliers["demand_satisfaction"]
        for plant, problem in self.subproblems.items():
            p = problem.vars["p"]
            thermal_prod_cost = problem.exprs["thermal_production_cost"]
            problem.model.setObjective(
                thermal_prod_cost + lpSum(dem_sat_multipliers[t] * p[t] for t in self.data.loads["period"])
            )

    def _extract_solution_kpis(
        self, multipliers: OrderedDict[str, pd.Series]
    ) -> Tuple[PrimalInformation, DualInformation]:
        solution = extract_and_aggregate_solutions(self.subproblems, id_name=["plant"])

        dem_sat_multipliers = multipliers["demand_satisfaction"]
        loads = self.data.loads
        single_ucps_obj = sum(problem.model.objective.value() for problem in self.subproblems.values())

        value = single_ucps_obj - sum(dem_sat_multipliers[t] * loads["value"][t] for t in self.data.loads["period"])
        original_objective = self._original_objective(solution)

        infeasibilities = self.infeasibilities(solution)
        intercept = self.linearization_intercept(solution)
        subgradient = self.subgradient(solution)

        primal_info = PrimalInformation(
            solution=solution, objective=original_objective, infeasibilities=infeasibilities
        )
        dual_info = DualInformation(
            objective=value, solution=multipliers, bundle=SeriesBundle(intercept=intercept, subgradient=subgradient)
        )
        return primal_info, dual_info

    def linearization_intercept(self, solution: Solution) -> float:
        return solution["thermal_production_cost"].sum()

    def _original_objective(self, solution: Solution) -> float:
        tp_cost = solution["thermal_production_cost"].sum()
        violations = self.violations(solution)["demand_satisfaction"]
        tot_ENP = violations[violations > 0].sum()
        tot_EIE = -violations[violations < 0].sum()

        return tp_cost + self.data.c_ENP * tot_ENP + self.data.c_EIE * tot_EIE

    def information_from_primal_solution(self, solution: Solution, **kwargs) -> Tuple[PrimalInformation, SeriesBundle]:
        primal_info = PrimalInformation(
            solution=solution,
            objective=solution["total_production_cost"],
            infeasibilities=self.infeasibilities(solution),
        )
        bundle = SeriesBundle(intercept=solution["thermal_production_cost"], subgradient=self.subgradient(solution))
        return primal_info, bundle

    def infeasibilities(self, solution) -> OrderedDict[str, pd.Series]:
        return full_series_dict(0.0, **{k: dc.index for k, dc in self.dualized_constraints.items()})

    def multipliers_from_primal_solution(self, solution, **kwargs) -> OrderedDict[str, pd.Series]:
        model = create_model(self.data)
        multipliers = compute_multipliers(model, solution)
        return collections.OrderedDict([(k, multipliers[k]) for k in self.dualized_constraints.keys()])


def _make_dualized_constraints(
    data: UCPData, penalty_increase_factor: float = 3
) -> OrderedDict[str, DualizedConstraint]:
    index = pd.Index(data.loads["period"], name="period")

    multipliers_lb = pd.Series(np.full(len(index), -penalty_increase_factor * data.c_EIE), index=index)
    multipliers_ub = pd.Series(np.full(len(index), penalty_increase_factor * data.c_ENP), index=index)

    return collections.OrderedDict(
        demand_satisfaction=DualizedConstraint(
            "demand_satisfaction", ConstraintSense.EQUAL, multipliers_lb, multipliers_ub
        )
    )
