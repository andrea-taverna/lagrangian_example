from itertools import product
from typing import List, NamedTuple
import numpy as np

from UCP.data import UCPData
from UCP.stochastic.scen_gen import ScenarioInfo
from generic.optimization.fix_variables import fix_variables, unfix_variables
from generic.optimization.model import Solution, MathematicalProgram
from pulp import PULP_CBC_CMD

from generic.optimization.solution_extraction import extract_solution

StochasticEvaluation = NamedTuple("StochasticEvaluation", (("expected_cost", float), ("cost_by_scenario", np.ndarray)))


def evaluate_solution(
    data: UCPData, scenarios: List[ScenarioInfo], solution: Solution, model: MathematicalProgram, **solver_options
) -> StochasticEvaluation:
    # fix commitments of solution
    previous_bounds = fix_variables({"s": model.vars["s"]}, {"s": solution["s"]})
    model.solve(PULP_CBC_CMD(msg=0, **solver_options))
    evaluation = extract_solution(model)
    unfix_variables(model, previous_bounds)

    Scenarios = list(range(len(scenarios)))
    Time = data.loads["period"]
    TPP = data.thermal_plants

    scen_costs = np.array(
        [
            sum(
                sum(
                    l_cost * evaluation["p"][plant, t, scen] + c_cost * evaluation["s"][plant, t]
                    for (plant, l_cost, c_cost) in TPP[["plant", "l_cost", "c_cost"]].itertuples(index=False)
                )
                + (data.c_EIE * evaluation["EIE"][t, scen] + data.c_ENP * evaluation["ENP"][t, scen])
                for t in Time
            )
            for scen in Scenarios
        ]
    )

    return StochasticEvaluation(expected_cost=evaluation["total_production_cost"], cost_by_scenario=scen_costs)
