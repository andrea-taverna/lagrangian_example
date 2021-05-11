import pandas as pd
from typing import Tuple, Dict, Any

from pulp import lpSum

from UCP.input.data import UCPData
from UCP.model import ucp
from UCP.relaxations.lagrangian.production_state.subproblems import make_economic_dispatch, make_single_UCP
from generic.optimization.constraint_adder import add_constraint
from generic.optimization.model import MathematicalProgram, Solution
from generic.optimization.solution_extraction import extract_solution


def commit_dispatch_heuristic(
    data: UCPData, eco_dispatch_options: Dict[str, Any] = None, commit_options: Dict[str, Any] = None
) -> Tuple[MathematicalProgram, Solution]:
    eco_dispatch_options = {} if eco_dispatch_options is None else eco_dispatch_options
    commit_options = {} if commit_options is None else commit_options

    eco_dispatch = make_economic_dispatch(data)
    eco_dispatch.model.setObjective(eco_dispatch.exprs["dispatch_cost"])
    eco_dispatch.solve(**eco_dispatch_options)
    ed_solution = extract_solution(eco_dispatch)

    commit = make_commitment_problems(data, ed_solution["p"])

    for model in commit.values():
        model.solve(**commit_options)

    commitments = {plant: extract_solution(model)["s"] for plant, model in commit.items()}

    final_model = ucp.create_model(data)
    _fix_commitments(data, final_model, commitments)

    final_model.solve()
    solution = extract_solution(final_model)
    return final_model, solution


def make_commitment_problems(
    data: UCPData, production_levels: pd.Series, min_production_rate=0
) -> Dict[int, MathematicalProgram]:
    TPP = data.thermal_plants
    Time = data.loads["period"]
    single_ucps: Dict[int, MathematicalProgram] = {plant: make_single_UCP(data, plant) for plant in TPP["plant"]}

    for plant, model in single_ucps.items():
        plant_data = data.thermal_plants[data.thermal_plants["plant"] == plant].iloc[0]
        lp_model = model.model
        s = model.vars["s"]

        # Satisfy given production levels
        def_satisfy_production = {
            t: add_constraint(lp_model,
                plant_data["max_power"] * s[t] >= production_levels[plant, t], f"def_sat_production_{plant}_{t}"
            )
            for t in Time
            if production_levels[plant, t] > min_production_rate * plant_data["min_power"]
        }
        model.cons.update(def_satisfy_production=def_satisfy_production)

        ### OBJECTIVE
        commitment_cost = lpSum(plant_data["c_cost"] * s[t] for t in Time)

        lp_model.setObjective(commitment_cost)

    return single_ucps


def _fix_commitments (data:UCPData, model:MathematicalProgram, selected_commitments:Dict[int, pd.Series]):
    for plant, commitment in selected_commitments.items():
        for time in data.loads["period"]:
            model.model.addConstraint(model.vars["s"][plant, time] == selected_commitments[plant][time])
