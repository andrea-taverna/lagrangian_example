import operator
from itertools import groupby
from typing import List, Dict, Tuple, Any
import pandas as pd
from pulp import PULP_CBC_CMD

from UCP.input.data import UCPData
import UCP.model.ucp as ucp_model
from generic.optimization.model import MathematicalProgram
from UCP.relaxations import heuristic_model


def combinatorial_heuristic(
    data: UCPData,
    commitments: List[pd.Series],
    flexibility_limit=(1, 1),
    combination_options: Dict[str, Any] = {},
    final_solution_options: Dict[str, Any] = {},
) -> Tuple[MathematicalProgram, MathematicalProgram]:
    model = heuristic_model.create_model(data, commitments)
    model.solve(PULP_CBC_CMD(**combination_options))

    pat_values = [
        (plant, pat, v.value()) for (plant, pat), v in model.vars["pat"].items()
    ]
    pat_values = sorted(pat_values, key=operator.itemgetter(0))
    grouper = groupby(pat_values, operator.itemgetter(0))
    pats_per_plants = {plant: list(group) for plant, group in grouper}

    plants_to_fix = {
        plant
        for plant, min_on, min_off in data.thermal_plants[
            ["plant", "min_on", "min_off"]
        ].itertuples(index=False)
        if min_off > flexibility_limit[0] or min_on > flexibility_limit[1]
    }
    selected_pats = {
        plant: max(pats, key=operator.itemgetter(2))
        for plant, pats in pats_per_plants.items()
        if plant in plants_to_fix
    }
    selected_commitments = {
        plant: commitments[pat[1]] for plant, pat in selected_pats.items()
    }
    final_model = ucp_model.create_model(data)
    _fix_commitments(data, final_model, selected_commitments)
    status = final_model.solve(PULP_CBC_CMD(**final_solution_options))
    return final_model, model


def _fix_commitments(
    data: UCPData,
    model: MathematicalProgram,
    selected_commitments: Dict[int, pd.Series],
):
    for plant, commitment in selected_commitments.items():
        for time in data.loads["period"]:
            model.model.addConstraint(
                model.vars["s"][plant, time] == selected_commitments[plant][plant, time]
            )
