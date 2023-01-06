import operator
from itertools import groupby
from typing import List, Dict, Any
import pandas as pd
from pulp import PULP_CBC_CMD, LpVariable, LpStatusOptimal

from UCP.data import UCPData
import UCP.model as ucp_model
from generic.optimization.model import MathematicalProgram, Solution
from UCP.lagrangian_relaxation import heuristic_model
from generic.optimization.solution_extraction import extract_solution


def combinatorial_heuristic(
    data: UCPData,
    commitments: List[pd.Series],
    flexibility_limit=(1, 1),
    combination_options: Dict[str, Any] = {},
    final_model_options: Dict[str, Any] = {},
) -> Solution:
    """
    Combinatorial heuristic for UCP.
    Given a collection of commitments for each plant:
        1. it uses a "combinatorial_heuristic" MIP to select/rank commitments for each plant
        2. It creates the original UCP MIP and fixes some of the power plants schedules to the selected/highest ranked
           commitment found at the previous step.
           It does not fix the plants whose flexibility (min_off, min_on values) is lower or equal than a
           "flexiblity_limit".
           The idea is not to fix the most flexible plants, which can be switched on/off easily to improve the solution.
        3. It then optimizes the original MIP, with some of the flexible plants fixed.

    :param data: UCP data
    :param commitments: list of plant commitments, as a series of binary states for each plant and time period
    :param flexibility_limit: touple (min_off,min_on).
           Plants having (min_off, min_off) above the limit are fixed according to the combinatorial heuristic
           selection/ranking.
    :param combination_options: solver options for the combination problem
    :param final_model_options: solver options for the original problem with selected commitments.
    :return: a solution to the original problem
    """

    # create the combination model
    model = heuristic_model.create_model(data, commitments)
    model.solve(PULP_CBC_CMD(msg=0,**combination_options))

    # extract the
    commit_values = [(plant, com, v.value()) for (plant, com), v in model.vars["commit"].items()]
    commit_values = sorted(commit_values, key=operator.itemgetter(0))
    grouper = groupby(commit_values, operator.itemgetter(0))
    pats_per_plants = {plant: list(group) for plant, group in grouper}

    plants_to_fix = {
        plant
        for plant, min_on, min_off in data.thermal_plants[["plant", "min_on", "min_off"]].itertuples(index=False)
        if min_off > flexibility_limit[0] or min_on > flexibility_limit[1]
    }
    selected_pats = {
        plant: max(pats, key=operator.itemgetter(2))
        for plant, pats in pats_per_plants.items()
        if plant in plants_to_fix
    }
    selected_commitments = {plant: commitments[pat[1]] for plant, pat in selected_pats.items()}
    final_model = ucp_model.create_model(data)
    _fix_commitments(data, final_model, selected_commitments)
    status = final_model.solve(PULP_CBC_CMD(msg=0,**final_model_options))
    assert status == LpStatusOptimal, "Final model in combinatorial heuristic not solved optimally."
    return extract_solution(final_model)


def _fix_commitments(
    data: UCPData,
    model: MathematicalProgram,
    selected_commitments: Dict[int, pd.Series],
):
    for plant, commitment in selected_commitments.items():
        for time in data.loads["period"]:
            state: LpVariable = model.vars["s"][plant, time]
            state.upBound = selected_commitments[plant][plant, time]
            state.lowBound = selected_commitments[plant][plant, time]
