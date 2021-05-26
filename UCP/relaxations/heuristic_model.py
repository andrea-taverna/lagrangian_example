from itertools import product
from typing import List

import pandas as pd
from pulp import LpProblem, LpBinary, lpSum, LpMinimize, LpVariable

from UCP.data import UCPData
from generic.optimization.constraint_adder import add_constraint
from generic.optimization.model import MathematicalProgram, OptimizationSense

index_names = dict(
    commit=["plant", "commitment_id"],
    p=["plant", "period"],
    EIE=["period"],
    ENP=["period"],
    up=["plant", "period"],
    dn=["plant", "period"],
)


def create_model(data: UCPData, commitments: List[pd.Series]) -> MathematicalProgram:
    """
    Creates a model for the combinatorial heuristic.
    :param data: data of the UCP problem
    :param commitments: commitments as a list of plant schedules
    :return: a model for the combinatorial heuristic.
    """
    TPP = data.thermal_plants
    Plants = TPP["plant"].to_list()
    Time = data.loads["period"].values
    Commitments = list(range(len(commitments)))

    model = LpProblem("UCP", sense=LpMinimize)

    ### VARIABLES
    p = {
        (plant, t): LpVariable(f"p_{plant}_{t}", lowBound=0, upBound=max_power)
        for ((plant, max_power), t) in product(TPP[["plant", "max_power"]].itertuples(index=False), Time)
    }

    commit = LpVariable.dict("commit", (Plants, Commitments), cat=LpBinary)

    EIE = LpVariable.dict("EIE", Time, lowBound=0)
    ENP = LpVariable.dict("ENP", Time, lowBound=0)

    commit_sum = {
        (plant, t): lpSum(commitments[com_id][plant, t] * commit[plant, com_id] for com_id in Commitments)
        for plant, t in product(Plants, Time)
    }

    # see https://github.com/coin-or/pulp/issues/331#issuecomment-681965566
    for k, v in commit.items():
        v.setInitialValue(0)

    commitment_choice = {
        plant: add_constraint(model, lpSum(commit[plant, com_id] for com_id in Commitments) <= 1, f"com_choice{plant}")
        for plant in Plants
    }

    max_production = {
        (plant, t): add_constraint(
            model, p[plant, t] <= max_power * commit_sum[plant, t], f"def_max_production_{plant}_{t}"
        )
        for ((plant, max_power), t) in product(TPP[["plant", "max_power"]].itertuples(index=False), Time)
        if t > 0
    }

    min_production = {
        (plant, t): add_constraint(
            model, p[plant, t] >= min_power * commit_sum[plant, t], f"def_min_production_{plant}_{t}"
        )
        for ((plant, min_power), t) in product(TPP[["plant", "min_power"]].itertuples(index=False), Time)
        if t > 0
    }

    demand_satisfaction = {
        t: add_constraint(
            model,
            lpSum(p[plant, t] for plant in Plants) + ENP[t] - EIE[t] == data.loads["value"][t],
            f"demand_satisfaction_{t}",
        )
        for t in Time
    }

    ### OBJECTIVE
    thermal_production_cost = lpSum(
        l_cost * p[plant, t] + c_cost * commit_sum[plant, t]
        for ((plant, l_cost, c_cost), t) in product(TPP[["plant", "l_cost", "c_cost"]].itertuples(index=False), Time)
    )

    total_production_cost = thermal_production_cost + lpSum(data.c_EIE * EIE[t] + data.c_ENP * ENP[t] for t in Time)

    model.setObjective(total_production_cost)

    ### MODEL DICT
    ucp = MathematicalProgram(
        sense=OptimizationSense.MIN,
        vars=dict(commit=commit, p=p, EIE=EIE, ENP=ENP),
        cons=dict(
            max_production=max_production,
            min_production=min_production,
            demand_satisfaction=demand_satisfaction,
            commitment_choice=commitment_choice,
        ),
        objs=dict(total_production_cost=total_production_cost),
        model=model,
        index_names=index_names,
    )

    return ucp
