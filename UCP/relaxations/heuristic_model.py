from itertools import product
from typing import Dict, Any, List

import pandas as pd
from pulp import LpProblem, LpBinary, lpSum, LpMinimize, LpVariable

from UCP.input.data import UCPData
from generic.optimization.constraint_adder import add_constraint
from generic.optimization.model import MathematicalProgram, Solution, OptimizationSense

index_names = dict(
    pat=["plant", "pattern_id"],
    p=["plant", "period"],
    EIE=["period"],
    ENP=["period"],
    up=["plant", "period"],
    dn=["plant", "period"],
)


def create_model(data: UCPData, patterns: List[pd.Series]) -> MathematicalProgram:
    TPP = data.thermal_plants
    Plants = TPP["plant"].to_list()
    Time = data.loads["period"].values
    Patterns = list(range(len(patterns)))

    model = LpProblem("UCP", sense=LpMinimize)

    ### VARIABLES
    p = {
        (plant, t): LpVariable(f"p_{plant}_{t}", lowBound=0, upBound=max_power)
        for ((plant, max_power), t) in product(TPP[["plant", "max_power"]].itertuples(index=False), Time)
    }

    pat = LpVariable.dict("pat", (Plants, Patterns), cat=LpBinary)

    EIE = LpVariable.dict("EIE", Time, lowBound=0)
    ENP = LpVariable.dict("ENP", Time, lowBound=0)

    pat_sum = {
        (plant, t): lpSum(patterns[pat_id][plant, t] * pat[plant, pat_id] for pat_id in Patterns)
        for plant, t in product(Plants, Time)
    }

    # see https://github.com/coin-or/pulp/issues/331#issuecomment-681965566
    for k, v in pat.items():
        v.setInitialValue(0)

    pattern_choice = {
        plant: add_constraint(model, lpSum(pat[plant, pat_id] for pat_id in Patterns) <= 1, f"pat_choice{plant}")
        for plant in Plants
    }

    max_production = {
        (plant, t): add_constraint(
            model, p[plant, t] <= max_power * pat_sum[plant, t], f"def_max_production_{plant}_{t}"
        )
        for ((plant, max_power), t) in product(TPP[["plant", "max_power"]].itertuples(index=False), Time)
        if t > 0
    }

    min_production = {
        (plant, t): add_constraint(
            model, p[plant, t] >= min_power * pat_sum[plant, t], f"def_min_production_{plant}_{t}"
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
        l_cost * p[plant, t] + c_cost * pat_sum[plant, t]
        for ((plant, l_cost, c_cost), t) in product(TPP[["plant", "l_cost", "c_cost"]].itertuples(index=False), Time)
    )

    total_production_cost = thermal_production_cost + lpSum(data.c_EIE * EIE[t] + data.c_ENP * ENP[t] for t in Time)

    model.setObjective(total_production_cost)

    ### MODEL DICT
    ucp = MathematicalProgram(
        sense=OptimizationSense.MIN,
        vars=dict(pat=pat, p=p, EIE=EIE, ENP=ENP),
        cons=dict(
            max_production=max_production,
            min_production=min_production,
            demand_satisfaction=demand_satisfaction,
            pattern_choice=pattern_choice,
        ),
        objs=dict(total_production_cost=total_production_cost),
        model=model,
        index_names=index_names,
    )

    return ucp