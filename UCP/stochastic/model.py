from itertools import product
from typing import List

from pulp import LpProblem, LpBinary, lpSum, LpMinimize, LpVariable

from UCP.data import UCPData
from UCP.stochastic.scen_gen import ScenarioInfo
from generic.optimization.constraint_adder import add_constraint
from generic.optimization.model import MathematicalProgram, OptimizationSense

index_names = dict(
    s=["plant", "period"],
    p=["plant", "period", "scenario"],
    EIE=["period", "scenario"],
    ENP=["period", "scenario"],
    up=["plant", "period"],
    dn=["plant", "period"],
)


def create_model(data: UCPData, scenarios: List[ScenarioInfo]) -> MathematicalProgram:
    """
    Creates a UCP model.
    Args:
        data: UCP data

    Returns:
        a mathematical program for the UCP
    """
    TPP = data.thermal_plants
    Plants = TPP["plant"].to_list()
    Time = data.loads["period"].values
    Scenarios = list(range(len(scenarios)))

    model = LpProblem("UCP", sense=LpMinimize)

    ### VARIABLES
    p = {
        (plant, t, scen): LpVariable(f"p_{plant}_{t}_{scen}", lowBound=0)
        for (plant, t, scen) in product(Plants, Time, Scenarios)
    }

    s = LpVariable.dict("s", (Plants, Time), cat=LpBinary)
    up = LpVariable.dict("up", (Plants, Time), cat=LpBinary)
    dn = LpVariable.dict("dn", (Plants, Time), cat=LpBinary)

    EIE = {(t, scen):  LpVariable(f"EIE_{t}_{scen}", lowBound=0) for (t,scen) in product(Time, Scenarios)}
    ENP = {(t, scen):  LpVariable(f"ENP_{t}_{scen}", lowBound=0) for (t,scen) in product(Time, Scenarios)}

    def_up = {
        (plant, t): add_constraint(model, up[plant, t] >= s[plant, t] - s[plant, t - 1], f"def_up_{plant}_{t}")
        for (plant, t) in product(Plants, Time)
        if t > 0
    }

    def_down = {
        (plant, t): add_constraint(model, dn[plant, t] >= s[plant, t - 1] - s[plant, t], f"def_dn_{plant}_{t}")
        for (plant, t) in product(Plants, Time)
        if t > 0
    }

    def_up_zero = {
        (plant, t): add_constraint(model, up[plant, t] <= s[plant, t], f"def_up_zero_{plant}_{t}")
        for (plant, t) in product(Plants, Time)
    }

    def_down_zero = {
        (plant, t): add_constraint(model, dn[plant, t] <= 1 - s[plant, t], f"def_dn_zero_{plant}_{t}")
        for (plant, t) in product(Plants, Time)
    }

    def_min_on = {
        (plant, t): add_constraint(
            model, s[plant, t] >= lpSum(up[plant, t1] for t1 in range(max(0, t - min_on), t)), f"def_min_on_{plant}_{t}"
        )
        for ((plant, min_on), t) in product(TPP[["plant", "min_on"]].itertuples(index=False), Time)
        if t > 0
    }

    def_min_off = {
        (plant, t): add_constraint(
            model,
            s[plant, t] <= 1 - lpSum(dn[plant, t1] for t1 in range(max(0, t - min_power), t)),
            f"def_min_off_{plant}_{t}",
        )
        for ((plant, min_power), t) in product(TPP[["plant", "min_off"]].itertuples(index=False), Time)
        if t > 0
    }

    max_production = {
        (plant, t, scen): add_constraint(
            model, p[plant, t, scen] - max_power * s[plant, t] <= 0, f"def_max_production_{plant}_{t}_{scen}"
        )
        for ((plant, max_power), t, scen) in product(
            TPP[["plant", "max_power"]].itertuples(index=False), Time, Scenarios
        )
    }

    min_production = {
        (plant, t, scen): add_constraint(
            model, p[plant, t, scen] - min_power * s[plant, t] >= 0, f"def_min_production_{plant}_{t}_{scen}"
        )
        for ((plant, min_power), t, scen) in product(
            TPP[["plant", "min_power"]].itertuples(index=False), Time, Scenarios
        )
    }

    demand_satisfaction = {
        t: add_constraint(
            model,
            lpSum(p[plant, t, scen] for plant in Plants) + ENP[t, scen] - EIE[t, scen] == scenarios[scen].loads[t],
            f"demand_satisfaction_{t}_{scen}",
        )
        for (t, scen) in product(Time, Scenarios)
    }

    ### OBJECTIVE
    thermal_production_cost = lpSum(
        lpSum(l_cost * scenarios[scen].probability * p[plant, t, scen] for scen in Scenarios) + c_cost * s[plant, t]
        for ((plant, l_cost, c_cost), t) in product(TPP[["plant", "l_cost", "c_cost"]].itertuples(index=False), Time)
    )

    demand_mismatch_cost = lpSum(
        scenarios[scen].probability * 2 * ( data.c_EIE * EIE[t, scen] + data.c_ENP * ENP[t, scen])
        for (t, scen) in product(Time, Scenarios)
    )
    total_production_cost = thermal_production_cost + demand_mismatch_cost

    model.setObjective(total_production_cost)

    ### MODEL DICT
    ucp = MathematicalProgram(
        sense=OptimizationSense.MIN,
        vars=dict(s=s, p=p, up=up, dn=dn, EIE=EIE, ENP=ENP),
        cons=dict(
            max_production=max_production,
            min_production=min_production,
            def_up=def_up,
            def_down=def_down,
            def_down_zero=def_down_zero,
            def_up_zero=def_up_zero,
            def_min_on=def_min_on,
            def_min_off=def_min_off,
            demand_satisfaction=demand_satisfaction,
        ),
        objs=dict(total_production_cost=total_production_cost),
        exprs=dict(thermal_production_cost=thermal_production_cost, demand_mismatch_cost=demand_mismatch_cost),
        model=model,
        index_names=index_names,
    )

    return ucp
