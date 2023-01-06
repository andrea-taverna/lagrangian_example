from itertools import product
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpBinary

from UCP.data import UCPData
from generic.optimization.constraint_adder import add_constraint
from generic.optimization.model import MathematicalProgram, OptimizationSense

single_ucp_columns_names = dict(s=["period"], up=["period"], dn=["period"])

eco_dispatch_columns_names = dict(p=["plant", "period"], ENP=["period"], EIE=["period"])


def make_economic_dispatch(data: UCPData) -> MathematicalProgram:
    TPP = data.thermal_plants
    Time = data.loads["period"].values
    model = LpProblem("UCP_ED", sense=LpMinimize)

    ### VARIABLES
    p = {
        (id, t): LpVariable(f"p_{id}_{t}", lowBound=0, upBound=max_power)
        for ((id, max_power), t) in product(TPP[["plant", "max_power"]].itertuples(index=False), Time)
    }

    EIE = LpVariable.dict("EIE", Time, lowBound=0)

    ENP = LpVariable.dict("ENP", Time, lowBound=0)

    ### CONSTRAINTS
    demand_satisfaction = {
        t: add_constraint(
            model,
            lpSum(p[id, t] for id in TPP["plant"].values) + ENP[t] - EIE[t] == data.loads["value"][t],
            f"demand_satisfaction_{t}",
        )
        for t in Time
    }

    ### OBJECTIVE
    thermal_production_cost = lpSum(
        l_cost * p[id, t] for ((id, l_cost), t) in product(TPP[["plant", "l_cost"]].itertuples(index=False), Time)
    )

    dispatch_cost = thermal_production_cost + lpSum(data.c_EIE * EIE[t] + data.c_ENP * ENP[t] for t in Time)

    ### MODEL MUNCH
    return MathematicalProgram(
        vars=dict(p=p, EIE=EIE, ENP=ENP),
        cons=dict(demand_satisfaction=demand_satisfaction),
        exprs=dict(dispatch_cost=dispatch_cost),
        model=model,
        sense=OptimizationSense.MIN,
        index_names=eco_dispatch_columns_names,
    )


def make_single_UCP(data, plant) -> MathematicalProgram:
    TPP = data.thermal_plants.set_index("plant")[data.thermal_plants["plant"] == plant].to_dict("index")[plant]
    Time = data.loads["period"].values
    model = LpProblem("UCP", sense=LpMinimize)

    ### VARIABLES
    s = LpVariable.dict(f"s_{plant}", Time, cat=LpBinary)

    up = LpVariable.dict(f"up_{plant}", Time, cat=LpBinary)

    dn = LpVariable.dict(f"dn_{plant}", Time, cat=LpBinary)

    ### CONSTRAINTS
    def_up = {t: add_constraint(model, up[t] >= s[t] - s[t - 1], f"def_up_{plant}_{t}") for t in Time if t > 0}

    def_down = {t: add_constraint(model, dn[t] >= s[t - 1] - s[t], f"def_dn_{plant}_{t}") for t in Time if t > 0}

    def_up_zero = {t: add_constraint(model, up[t] <= s[t], f"def_up_zero_{plant}_{t}") for t in Time}

    def_down_zero = {t: add_constraint(model, dn[t] <= 1 - s[t], f"def_dn_zero_{plant}_{t}") for t in Time}

    def_min_on = {
        t: add_constraint(
            model,
            s[t] >= lpSum(up[t1] for t1 in range(max(0, t - TPP["min_on"]+1), t+1)),
            f"def_min_on_{plant}_{t}",
        )
        for t in Time
        if t > 0
    }

    def_min_off = {
        t: add_constraint(
            model,
            s[t] <= 1 - lpSum(dn[t1] for t1 in range(max(0, t - TPP["min_off"]+1), t+1)),
            f"def_min_off_{plant}_{t}",
        )
        for t in Time
        if t > 0
    }

    ### OBJECTIVE (to be set later)
    commitment_cost = lpSum(TPP["c_cost"] * s[t] for t in Time)

    ### MODEL MUNCH
    return MathematicalProgram(
        vars=dict(s=s, up=up, dn=dn),
        cons=dict(
            def_up=def_up,
            def_down=def_down,
            def_down_zero=def_down_zero,
            def_up_zero=def_up_zero,
            def_min_on=def_min_on,
            def_min_off=def_min_off,
        ),
        exprs=dict(commitment_cost=commitment_cost),
        model=model,
        sense=OptimizationSense.MIN,
        index_names=single_ucp_columns_names,
    )
