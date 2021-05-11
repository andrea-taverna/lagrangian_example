from pulp import LpVariable, LpProblem, LpMinimize, LpBinary, lpSum

from UCP.input.data import UCPData
from generic.optimization.constraint_adder import add_constraint
from generic.optimization.model import MathematicalProgram, OptimizationSense

single_ucp_columns_names = dict(s=["period"], p=["period"], up=["period"], dn=["period"])


def make_single_UCP(data: UCPData, plant: int) -> MathematicalProgram:
    TPP = data.thermal_plants.set_index("plant")[data.thermal_plants["plant"] == plant].to_dict("index")[plant]
    Time = data.loads["period"].values.astype(int)

    model = LpProblem(f"UCP_{plant:02d}", sense=LpMinimize)

    ### VARIABLES
    p = LpVariable.dict(f"p_{plant}", Time)

    s = LpVariable.dict(f"s_{plant}", Time, cat=LpBinary)

    up = LpVariable.dict(f"up_{plant}", Time, cat=LpBinary)
    dn = LpVariable.dict(f"dn_{plant}", Time, cat=LpBinary)

    def_up = {t: add_constraint(model, up[t] >= s[t] - s[t - 1], f"def_up_{plant}_{t}") for t in Time if t > 0}

    def_down = {t: add_constraint(model, dn[t] >= s[t - 1] - s[t], f"def_dn_{plant}_{t}") for t in Time if t > 0}

    def_up_zero = {t: add_constraint(model, up[t] <= s[t], f"def_up_zero_{plant}_{t}") for t in Time}

    def_down_zero = {t: add_constraint(model, dn[t] <= 1 - s[t], f"def_dn_zero_{plant}_{t}") for t in Time}

    def_min_on = {
        t: add_constraint(model,
            s[t] >= lpSum(up[t1] for t1 in range(max(0, t - TPP["min_on"]), t)), f"def_min_on_{plant}_{t}"
        )
        for t in Time
        if t > 0
    }

    def_min_off = {
        t: add_constraint(model,
            s[t] >= lpSum(dn[t1] for t1 in range(max(0, t - TPP["min_off"]), t)), f"def_min_off_{plant}_{t}"
        )
        for t in Time
        if t > 0
    }

    max_production = {
        t: add_constraint(model, p[t] <= TPP["max_power"] * s[t], f"def_max_production_{plant}_{t}") for t in Time
    }

    min_production = {
        t: add_constraint(model, p[t] >= TPP["min_power"] * s[t], f"def_min_production_{plant}_{t}") for t in Time
    }

    ### OBJECTIVE
    thermal_production_cost = lpSum(TPP["l_cost"] * p[t] + TPP["c_cost"] * s[t] for t in Time)

    ### MODEL MUNCH
    single_ucp = MathematicalProgram(
        sense=OptimizationSense.MIN,
        vars=dict(s=s, up=up, dn=dn, p=p),
        cons=dict(
            def_up=def_up,
            def_down=def_down,
            def_down_zero=def_down_zero,
            def_up_zero=def_up_zero,
            def_min_on=def_min_on,
            def_min_off=def_min_off,
            max_production=max_production,
            min_production=min_production,
        ),
        exprs=dict(thermal_production_cost=thermal_production_cost),
        model=model,
        index_names=single_ucp_columns_names,
    )

    return single_ucp
