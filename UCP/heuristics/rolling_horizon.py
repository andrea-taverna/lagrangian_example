from itertools import islice

from pulp import LpContinuous, LpInteger, PULP_CBC_CMD

from UCP.input.data import UCPData
from UCP.model.ucp import create_model
from generic.optimization.model import MathematicalProgram, Solution
from generic.optimization.solution_extraction import extract_solution


def rolling_horizon_heuristic(data: UCPData, window_size: int, step: int, **kwargs) -> Solution:
    program = create_model(data,)
    start, end = data.loads["period"].min(), data.loads["period"].max()
    set_var_category(start, end, LpContinuous, data, program)

    for start in islice(data.loads["period"], 0, None, step):
        print(start, min(start + window_size - 1, end))
        set_var_category(start, min(start + window_size - 1, end), LpInteger, data, program)
        program.model.solve(PULP_CBC_CMD(**kwargs))
        fix_variables(start, min(start + step - 1, end), data, program)

    start, end = data.loads["period"].min(), data.loads["period"].max()
    unfix_variables(start, end, data, program)

    return program, extract_solution(program)


def set_var_category(start: int, end: int, cat: str, data: UCPData, model: MathematicalProgram):
    for t in range(start, end + 1):
        for p in data.thermal_plants["plant"]:
            for var_name in ("s", "dn", "up"):
                model.vars[var_name][p, t].cat = cat


def fix_variables(start: int, end: int, data: UCPData, model: MathematicalProgram):
    for t in range(start, end + 1):
        for p in data.thermal_plants["plant"]:
            for var_name in ("s", "dn", "up"):
                var = model.vars[var_name][p, t]
                val = var.value()
                var.lb, var.ub = val, val


def unfix_variables(start: int, end: int, data: UCPData, model: MathematicalProgram):
    for t in range(start, end + 1):
        for p in data.thermal_plants["plant"]:
            for var_name in ("s", "dn", "up"):
                var = model.vars[var_name][p, t]
                var.lb, var.ub = 0, 1
