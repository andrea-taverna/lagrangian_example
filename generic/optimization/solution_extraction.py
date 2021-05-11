import collections
from typing import Dict, List, Union, OrderedDict, Any
import pandas as pd
from pulp import LpVariable, LpAffineExpression, LpConstraint, LpContinuous, PULP_CBC_CMD

from generic.optimization.fix_variables import fix_variables, unfix_variables
from generic.optimization.model import MathematicalProgram, Solution

def extract_solution(model: MathematicalProgram) -> Solution:
    extraction_method = {
        dict: lambda k, v: dict_var_to_series(v, name=k, index_name=model.index_names.get(k, None)),
        LpVariable: lambda k, v: v.value(),
        LpAffineExpression: lambda k, v: v.value(),
    }

    solution = {k: extraction_method[type(v)](k, v) for k, v in model.vars.items()}
    solution.update({k: extraction_method[type(v)](k, v) for k, v in model.exprs.items()})
    solution.update({k: extraction_method[type(v)](k, v) for k, v in model.objs.items()})

    return solution


def extract_multipliers(model:MathematicalProgram) -> OrderedDict[str, pd.Series]:
    def _to_series(name:str, cons:Dict[Any, LpConstraint]) -> pd.Series:
        table = pd.Series({k: c.pi for k, c in cons.items()}, name=name)
        index_name = model.index_names.get(name, None)
        if index_name is not None:
            table.index.name = index_name
        return table

    return collections.OrderedDict([(k,_to_series(k,c)) for k,c in model.cons.items()])


def compute_multipliers(model: MathematicalProgram, solution:Solution, **kwargs) -> OrderedDict[str, pd.Series]:
    int_vars = {name: {k: v for k,v in vars.items() if v.cat != LpContinuous} for name, vars in model.vars.items()}

    fixed = False
    previous_bounds = {}
    if sum(map(len,int_vars.values()))>0:
        previous_bounds = fix_variables(int_vars, model, solution)
        kwargs["mip"]=0
        model.solve(PULP_CBC_CMD(**kwargs))
        fixed = True

    multipliers = extract_multipliers(model)

    if fixed:
        unfix_variables(model, previous_bounds)

    return multipliers


def dict_var_to_series(
    dict_var: Dict[str, Union[LpVariable, LpAffineExpression]], name: str = None, index_name: List[str] = None
) -> pd.Series:
    table = pd.Series({k: v.value() for k, v in dict_var.items()}, name=name)

    if index_name is not None:
        table.index.names = index_name

    return table
