import collections
from typing import Dict, List, Union, OrderedDict, Any
import pandas as pd
from pulp import LpVariable, LpAffineExpression, LpConstraint, LpContinuous, COIN_CMD

from generic.optimization.fix_variables import fix_variables, unfix_variables
from generic.optimization.model import MathematicalProgram, Solution


def extract_solution(model: MathematicalProgram) -> Solution:
    """
    Extract solution from a solved model.
    Args:
        model:  a solved mathematical model

    Returns: solution as a series_dict

    """
    extraction_method = {
        dict: lambda k, v: dict_var_to_series(v, name=k, index_name=model.index_names.get(k, None)),
        LpVariable: lambda k, v: v.value(),
        LpAffineExpression: lambda k, v: v.value(),
    }

    solution = {k: extraction_method[type(v)](k, v) for k, v in model.vars.items()}
    solution.update({k: extraction_method[type(v)](k, v) for k, v in model.exprs.items()})
    solution.update({k: extraction_method[type(v)](k, v) for k, v in model.objs.items()})

    return solution


def compute_multipliers(model: MathematicalProgram, solution: Solution, **kwargs) -> OrderedDict[str, pd.Series]:
    """
    Compute multipliers for a MIP problem by fixing the integer variables to the value find in `solution` and re-solving
    the model as an LP.
    TODO: add multiplier extractions for variables bounds

    Args:
        model: mathematical program for the problem
        solution: solution for the model
        **kwargs: options for the solver

    Returns:
        a series_dict with multipliers for each constraint.
    """
    # extract variables that are integer/non-continuous
    int_vars = {name: {k: v for k,v in vars.items() if v.cat != LpContinuous} for name, vars in model.vars.items()}

    # if there are integer variables, fix their values to the solution's, then solve the model as an LP
    fixed = False
    previous_bounds = {}
    if sum(map(len,int_vars.values()))>0:
        previous_bounds = fix_variables(int_vars, model, solution)
        fixed = True

    # solve the model as an LP and extract the multipliers
    kwargs["mip"] = 0
    model.solve(COIN_CMD(**kwargs))
    multipliers = extract_multipliers(model)

    # if some variables were fixed, restore their bounds
    if fixed:
        unfix_variables(model, previous_bounds)

    return multipliers


def extract_multipliers(model:MathematicalProgram) -> OrderedDict[str, pd.Series]:
    """
    Extract lagrangian multipliers from solved model.
    TODO: add multiplier extractions for variables bounds

    Args:
        model: solved model

    Returns:
        dict of series containing the multipliers for each constraint
    """
    def _to_series(name:str, cons:Dict[Any, LpConstraint]) -> pd.Series:
        table = pd.Series({k: c.pi for k, c in cons.items()}, name=name)
        index_name = model.index_names.get(name, None)
        if index_name is not None:
            table.index.name = index_name
        return table

    return collections.OrderedDict([(k,_to_series(k,c)) for k,c in model.cons.items()])


def dict_var_to_series(
    dict_var: Dict[str, Union[LpVariable, LpAffineExpression]], name: str = None, index_name: List[str] = None
) -> pd.Series:
    """
    Given a dict of LpVariables and expressions of a solved model it returns their values as dict of series.

    Args:
        dict_var: dictionary of vars or expressions in an LP model
        name: name of the series
        index_name: name of the indexes

    Returns:
        a series_dict with the values of the variables/expressions in `dict_var`.
    """
    table = pd.Series({k: v.value() for k, v in dict_var.items()}, name=name)

    if index_name is not None:
        table.index.names = index_name

    return table
