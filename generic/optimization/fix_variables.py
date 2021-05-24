from typing import Dict, Tuple, Any

from pulp import LpVariable

from generic.optimization.model import MathematicalProgram, Solution


def fix_variables(
    variables: Dict[str, Dict[Any,LpVariable]], model: MathematicalProgram, solution: Solution
) -> Dict[str, Dict[Any, Tuple[float, float]]]:
    """
    Fixes variables in `variables` to the values found in `solution` for model `model`.

    Args:
        variables: dict of variables in `model` to fix
        model: LP model
        solution: values for the variables in `variables`

    Returns:
        previous bounds of the variables. See `unfix_variables`
    """
    previous_bounds = {name:{k:(v.lowBound, v.upBound) for k, v in vars.items()} for name, vars in variables.items()}

    for name, vars in variables.items():
        for k, v in vars.items():
            v.lowBound = v.upBound = solution[name][k]

    return previous_bounds


def unfix_variables(model: MathematicalProgram, previous_bounds: Dict[str, Dict[Any, Tuple[float, float]]]):
    """
    Given a model with fixed variables it sets their bounds to the values set in `previous_bounds`.

    Args:
        model: model with fixed variables
        previous_bounds:  dict {variable_name: {variable_index: (lower_bound, upper_bound)}}
    """
    for name, bounds in previous_bounds.items():
        var = model.vars[name]
        for key, (low, up) in bounds.items():
            var[key].lowBound = low
            var[key].upBound = up