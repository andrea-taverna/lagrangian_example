from typing import Union, Dict, List, Any, TypeVar

import pandas as pd

from generic.optimization.model import MathematicalProgram, Solution
from generic.optimization.solution_extraction import extract_solution

T = TypeVar("T")


def extract_and_aggregate_solutions(subproblems: Dict[Any, MathematicalProgram], id_name: List[str]) -> Solution:
    """
    Given a dictionary of subproblems from a (lagrangian) decomposition, returns a full solution for the non-decomposed
    problem by merging the solutions of each subproblem.
    Args:
        subproblems: dictionary of subproblems (mathematical programs) indexed by subproblem keys.
                     Subproblems have the same model structure but different data.
        id_name: column name of the keys

    Returns:
        full solution
    """
    # extract solutions from subproblems as dict {subproblem_id:solution}
    sub_solutions = {sub_id: extract_solution(model) for sub_id, model in subproblems.items()}
    # extract the keys in the first subproblem's solution
    solution_keys = list(peek_dict_values(sub_solutions).keys())

    # index solutions' values by key field, yielding  dict-of-dict {solution_key : {subproblem_id: solution_value}}
    full_solution_dict = {
        k: {sub_id: solution[k] for sub_id, solution in sub_solutions.items()} for k in solution_keys
    }

    # merge the values for each solution_key
    temp = {k: _aggregate(data, id_name) for k, data in full_solution_dict.items()}

    return temp


def _aggregate(data: Dict[Any, Union[float, pd.Series]], id_name: List[str]) -> pd.Series:
    """
    Aggregates a dictionary of values in one pandas series indexed by the dict's keys.
    Args:
        data: either a dict of series or floats
        id_name: index names for the keys

    Returns:

    """
    sample = peek_dict_values(data)

    if isinstance(sample, pd.Series):
        data: Dict[Any, pd.Series]
        return pd.concat(list(data.values()), keys=data.keys(), names=id_name)
    else:
        data: Dict[Any, float]
        if len(id_name) > 1:
            index = pd.MultiIndex.from_tuples(list(data.keys()), names=id_name)
        else:
            index = pd.Index(list(data.keys()), name=id_name[0])
        return pd.Series(list(data.values()), index=index)


def peek_dict_values(my_dict: Dict[Any, T]) -> T:
    return next(v for v in my_dict.values())

