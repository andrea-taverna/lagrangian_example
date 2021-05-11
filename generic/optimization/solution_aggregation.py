from typing import Union, Dict, List, Any, TypeVar

import pandas as pd

from generic.optimization.model import MathematicalProgram, Solution
from generic.optimization.solution_extraction import extract_solution

T = TypeVar("T")


def peek_dict_values(my_dict: Dict[Any, T]) -> T:
    return next(v for v in my_dict.values())


def extract_and_aggregate_solutions(subproblems: Dict[Any, MathematicalProgram], id_name: List[str]) -> Solution:
    sub_solutions = {sub_id: extract_solution(model) for sub_id, model in subproblems.items()}
    solution_names = list(peek_dict_values(sub_solutions).keys())

    full_solution_dict = {
        k: {sub_id: solution[k] for sub_id, solution in sub_solutions.items()} for k in solution_names
    }

    temp = {k: _aggregate(data, id_name) for k, data in full_solution_dict.items()}

    return temp


def _aggregate(data: Union[Dict[Any, float], Dict[Any, pd.Series]], id_name: List[str]) -> pd.Series:
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
