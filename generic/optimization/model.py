from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Union, TypeVar, Any, List

from pandas import Series
from pulp import LpVariable, LpConstraint, LpAffineExpression, LpProblem


class OptimizationSense(IntEnum):
    MAX = 1
    MIN = -1

    def __neg__(self) -> "OptimizationSense":
        return OptimizationSense(-self.value)


Solution = Dict[str, Union[float, Series]]

T = TypeVar("T")
ModelComponentCollection = Dict[str, Union[T, Dict[Any, T]]]


@dataclass
class MathematicalProgram:
    model: LpProblem
    sense: OptimizationSense
    vars: Dict[str, ModelComponentCollection[LpVariable]]
    cons: Dict[str, ModelComponentCollection[LpConstraint]] = field(default_factory=dict)
    objs: Dict[str, ModelComponentCollection[LpAffineExpression]] = field(default_factory=dict)
    exprs: Dict[str, ModelComponentCollection[LpAffineExpression]] = field(default_factory=dict)

    index_names: Dict[str, List[str]] = field(default_factory=dict)

    def solve(self, *args, **kwargs) -> Any:
        return self.model.solve(*args, **kwargs)
