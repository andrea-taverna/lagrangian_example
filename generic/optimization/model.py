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
    """
    Represent a mathematical program in terms of variables, constraints, objectives... by names and indexes and the
    actual implementation in PuLP.
    """

    model: LpProblem
    """ model implementation in PuLP"""
    sense: OptimizationSense
    vars: Dict[str, ModelComponentCollection[LpVariable]]
    """ Dict of LpVariables or dicts of them. Indexed by variables' names."""

    cons: Dict[str, ModelComponentCollection[LpConstraint]] = field(default_factory=dict)
    """ Dict of LpConstraints or dicts of them. Indexed by constraints' names."""

    objs: Dict[str, LpAffineExpression] = field(default_factory=dict)
    """ Dict of objectives (LpAffineExpression) for the model. Indexed by objectives' name."""

    exprs: Dict[str, ModelComponentCollection[LpAffineExpression]] = field(default_factory=dict)
    """"Dict of expressions (LpAffineExpressions) for the model. Useful to reference recurring values to be computed 
    from the model without creating variables."""

    index_names: Dict[str, List[str]] = field(default_factory=dict)
    """Dict of index labels for variables and expressions. To be used in solution extraction."""

    def solve(self, *args, **kwargs) -> Any:
        """
        Calls the solver.

        Returns:
            value returns by LpProblem.solve
        """
        return self.model.solve(*args, **kwargs)
