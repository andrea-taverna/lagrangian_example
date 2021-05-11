from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import typing

SenseData = typing.NamedTuple("SenseData", (("name", str), ("sign", int)))


class ConstraintSense(Enum):
    EQUAL = SenseData("EQUAL", -1)
    GREATER_THAN = SenseData("GREATER_THAN", 1)
    LESS_THAN = SenseData("LESS_THAN", -1)

    @property
    def sign(self) -> float:
        return self.value.sign


@dataclass
class DualizedConstraint:
    name: str
    sense: ConstraintSense
    index: pd.Index
    multipliers_lb: pd.Series
    multipliers_ub: pd.Series

    def __init__(self, name: str, sense: ConstraintSense, multipliers_lb: pd.Series, multipliers_ub: pd.Series):
        assert _check_same_indexes(
            multipliers_lb, multipliers_ub
        ), f"Multipliers  of dualized constraints {name} have different indexes."

        assert _check_range_not_empty(
            multipliers_lb, multipliers_ub
        ), f"Empty range for multipliers of dualized constraints {name}."

        assert _check_range_vs_sense(
            sense, multipliers_lb, multipliers_ub
        ), f"Wrong range for multipliers of dualized constraints {name} for sense {sense.name}."

        self.name = name
        self.sense = sense
        self.index = multipliers_lb.index
        self.multipliers_lb, self.multipliers_ub = multipliers_lb, multipliers_ub


def _check_same_indexes(multipliers_lb: pd.Series, multipliers_ub: pd.Series) -> bool:
    return multipliers_lb.index.equals(multipliers_ub.index)


def _check_range_not_empty(multipliers_lb: pd.Series, multipliers_ub: pd.Series) -> bool:
    return np.all(multipliers_lb < multipliers_ub)


def _check_range_vs_sense(sense: ConstraintSense, multipliers_lb: pd.Series, multipliers_ub: pd.Series) -> bool:
    if sense in (ConstraintSense.GREATER_THAN, ConstraintSense.LESS_THAN):
        return np.allclose(multipliers_lb.values, 0.0) and (multipliers_ub > 0).all()
    else:
        return (multipliers_lb < 0).all() and (multipliers_ub > 0).all()
