from typing import List
import pandas as pd
from dataclasses import dataclass
from pulp import *


@dataclass
class KnapsackProblem:
    items_data: pd.DataFrame
    capacity: float
    model: LpProblem
    x: List[LpVariable]
    total_value: LpAffineExpression
    total_weight: LpAffineExpression
    capacity_constraint: LpConstraint


def make_knapsack_model(items_data: pd.DataFrame, capacity: float) -> KnapsackProblem:
    I = list(range(len(items_data)))

    model = LpProblem("Knapsack", LpMaximize)

    x = [LpVariable(cat=LpBinary, name=f"x_{i}") for i in I]

    total_value = lpSum(items_data["value"][i] * x[i] for i in I)
    total_weight = lpSum(items_data["weight"][i] * x[i] for i in I)

    model.addConstraint(total_weight <= capacity, "capacity_constraint")

    model.setObjective(total_value)

    return KnapsackProblem(
        items_data=items_data,
        capacity=capacity,
        model=model,
        x=x,
        total_value=total_value,
        total_weight=total_weight,
        capacity_constraint=model.constraints["capacity_constraint"],
    )