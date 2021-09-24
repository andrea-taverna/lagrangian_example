from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from plotnine import *

from Knapsack import KnapsackProblem


def knapsack_kpi(problem) -> Tuple[float, float, float]:
    value = problem.total_value.value()
    weight = problem.total_weight.value()

    return value, weight


def _add_labels(plot: ggplot, items_data: pd.DataFrame) -> ggplot:
    data = items_data[["weight", "value"]].reset_index()
    data["weight"] += 0.035
    data["value"] += 0.035
    return plot + geom_text(data=data, mapping=aes("weight", "value", label="index"), inherit_aes=False)


def plot_knapsack(problem: KnapsackProblem, label: bool = True, force_selected: Optional[List[int]] = None) -> ggplot:
    force_selected = [] if force_selected is None else force_selected

    items_data = problem.items_data.copy()
    items_data["selected"] = np.array([x.value() > 0.5 for x in problem.x])
    items_data["force_selected"] = [i in force_selected for i in items_data.index]

    shape_aes = {"shape": "force_selected"} if any(force_selected) else {}

    plot = (
        ggplot(items_data, aes("weight", "value"))
        + geom_point(aes(**shape_aes), size=5, color="black")
        + geom_point(aes(color="factor(selected)", **shape_aes), size=4.5)
    )

    if label:
        plot = _add_labels(plot, items_data)
    plot += labs(color="Selected", shape="Forced")

    value, weight = knapsack_kpi(problem)
    used_capacity = weight / problem.capacity
    plot += ggtitle(
        f"Knapsack solution with {len(problem.items_data)} items and capacity {problem.capacity:5.2f}.\n"
        + f"Value:{value:5.2f} | Weight: {weight:5.2f}, {used_capacity:5.2%} of capacity"
    )
    return plot


def plot_items(items_data: pd.DataFrame, capacity:float, label: bool = True) -> ggplot:
    plot = ggplot(items_data, aes("weight", "value")) + geom_point(size=5, color="black")
    if label:
        plot = _add_labels(plot, items_data)

    plot += ggtitle(f"Knapsack problem with {len(items_data)} items and capacity {capacity:5.2f}.")
    return plot
