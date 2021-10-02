from typing import Sequence
import numpy as np
import pandas as pd
from plotnine import *

from UCP.data import UCPData
from generic.optimization.model import Solution


def constraints_violations(data:UCPData, solutions: Sequence[Solution]) -> ggplot:
    iterations = list(range(len(solutions)))
    p = pd.concat([sol["p"] for sol in solutions], keys=iterations, names=["iteration"])
    s = pd.concat([sol["s"] for sol in solutions], keys=iterations, names=["iteration"])

    production_bounds = data.thermal_plants[["plant", "min_power", "max_power"]]

    temp = pd.merge(p, s, on=["iteration", "plant", "period"]).reset_index()
    temp = pd.merge(temp, production_bounds, on=["plant"]).set_index(["iteration", "plant", "period"])

    temp["violation"] = (temp["p"] - temp["max_power"] * temp["s"]).clip(0) \
                        - (temp["min_power"] * temp["s"] - temp["p"]).clip(0)
    temp["sign"] = np.sign(temp["violation"]).astype(int)
    temp = temp[temp["sign"] != 0]

    violations = (
        temp.reset_index()[["iteration", "plant", "sign", "violation"]]
            .groupby(["iteration", "plant", "sign"])
            .sum()
            .reset_index()
    )

    violations["violation"] = violations["violation"].abs()
    return ggplot(violations, aes("factor(plant)", "violation", fill="factor(sign)")) + [
        geom_col(),
        facet_wrap("~iteration", labeller=lambda s: f"iteration #{s}"),
        labs(x="Power plant id #", y="Violation [MWh]", fill="Violation type"),
        scale_fill_discrete(
            breaks=[1, -1], labels=["over-production", "under-production"]
        ),
        ggtitle("Total Constraints Violations by Iteration and Plant"),
    ]


