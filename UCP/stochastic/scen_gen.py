from math import sqrt
import pandas as pd
from typing import NamedTuple, List, Tuple
import numpy as np
from numpy.random import normal
from UCP.data import UCPData

ScenarioInfo = NamedTuple("ScenarioInfo", [("loads", pd.Series), ("probability", float)])


def make_scenarios(data: UCPData, noise_ratio: float, n:int, step_var_increase:float=1.0) -> List[ScenarioInfo]:
    mean_loads = data.loads[["period", "value"]].set_index("period")["value"]

    # compute forecast sd/variance
    base_abs_std = mean_loads.std() * noise_ratio
    min_load = mean_loads.min()
    # limit the sd of the forecast distribution to make negative load values unlikely
    max_std = (mean_loads - 0.15*min_load)/3
    # variance increases linearly with time
    abs_std = np.array([min(max_std[t], base_abs_std * sqrt((step_var_increase*t)+1)) for t in range(len(mean_loads))])

    loads_scenario = [pd.Series(normal(mean_loads, abs_std), index=mean_loads.index) for _ in range(n)]
    return [ScenarioInfo(loads=loads, probability=1.0 / n) for loads in loads_scenario]


def to_pandas(*scenarios: ScenarioInfo) -> Tuple[pd.Series, pd.Series]:
    loads = pd.concat({i: s.loads for i, s in enumerate(scenarios)}, names=["scenario"])
    probabilities = pd.Series([s.probability for s in scenarios])
    return loads, probabilities
