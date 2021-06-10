import math

import pandas as pd
from typing import NamedTuple, List, Tuple
from numpy.random import normal
from UCP.data import UCPData

ScenarioInfo= NamedTuple("ScenarioInfo", [("loads", pd.Series), ("probability", float)])


def make_scenarios (data:UCPData, noise_ratio:float, n:int=10) -> List[ScenarioInfo]:
    exp_loads = data.loads[["period","value"]].set_index("period")["value"]
    abs_std = (exp_loads.std()*noise_ratio)
    loads_scenario = [pd.Series(normal(exp_loads, abs_std), index=exp_loads.index) for _ in range(n)]
    return [ScenarioInfo(loads=loads, probability=1.0/n) for loads in loads_scenario]


def to_pandas(*scenarios:ScenarioInfo) -> Tuple[pd.Series, pd.Series]:
    loads = pd.concat({i: s.loads for i, s in enumerate(scenarios)}, names=["scenario"])
    probabilities = pd.Series([s.probability for s in scenarios])
    return loads, probabilities