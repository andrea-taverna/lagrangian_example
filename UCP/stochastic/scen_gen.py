from math import sqrt
import pandas as pd
from typing import NamedTuple, List, Tuple
import numpy as np
from numpy.random import normal
from UCP.data import UCPData

ScenarioInfo = NamedTuple("ScenarioInfo", [("loads", pd.Series), ("probability", float)])


def log_params(mean:np.ndarray, std:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean2 = mean**2
    std2 = std**2

    log_mean = np.log(mean2/np.sqrt(mean2+std2))
    log_std = np.log(1+ (std2/mean2))
    return log_mean, log_std


def make_scenarios(data: UCPData, noise_ratio: float, n: int = 10, time_error_inflation:float=1.25) -> List[ScenarioInfo]:
    mean_loads = data.loads[["period", "value"]].set_index("period")["value"]
    base_abs_std = mean_loads.std() * noise_ratio
    abs_std = np.array([base_abs_std * sqrt(1.25*(t+1)) for t in range(len(mean_loads))])
    log_mean, log_std = log_params(mean_loads, abs_std)
    loads_scenario = [pd.Series(np.exp(normal(log_mean, log_std)), index=mean_loads.index) for _ in range(n)]
    return [ScenarioInfo(loads=loads, probability=1.0 / n) for loads in loads_scenario]


def to_pandas(*scenarios: ScenarioInfo) -> Tuple[pd.Series, pd.Series]:
    loads = pd.concat({i: s.loads for i, s in enumerate(scenarios)}, names=["scenario"])
    probabilities = pd.Series([s.probability for s in scenarios])
    return loads, probabilities
