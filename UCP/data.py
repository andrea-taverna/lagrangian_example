from dataclasses import dataclass

import pandas as pd


@dataclass
class UCPData:
    thermal_plants: pd.DataFrame
    loads: pd.DataFrame

    c_ENP: float
    c_EIE: float

    def __init__(
        self,
        thermal_plants: pd.DataFrame,
        loads: pd.DataFrame,
        EIE_cost_factor: float = 5.0,
        ENP_cost_factor: float = 5.0,
    ):
        self.thermal_plants = thermal_plants
        self.loads = loads
        max_cost = thermal_plants.apply(
            lambda r: r["l_cost"] + (r["c_cost"] / max(r["min_power"], 1)),
            axis="columns",
        ).max()
        self.c_ENP = ENP_cost_factor * max_cost
        self.c_EIE = EIE_cost_factor * max_cost
