from dataclasses import dataclass

import pandas as pd


@dataclass
class UCPData:
    thermal_plants: pd.DataFrame
    loads: pd.DataFrame

    days: int
    daily_time_periods: int

    c_ENP: float
    c_EIE: float
