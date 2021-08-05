import pandas as pd
from numpy.random import uniform

def make_items(size: int, min_val: float = 1e-4) -> pd.DataFrame:
    def _random_data():
        return uniform(min_val, 1, size)

    return pd.DataFrame({"value": _random_data(), "weight": _random_data()})
