import numpy as np
import pandas as pd
import collections
from typing import OrderedDict


def series_dict_to_array(**series: pd.Series) -> np.ndarray:
    return np.array([]) if len(series) == 0 else np.concatenate([s.to_numpy() for s in series.values()])


def series_dict_into_array(array: np.ndarray, **series: pd.Series):
    cur_size = 0
    for s in series.values():
        array[cur_size : cur_size + s.size] = s.values
        cur_size += s.size


def array_into_series_dict(array: np.ndarray, **series: pd.Series):
    cur_size = 0
    for s in series.values():
        s.values[:] = array[cur_size : cur_size + s.size]
        cur_size += s.size


def full_series_dict(value: float, **indexes: pd.Index) -> OrderedDict[str, pd.Series]:
    return collections.OrderedDict(
        [(k, pd.Series(np.full(shape=len(idx), fill_value=value), index=idx)) for k, idx in indexes.items()]
    )


def series_dict_indexes(**series: pd.Series) -> OrderedDict[str, pd.Index]:
    return collections.OrderedDict([(k, v.index) for k, v in series.items()])


def allocate_array_for(**series: pd.Series) -> np.ndarray:
    return np.empty(sum([len(v) for v in series.values()]))
