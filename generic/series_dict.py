import numpy as np
import pandas as pd
import collections
from typing import OrderedDict


def series_dict_to_array(**series: pd.Series) -> np.ndarray:
    """
    Concatenate given series into 1d array
    Args:
        **series: dict of series

    Returns:
        1d array with values from series
    """
    return np.array([]) if len(series) == 0 else np.concatenate([s.to_numpy() for s in series.values()])


def array_into_series_dict(array: np.ndarray, **series: pd.Series):
    """
    Copies a 1d array into a dict of series
    Args:
        array: 1d array to copy values from
        **series: dict of series
    """
    cur_size = 0
    for s in series.values():
        s.values[:] = array[cur_size : cur_size + s.size]
        cur_size += s.size


def series_dict_indexes(**series: pd.Series) -> OrderedDict[str, pd.Index]:
    """
    Given a dict of series it returns an ordered dict with their indexes (e.g. to be used for `full_series_dict`
    Args:
        **series: dict of series

    Returns:
        ordered dict of indexes of the input dict of series.
    """
    return collections.OrderedDict([(k, v.index) for k, v in series.items()])


def full_series_dict(value: float, **indexes: pd.Index) -> OrderedDict[str, pd.Series]:
    """
    Returns a series dict initialized with the given value. Shape and index of the series is determined by the dict of
    indexes.

    Args:
        value: initial value for the series
        **indexes: dict of indexes for the series

    Returns:

    """
    return collections.OrderedDict(
        [(k, pd.Series(np.full(shape=len(idx), fill_value=value), index=idx)) for k, idx in indexes.items()]
    )
