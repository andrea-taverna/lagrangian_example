from itertools import groupby, accumulate
from typing import Iterable, Tuple, Dict, Any

import pandas as pd

from UCP.data import UCPData


# Running Length Encoding of sequence (from R)
# Example: rle([1 1 0 1 0 0]) -> [(1,2), (0,1), (1,1), (0,2)]
from generic.optimization.model import Solution


def rle(seq: Iterable):
    return [(k, len(list(g))) for k, g in groupby(seq)]


# Convert sequence of symbols into table of RLE
# Example: seq2rle_table([1 1 0 1 0 0]) ->
#  value    start   end
#    1        0      1
#    0        2      2
#    1        3      3
#    0        4      5
#
def seq2rle_table(seq):
    rl_enc = rle(seq)
    cum_length = accumulate(l for _, l in rl_enc)

    # create "table" with (symbol, sequence length, cumulated length)
    temp = [(sym, l, cl) for (sym, l), cl in list(zip(rl_enc, cum_length))]
    # compute "start" and "end" of each sequence of symbols
    temp = [(sym, l, cl - l, cl - 1) for sym, l, cl in temp]

    return pd.DataFrame(temp, columns=["value", "duration", "start", "end"])


# Returns table of RLE for power plants states with "OK flag" regarding their minimum duration
def check_updown_constraints(states: pd.Series, data: UCPData):
    rles = states.groupby(level=["plant"]).apply(lambda g: seq2rle_table(g))
    rles = pd.merge(rles, data.thermal_plants[["plant", "min_on", "min_off"]], left_on=["plant"], right_on=["plant"])

    max_time = data.loads["period"].max()
    rles["OK"] = False
    rles.loc[rles["value"] == 0, "OK"] = rles.apply(
        lambda r: r["duration"] >= r["min_off"] or (r["start"] == 0 or r["end"] == max_time), axis=1
    )
    rles.loc[rles["value"] == 1, "OK"] = rles.apply(
        lambda r: r["duration"] >= r["min_on"] or (r["start"] == 0 or r["end"] == max_time), axis=1
    )

    return rles


def check_solution(data: UCPData, solution: Solution) -> Tuple[bool, Dict[str, Tuple[bool, Any]]]:
    production = pd.concat([solution["p"], solution["s"]], axis=1).reset_index(drop=False)
    production = production.merge(data.thermal_plants, on=["plant"])
    production["p_max"] = production["max_power"] * production["s"]
    production["p_min"] = production["min_power"] * production["s"]
    production["OK"] = (production["p_min"] <= production["p"]) & (production["p"] <= production["p_max"])

    production_within_bounds = production["OK"].all()
    rles = check_updown_constraints(solution["s"], data)
    updown_constraints_ok = rles["OK"].all()

    # TODO: check ENP and EIE
    return (
        production_within_bounds and updown_constraints_ok,
        dict(
            production_within_bounds=(production_within_bounds, production),
            updown_constraints_check=(updown_constraints_ok, rles),
        ),
    )
