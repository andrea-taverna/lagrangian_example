import logging
import re
from typing import TextIO

import pandas as pd

from generic.parsing import parse_num, INTEGER_RE, NUMBER_RE, RawData, parse_stream_by_keywords
from UCP.input.data import UCPData

logger = logging.getLogger(__name__)


def parse_loads_section(current_line: str, data: RawData, stream: TextIO) -> RawData:
    tokens = re.findall(INTEGER_RE, current_line)
    num_days = int(tokens[0])
    daily_time_periods = int(tokens[1])

    rows = []
    for i in range(num_days):
        line = next(stream)
        rows.append([float(v) for v in re.findall(NUMBER_RE, line)])
    assert all(len(x) == daily_time_periods for x in rows)

    loads = pd.DataFrame(rows).reset_index()
    loads = loads.rename(columns={"index": "day"})
    loads = loads.melt(id_vars=["day"], var_name="hour").sort_values(["day", "hour"])

    loads.reset_index(drop=True, inplace=True)  # remove old index made with old ordering
    loads.reset_index(drop=False, inplace=True)  # use new sequential index as period column.
    loads = loads.rename(columns={"index": "period"})

    return {"Days": num_days, "DailyTimePeriods": daily_time_periods, "Loads": loads}


# format of Thermal Section table
THERMAL_SECTION_COLUMN_SPEC = [
    ("plant", int),
    ("q_cost", float),
    ("l_cost", float),
    ("c_cost", float),
    ("min_power", float),
    ("max_power", float),
    ("init_status", int),
    ("min_on", int),
    ("min_off", int),
]


def parse_thermal_section(current_line: str, data: RawData, stream: TextIO) -> RawData:
    tpp_list = []
    for t in range(data["NumThermal"]):
        line = next(stream)
        if not line.startswith("RampConstraint"):
            tokens = re.findall("[\S]+", line)
            tpp = {col: parser(token) for (col, parser), token in zip(THERMAL_SECTION_COLUMN_SPEC, tokens)}
            tpp_list.append(tpp)
        else:
            tpp = tpp_list[-1]
            tokens = re.findall(NUMBER_RE, line)
            tpp["ramp_up"] = float(tokens[0])
            tpp["ramp_down"] = float(tokens[1])

    TPP = pd.DataFrame.from_records(tpp_list)

    return {"ThermalSection": TPP}


parse_keyword = {
    "ProblemNum": parse_num("ProblemNum", int),
    "HorizonLen": parse_num("HorizonLen", int),
    "NumThermal": parse_num("NumThermal", int),
    "MinSystemCapacity": parse_num("MinSystemCapacity", float),
    "MaxSystemCapacity": parse_num("MaxSystemCapacity", float),
    "MaxThermalCapacity": parse_num("MaxThermalCapacity", float),
    "Loads": parse_loads_section,
    "ThermalSection": parse_thermal_section,
}


def post_process(data: RawData, EIE_cost_factor: float = 5.0, ENP_cost_factor: float = 5.0):

    max_cost = (
        data["ThermalSection"]
        .apply(lambda r: r["l_cost"] + (r["c_cost"] / max(r["min_power"], 1)), axis="columns")
        .max()
    )
    data["c_EIE"] = EIE_cost_factor * max_cost
    data["c_ENP"] = ENP_cost_factor * max_cost
    return data


def read_instance(file_path: str, **post_process_options):

    with open(file_path, "r") as f:
        raw_data = parse_stream_by_keywords(f, parse_keyword)
    raw_data.update(post_process(raw_data, **post_process_options))

    raw_data = post_process(raw_data, **post_process_options)

    ucp_data = UCPData(
        thermal_plants=raw_data["ThermalSection"],
        loads=raw_data["Loads"],
        days=raw_data["Days"],
        daily_time_periods=raw_data["DailyTimePeriods"],
        c_ENP=raw_data["c_ENP"],
        c_EIE=raw_data["c_EIE"],
    )

    return ucp_data
