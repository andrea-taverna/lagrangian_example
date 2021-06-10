from functools import reduce
from typing import List

import pandas as pd
from mizani.formatters import percent_format, percent
from numpy import percentile
from scipy.stats import truncnorm
from plotnine import *

from UCP.data import UCPData
from UCP.stochastic import scen_gen
from UCP.stochastic.scen_gen import ScenarioInfo
from generic.optimization.model import Solution

alpha_values = [0.1, 0.2, 0.5]
alpha_breaks = [0.1, 0.2, 0.5]
alpha_labels = ["100%", "80%", "50%"]


def compute_intervals(
    data,
    target,
    groupby_cols,
    percentiles={0: "min", 10: "10%", 25: "25%", 75: "75%", 90: "90%", 100: "max"},
    mean_label="mean",
):

    temp = data[groupby_cols + [target]].groupby(groupby_cols)
    data_columns = [temp.apply(lambda x: percentile(x[target], p)).reset_index() for p in percentiles.keys()]

    data_columns.append(temp[target].mean().reset_index())
    intervals = reduce(lambda x, y: pd.merge(x, y, on=groupby_cols), data_columns)
    intervals.columns = groupby_cols + list(percentiles.values()) + [mean_label]

    return intervals


def confidence_intervals(data=None, fill="grey", inherit_aes=True, name:str="Confidence Interval"):
    return [geom_ribbon(
        data=data,
        mapping=aes("period", ymin="min", ymax="max", alpha=alpha_values[0]),
        fill=fill,
        show_legend=True,
        inherit_aes=inherit_aes,
    ) , geom_ribbon(
        data=data,
        mapping=aes("period", ymin="10%", ymax="90%", alpha=alpha_values[1]),
        fill=fill,
        show_legend=False,
        inherit_aes=inherit_aes,
    ) , geom_ribbon(
        data=data,
        mapping=aes("period", ymin="25%", ymax="75%", alpha=alpha_values[2]),
        fill=fill,
        show_legend=False,
        inherit_aes=inherit_aes,
    ) , scale_alpha_continuous(
        range=(0.1, 0.5), name=name, breaks=alpha_breaks, na_value=-1, labels=alpha_labels
    )]


def total_production(data, solution: Solution) -> ggplot:
    tot_production = solution["p"].groupby(level=["period", "scenario"]).sum().reset_index()

    intervals = compute_intervals(tot_production, "p", ["period"])
    tot_production = pd.merge(intervals[["period", "mean"]], data.loads[["period", "value"]], on="period")
    tot_production = tot_production.rename(columns={"value": "load", "mean": "p"})
    tot_production = pd.melt(tot_production, id_vars=["period"])

    return (
        ggplot(tot_production, aes("period", "value", color="variable", linetype="variable"))
        + confidence_intervals(intervals, inherit_aes=False, name="Confidence interval (production)")
        + geom_line(size=2)
        + scale_color_discrete(name="Series", breaks=["load", "p"], labels=["Exp. Load", "Exp. Production"])
        + scale_linetype_discrete(name="Series", breaks=["load", "p"], labels=["Exp. Load", "Exp. Production"])
        + labs(y="Value [MW]")
        + ggtitle("Production vs Load")
    )


def enp_vs_eie(data:UCPData, scenarios: List[ScenarioInfo] , solution: Solution) -> ggplot:
    demand_per_scenario, _ = scen_gen.to_pandas(*scenarios)
    demand_gap = (solution["EIE"] - solution["ENP"])

    demand_gap /= demand_per_scenario
    demand_gap.name = "demand_gap"

    demand_gap = compute_intervals(demand_gap.reset_index(), "demand_gap", ["period"])

    return (
        ggplot(demand_gap, aes(x="period", y="mean"))
        + geom_line()
        + confidence_intervals()
        + scale_y_continuous(labels=percent)
        + labs(x="period", y="Exp. mismatch %")
        + ggtitle("Demand mismatch")
    )


def production_by_plant(data, solution: Solution) -> ggplot:
    production = solution["p"].reset_index()
    intervals = compute_intervals(production, "p", ["period", "plant"])

    production = pd.merge(
        intervals[["period", "plant", "mean"]], data.thermal_plants[["plant", "max_power", "min_power"]]
    )

    production = pd.melt(production, id_vars=["period", "plant"])
    return (
        ggplot(production, aes("period", "value", color="variable", linetype="variable"))
        + geom_line()
        + scale_color_manual(
            breaks=["mean", "min_power", "max_power"],
            labels=["production", "min power", "max power"],
            values=["black", "darkcyan", "red"],
        )
        + scale_linetype_manual(
            breaks=["mean", "min_power", "max_power"],
            labels=["production", "min power", "max power"],
            values=["solid", "dashed", "dashed"],
        )
        + confidence_intervals(intervals, inherit_aes=False)
        + facet_wrap("~plant", labeller=lambda s: f"plant #{s}")
        + labs(x="hour", y="production [MWh]")
        + ggtitle("Hourly production per plant")
    )


def plant_utilization(data, solution: Solution) -> ggplot:
    TPP = data.thermal_plants
    avg_coef = TPP.l_cost + (TPP.c_cost / TPP.max_power)
    avg_coef.name = "avg_coef"

    avg_coef = pd.concat([avg_coef, TPP["plant"]], axis=1)

    utilization = solution["p"].groupby(level=["plant", "scenario"]).sum().reset_index(0)
    utilization = compute_intervals(utilization, "p", ["plant"])
    utilization = utilization.merge(TPP[["plant", "max_power"]])
    for c in ("mean", "min", "max", "25%", "75%"):
        utilization[c] = utilization[c] / (utilization["max_power"] * len(data.loads))

    temp = avg_coef[["plant", "avg_coef"]].merge(utilization, sort=True)
    avg_coef_delta = (avg_coef["avg_coef"].max() - avg_coef["avg_coef"].min())/35
    temp["x_label"] = temp["avg_coef"] - avg_coef_delta
    return (
        ggplot(temp, aes("avg_coef", "mean"))
        + geom_point(size=2)
        + geom_text(aes(y="mean+0.02", x="x_label", label="plant"))
        + geom_errorbar(aes(ymin="min", ymax="max"), width=0, color="red")
        + geom_errorbar(aes(ymin="25%", ymax="75%"), width=0.05, color="green")
        + scale_y_continuous(labels=percent_format())
        + labs(x="Avg. Hourly cost [€/MW]", y="Utilization % (total production/total max production)", label="Plant id")
        + ggtitle("Utilization vs Hourly cost")
    )
