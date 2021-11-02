from typing import List

import pandas as pd
from mizani.formatters import percent_format, percent
from numpy import percentile
from plotnine import *

from UCP.data import UCPData
from UCP.stochastic import scen_gen
from UCP.stochastic.model import create_model
from UCP.stochastic.scen_gen import ScenarioInfo, to_pandas
from UCP.stochastic.solution_evalution import StochasticEvaluation
from generic.optimization.model import Solution, MathematicalProgram
from generic.optimization.solution_extraction import compute_multipliers

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
    data_columns = [temp.apply(lambda x: percentile(x[target], p)) for p, k in percentiles.items()]

    data_columns.append(temp[target].mean())
    intervals = pd.concat(data_columns, axis=1).reset_index()
    intervals.columns = groupby_cols + list(percentiles.values()) + [mean_label]

    return intervals


def confidence_intervals(data=None, fill="grey", inherit_aes=True, name: str = "Confidence Interval"):
    return [
        geom_ribbon(
            data=data,
            mapping=aes("period", ymin="min", ymax="max", alpha=alpha_values[0]),
            fill=fill,
            show_legend=True,
            inherit_aes=inherit_aes,
        ),
        geom_ribbon(
            data=data,
            mapping=aes("period", ymin="10%", ymax="90%", alpha=alpha_values[1]),
            fill=fill,
            show_legend=False,
            inherit_aes=inherit_aes,
        ),
        geom_ribbon(
            data=data,
            mapping=aes("period", ymin="25%", ymax="75%", alpha=alpha_values[2]),
            fill=fill,
            show_legend=False,
            inherit_aes=inherit_aes,
        ),
        scale_alpha_continuous(range=(0.1, 0.5), name=name, breaks=alpha_breaks, na_value=-1, labels=alpha_labels),
    ]


def load(scenarios: List[ScenarioInfo]) -> ggplot:
    loads_scen, _ = to_pandas(*scenarios)
    loads_scen_df = pd.DataFrame(loads_scen, columns=["value"]).reset_index()

    intervals = compute_intervals(loads_scen_df, "value", ["period"])

    return (
        ggplot(loads_scen_df, aes("period", "value", color="factor(scenario)"))
        + geom_line(show_legend=False)
        + confidence_intervals(intervals, inherit_aes=False)
        + labs(y="Load [MWh]", color="Scenario")
        + ggtitle("Load scenarios")
    )


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
        + labs(y="Value [MWh]")
        + ggtitle("Production vs Load")
    )


def enp_vs_eie(data: UCPData, scenarios: List[ScenarioInfo], solution: Solution) -> ggplot:
    demand_per_scenario, _ = scen_gen.to_pandas(*scenarios)
    demand_gap = solution["EIE"] - solution["ENP"]

    demand_gap /= demand_per_scenario
    demand_gap.name = "demand_gap"

    demand_gap = compute_intervals(demand_gap.reset_index(), "demand_gap", ["period"])

    return (
        ggplot(demand_gap, aes(x="period", y="mean"))
        + geom_line(size=1)
        + confidence_intervals()
        + scale_y_continuous(labels=percent)
        + labs(x="period", y="Exp. mismatch %")
        + ggtitle("Demand mismatch")
    )


def electricity_prices(
    data: UCPData,
    scenarios: List[ScenarioInfo],
    solution: Solution,
    model: MathematicalProgram = None,
    **solver_options,
) -> ggplot:
    model = model if model is not None else create_model(data, scenarios)

    electricity_prices = (
        compute_multipliers(model, solution, **solver_options)["demand_satisfaction"].to_frame().reset_index()
    )
    electricity_prices.columns = ["period", "scenario", "electricity_price"]

    scenarios_prob = pd.DataFrame(
        [(i, s.probability) for i, s in enumerate(scenarios)], columns=["scenario", "probability"]
    )

    electricity_prices = pd.merge(electricity_prices, scenarios_prob, on="scenario")

    electricity_prices["electricity_price"] /= electricity_prices["probability"]
    electricity_prices_data = compute_intervals(electricity_prices, "electricity_price", ["period"])

    return (
        ggplot(electricity_prices_data, aes("period", "mean"))
        + geom_line(size=2, color="darkorange")
        + confidence_intervals(inherit_aes=False)
        + ggtitle("Hourly Electricity Prices")
        + labs(x="period", y="hourly Electricity price [€/MWh]")
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

    out_error_bars = temp[["min", "max", "avg_coef"]].copy(deep=False)
    out_error_bars["type"] = "out"

    in_error_bars = temp[["25%", "75%", "avg_coef"]].copy(deep=False)
    in_error_bars["type"] = "in"

    return (
        ggplot(temp, aes("avg_coef", "mean"))
        + geom_errorbar(
            aes(x="avg_coef", ymin="min", ymax="max", color="type"),
            data=out_error_bars,
            width=0.1,
            size=1,
            inherit_aes=False,
        )
        + geom_errorbar(
            aes(x="avg_coef", ymin="25%", ymax="75%", color="type"),
            data=in_error_bars,
            width=0.1,
            size=1,
            inherit_aes=False,
        )
        + geom_label(aes(label="plant"))
        + scale_y_continuous(labels=percent_format())
        + scale_color_discrete(breaks=["out", "in"], labels=["100%", "50%"], name="Confidence interval")
        + labs(x="Avg. Hourly cost [€/MWh]", y="Utilization % (total production/total max production)")
        + ggtitle("Utilization vs Hourly cost")
    )


def compare_deterministic_stochastic(
    deterministic_value: float, deterministic_eval: StochasticEvaluation, stochastic_eval: StochasticEvaluation
) -> ggplot:
    evaluation = pd.DataFrame.from_dict(
        {"Deterministic": deterministic_eval.cost_by_scenario, "Stochastic": stochastic_eval.cost_by_scenario}
    )
    evaluation["scenario"] = evaluation.index
    data = pd.melt(evaluation, id_vars="scenario")

    lines = pd.DataFrame(
        [
            ("Deterministic", deterministic_value),
            ("Evaluated Deterministic ", deterministic_eval.expected_cost),
            ("Evaluated Stochastic", stochastic_eval.expected_cost),
        ],
        columns=["name", "position"],
    )

    return (
        ggplot(data, aes("value", fill="variable"))
        + geom_density(alpha=0.4, color="lightgrey")
        + geom_vline(lines, aes(xintercept="position", color="name"), size=2, linetype="-.")
        + ggtitle("Cost comparison between deterministic and stochastic solution")
        + scale_color_manual(breaks=lines["name"], values=["black", "red", "cyan"])
        + labs(x="Cost", fill="Cost distribution for solution type", color="Cost value")
    )
