import pandas as pd
from mizani.formatters import percent_format, percent
from plotnine import *

from UCP.data import UCPData
from UCP.model import create_model
from generic.optimization.model import Solution, MathematicalProgram
from generic.optimization.solution_extraction import compute_multipliers


def total_production(data, solution: Solution) -> ggplot:
    tot_production = solution["p"].groupby(level=["period"]).sum().reset_index()
    tot_production = pd.merge(tot_production, data.loads[["period", "value"]], on="period")
    tot_production = tot_production.rename(columns={"value": "load"})

    tot_production = pd.melt(tot_production, id_vars=["period"])

    return (
        ggplot(tot_production, aes("period", "value", color="variable", linetype="variable"))
        + geom_step(size=2)
        + scale_color_discrete(name="Series", breaks=["load", "p"], labels=["Load", "Production"])
        + scale_linetype_discrete(name="Series", breaks=["load", "p"], labels=["Load", "Production"])
        + labs(y="Value [MWh]")
        + ggtitle("Production vs Load")
    )


def enp_vs_eie(data, solution: Solution) -> ggplot:
    demand_gap = pd.merge(solution["EIE"], solution["ENP"], on="period")
    demand = data.loads[["period", "value"]].set_index("period")["value"]
    demand_gap["ENP"] /= demand
    demand_gap["EIE"] /= demand
    demand_gap = pd.melt(demand_gap.reset_index(), id_vars="period", var_name="Series")

    return (
        ggplot(demand_gap, aes(x="period", y="value", color="Series", linetype="Series"))
        + geom_step(size=2)
        + scale_y_continuous(labels=percent)
        + labs(x="period", y="Demand mismatch%")
        + ggtitle("Demand mismatch")
    )


def production_by_plant(data, solution: Solution) -> ggplot:
    production = solution["p"].reset_index()
    production = production.merge(data.thermal_plants[["plant", "max_power", "min_power"]])
    production = pd.melt(production, id_vars=["period", "plant"])

    return (
        ggplot(production, aes("period", "value", color="variable", linetype="variable", alpha="variable"))
        + geom_step()
        + scale_color_manual(
            breaks=["p", "min_power", "max_power"],
            labels=["production", "min power", "max power"],
            values=["black", "darkcyan", "red"],
        )
        + scale_linetype_manual(
            breaks=["p", "min_power", "max_power"],
            labels=["production", "min power", "max power"],
            values=["solid", "dashed", "dashed"],
        )
        + scale_alpha_manual(
            breaks=["p", "min_power", "max_power"],
            labels=["production", "min power", "max power"],
            values=[1, 0.6, 0.6],
        )
        + facet_wrap("~plant", labeller=lambda s: f"plant #{s}")
        + labs(x="hour", y="production [MWh]")
        + ggtitle("Hourly production per plant")
    )


def plant_utilization(data: UCPData, solution: Solution) -> ggplot:
    TPP = data.thermal_plants
    avg_coef = TPP.l_cost + (TPP.c_cost / TPP.max_power)
    avg_coef.name = "avg_coef"

    avg_coef = pd.concat([avg_coef, TPP["plant"]], axis=1)

    utilization = solution["p"].groupby(level="plant").sum().reset_index(0)
    utilization = utilization.merge(TPP[["plant", "max_power"]])
    utilization["utilization"] = utilization["p"] / (utilization["max_power"] * len(data.loads))

    temp = avg_coef[["plant", "avg_coef"]].merge(utilization, sort=True)[["plant", "utilization", "avg_coef"]]

    return (
        ggplot(temp, aes("avg_coef", "utilization", label="plant"))
        + geom_label()
        + scale_y_continuous(labels=percent_format())
        + labs(
            x="Avg. Hourly cost [€/MWh]", y="Utilization % (total production/total max production)"
        )
        + ggtitle("Utilization vs Hourly cost")
    )


def electricity_prices(
    data: UCPData, solution: Solution, model: MathematicalProgram = None, **solver_options
) -> ggplot:
    model = model if model is not None else create_model(data)
    energy_prices = (
        compute_multipliers(model, solution, **solver_options)["demand_satisfaction"].to_frame().reset_index()
    )
    energy_prices.columns = ["period", "electricity_price"]
    return (
        ggplot(energy_prices, aes("period", "electricity_price"))
        + geom_step(size=2, color="darkorange")
        + ggtitle("Hourly Electricity Prices")
        + labs(x="period", y="hourly Electricity price [€/MWh]")
    )
