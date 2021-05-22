import logging

from pulp import PULP_CBC_CMD, LpStatusOptimal

from UCP.input.data import UCPData
from UCP.model.ucp import create_model
from generic.optimization.model import Solution
from generic.optimization.solution_extraction import (
    compute_multipliers,
    extract_solution,
)

logger = logging.getLogger(__name__)


def local_search(
    data: UCPData, solution: Solution, change_points=5, radius=2, **solver_options
) -> Solution:
    """
    Computes a new UCP solution by performing local-search on an already existing solution using a MIP solver.
    The algorithm is allowed to change commitments in some periods. It selects the periods with the highest
     absolute "energy price", estimated via the lagrangian multipliers.

    :param data: data for the UCP problem
    :param solution: current solution
    :param change_points: number of periods in the horizon in which commitments can be changed
    :param radius: for each change_point t, consider the periods in [t-2, t+2] to be changed.
    :param solver_options: options to be passed to the solver.
    :return: a new solution.
    """
    # create model
    model = create_model(data)

    # find time periods to change
    energy_prices = compute_multipliers(model, solution, **solver_options)[
        "demand_satisfaction"
    ]
    abs_nn_energy_prices = energy_prices[
        (energy_prices > 0.5) | (energy_prices < -0.5)
    ].abs()
    periods_to_change = (
        abs_nn_energy_prices.sort_values(ascending=False)
        .astype(int)
        .index[0:change_points]
    )

    cost = solution["total_production_cost"]
    demand_mismatch_cost = solution["demand_mismatch_cost"]
    logger.warning(
        f"Current cost: {cost:5.2g}. Demand mismatch cost:{demand_mismatch_cost:5.2g}"
    )

    # fix the commitments
    min_time, max_time = data.loads["period"].min(), data.loads["period"].max()

    for (plant, time), v in model.vars["s"].items():
        v.lowBound, v.upBound = solution["s"][plant, time], solution["s"][plant, time]

    # let the commitments in the periods to change free
    for time in periods_to_change:
        for t in range(max(time - radius, min_time), min(time + radius, max_time)):
            for plant in data.thermal_plants["plant"]:
                v = model.vars["s"][plant, t]
                v.lowBound, v.upBound = 0, 1

    # solve the model
    status = model.solve(PULP_CBC_CMD(**solver_options))
    assert status == LpStatusOptimal, "Model in local search not solved optimally"
    new_solution = extract_solution(model)

    new_cost = new_solution["total_production_cost"]
    new_demand_mismatch_cost = new_solution["demand_mismatch_cost"]
    logger.warning(
        f"New cost: {new_cost:5.2g}."
        + f"New demand mismatch cost:{new_demand_mismatch_cost:5.2g}. "
        + f"Improvement: {1-(new_cost/cost):5.2%}"
    )

    return new_solution
