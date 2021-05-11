#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools

import numpy as np
import pandas as pd
from plotnine import *
from pulp import COIN_CMD
#from wurlitzer import sys_pipes

import generic.optimization.solution_extraction as out
import UCP.input.parser as ucp_parser
import UCP.model.ucp as original_model
import UCP.output.check_solution as ck
from generic.optimization.dual.algorithms.cutting_plane import *
from generic.optimization.dual.algorithms.subgradient import *
from generic.optimization.dual.combined_algorithm import (
    AlgorithmConfiguration,
    BoundsTracker,
    CombinedAlgorithm,
    KpiCollector,
    extract_kpis,
    pretty_printer,
    charts as algo_charts
)
from generic.optimization.dual.lagrangian_decomposition import (
    fill_multipliers,
    multipliers_range,
)
from generic.optimization.model import OptimizationSense
from generic.optimization.solution_extraction import extract_solution
from generic.series_dict import (
    full_series_dict,
    series_dict_indexes,
    series_dict_to_array,
)
from UCP.heuristics.commit_dispatch import commit_dispatch_heuristic
from UCP.heuristics.rolling_horizon import rolling_horizon_heuristic
from UCP.output import charts
from UCP.relaxations.heuristic import combinatorial_heuristic
from UCP.relaxations.lagrangian.production_state.relaxation import (
    ProductionStateRelaxation,
)


# In[2]:


theme_set(theme_bw() + theme(figure_size=(10, 10 / 1.61)))


# In[3]:


data = ucp_parser.read_instance("./UCP/data/instance_week1.ucp")


# ## Full optimization with Solver

# In[4]:


ucp = original_model.create_model(data)
with sys_pipes():
    ucp.model.solve(solver=COIN_CMD(mip=1, options=["node depth seconds 180"]))

optimal_cost = optimal_value = ucp.model.objective.value()
print(f"Cost: {optimal_cost}")


# # Optimization with Lagrangian Decomposition

# ## Setup

# In[18]:


relaxation = ProductionStateRelaxation(data)

var_lb, var_ub = tuple(
    series_dict_to_array(**m)
    for m in multipliers_range(relaxation.dualized_constraints)
)
bounds_tracker = BoundsTracker(sense=relaxation.sense)


# In[19]:


upper_bound = np.nan
lower_cost_estimate = data.loads["value"].sum() * data.thermal_plants["l_cost"].min()

fixed_tracker = FixedTargetTracker(sense=-relaxation.sense, overestimation_factor=0.35)


def target_tracker(current_value: float) -> float:
    upper_bound = bounds_tracker.best_primal_solution.objective
    if abs(upper_bound - current_value) / abs(current_value + 1e-3) > 0.2:
        fixed_target = fixed_tracker(current_value)
        result = max(fixed_target, lower_cost_estimate)
    else:
        result = upper_bound
    return result


# In[20]:


sgd_algorithms = {
    "Polyak+CFM": (
        "ComposableSubgradientMethod",
        dict(
            sense=-relaxation.sense,
            var_lb=var_lb,
            var_ub=var_ub,
            step_size_fun=PolyakStepSizeRule(-relaxation.sense, target_tracker),
            deflection_fun=CFMDeflection(),
        ),
    ),
    "Volume": (
        "VolumeAlgorithm",
        dict(
            sense=-relaxation.sense,
            var_lb=var_lb,
            var_ub=var_ub,
            step_size_fun=PolyakStepSizeRule(-relaxation.sense, target_tracker),
        ),
    ),
    "Bundle_Volume": (
        "BundleVolumeAlgorithm",
        dict(
            sense=-relaxation.sense,
            var_lb=var_lb,
            var_ub=var_ub,
            step_size_fun=PolyakStepSizeRule(-relaxation.sense, target_tracker),
        ),
    ),
}


# In[21]:


make_min_desc = lambda: MinDescentPolicy(
    sense=-relaxation.sense, descent_coefficient=0.0
)
stabilizer = TwoThresholdsParameterUpdater(
    0.05,
    200,
    bad_step=StepData(threshold=-1, update=2),
    good_step=StepData(threshold=0.2, update=0.9),
)
prox_stabilizer = SimpleProximalStabilizer(stabilizer)
stabilizer2 = TwoThresholdsParameterUpdater(
    low_bound=0.05,
    up_bound=1,
    bad_step=StepData(threshold=0, update=0.5),
    good_step=StepData(threshold=0.2, update=1.25),
)
trm_stabilizer = SimpleTrustRegionStabilizer((var_lb, var_ub), stabilizer2)
bundle_methods = {
    "CPM": (
        "CuttingPlane",
        dict(sense=-relaxation.sense, var_lb=var_lb, var_ub=var_ub),
    ),
    "PBM": (
        "ProximalBundle",
        dict(
            sense=-relaxation.sense,
            var_lb=var_lb,
            var_ub=var_ub,
            initial_stabilization_parameter=20,
            center_updater=make_min_desc(),
            stabilize_rule=prox_stabilizer,
        ),
    ),
    "IOB": (
        "InOutBundle",
        dict(
            sense=-relaxation.sense,
            var_lb=var_lb,
            var_ub=var_ub,
            initial_smoothing_factor=0.7,
            # center_updater=make_min_desc(),
        ),
    ),
    "TRM": (
        "TrustRegionBundle",
        dict(
            sense=-relaxation.sense,
            var_lb=var_lb,
            var_ub=var_ub,
            initial_box_size=0.05,
            center_updater=make_min_desc(),
            box_size_updater=trm_stabilizer,
        ),
    ),
    "TRLM": (
        "TrustRegionLevelBundle",
        dict(
            sense=-relaxation.sense,
            var_lb=var_lb,
            var_ub=var_ub,
            initial_box_size=0.1,
            initial_overestimation_factor=0.5,
            center_updater=make_min_desc(),
            box_size_updater=trm_stabilizer,
        ),
    ),
    "ParetoTRM": (
        "ParetoCutBundle",
        dict(
            sense=-relaxation.sense,
            var_lb=var_lb,
            var_ub=var_ub,
            initial_box_size=0.1,
            initial_overestimation_factor=0.1,
            center_updater=make_min_desc(),
            box_size_updater=trm_stabilizer,
        ),
    ),
}


# In[22]:


def run_heuristic(primal_solutions):
    mip = 0
    commitments = [ps["s"] for ps in primal_solutions]
    fixed_model, _ = combinatorial_heuristic(
        data,
        commitments,
        combination_options=dict(
            mip=mip, options=["ratio", "0.05", "sec", "40", "doh", "node depth"]
        ),
    )
    solution = extract_solution(fixed_model)
    print(
        f" => Heuristic:\tTotal cost: {solution['total_production_cost']:15.5g}\t"
        + f"Demand mismatch cost:{solution['demand_mismatch_cost']:15.5g}"
    )
    return relaxation.information_from_primal_solution(solution)


# ## Initialization

# In[23]:


initial_heuristic = False
if initial_heuristic:
    ucp_model, solution = rolling_horizon_heuristic(
        data, 72, 72, options=["sec 25 doh node depth"]
    )
    primal_info = relaxation.information_from_primal_solution(solution)
    initial_multipliers = relaxation.multipliers_from_primal_solution(solution)
    initial_solution = primal_info
else:
    initial_solution = None
    initial_multipliers = fill_multipliers(relaxation.dualized_constraints, 0.0)


# In[24]:


configuration = AlgorithmConfiguration(
    primal_feasibility_tolerance=1e-4,
    subgradient_tolerance=1e-6,
    sgd_iterations=5,
    cp_iterations=1,
    sgd_start=True,
    heuristic_frequency=10,
    max_gap=0.025,
    max_cp_gap=0.01,
    relaxation_solver_options={},
    cp_solver_options={"options": ["seconds 60"]},
    max_iterations=60,
    sgd_name="Polyak+CFM",
    cp_name="CPM",
)


# In[25]:


combined_algorithm = CombinedAlgorithm(
    configuration,
    relaxation,
    initial_multipliers,
    bundle_methods[configuration.cp_name],
    sgd_algorithms[configuration.sgd_name],
    run_heuristic,
    initial_heuristic_solution=initial_solution,
    bounds_tracker=bounds_tracker,
)


# ## Execution

# In[26]:


kpi_collector = KpiCollector()
print(pretty_printer.header() + f"|{'Gap% from Optimum*':>25}")
stop = False
while not stop:
    stop = combined_algorithm()
    kpi_collector.collect(combined_algorithm)
    print(
        pretty_printer.row(kpi_collector)
        + f"|{abs(1- (bounds_tracker.best_primal_solution.objective/optimal_value)):>25.2%}"
    )


# ## Output

# In[14]:


kpis = extract_kpis(kpi_collector.table())
algo_charts.bounds(kpis)


# In[15]:


solution = bounds_tracker.best_primal_solution.solution


# In[16]:


feasible, analysis = ck.check_solution(data, solution)
assert feasible , "ERROR: final solution is infeasible."


# In[17]:


plots = [
    charts.total_production(data, solution),
    charts.enp_vs_eie(data, solution),
    charts.production_by_plant(data, solution),
    charts.plant_utilization(data, solution),
]

for p in plots:
    print(p)


# In[ ]:




