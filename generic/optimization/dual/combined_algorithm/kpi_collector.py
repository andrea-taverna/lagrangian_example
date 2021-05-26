from typing import List, Tuple, Any, Dict
import pandas as pd
import numpy as np

from generic.optimization.dual.combined_algorithm.combined_algorithm import CombinedAlgorithm
from generic.optimization.dual.combined_algorithm.bounds_tracker import BoundsTracker
from generic.series_dict import series_dict_to_array


def kpis_algorithm(algorithm: CombinedAlgorithm) -> Dict[str, Any]:
    return {
        "iteration": algorithm.iteration,
        "dual_algorithm": algorithm.dual_algorithm,
        "relaxation_primal": algorithm.relaxation_primal,
        "relaxation_dual": algorithm.relaxation_dual,
        "heuristic_solution": algorithm.heuristic_solution,
        "multipliers": algorithm.current_multipliers,
        "dual_bound": algorithm.relaxation_dual.objective,
        "master_bound": algorithm.master_bound,
    }


def kpis_bounds(bounds: BoundsTracker) -> Dict[str, Any]:
    return {
        "best_dual_bound": bounds.best_dual_solution.objective,
        "best_primal_bound": bounds.best_primal_solution.objective,
        "best_master_bound": bounds.best_master_bound,
        "optimality_gap": bounds.optimality_gap,
        "cutting_plane_gap": bounds.cutting_plane_gap,
    }


class KpiCollector:
    rows: Dict[str, List[Any]]

    def __init__(self):
        self.rows = {}

    def collect(self, algorithm: CombinedAlgorithm):
        kpis = {**kpis_algorithm(algorithm), **kpis_bounds(algorithm.bounds)}
        if len(self.rows) == 0:
            self.rows = {k: [] for k in kpis.keys()}
        for k, l in self.rows.items():
            l.append(kpis[k])

    def table(self):
        return pd.DataFrame.from_dict(self.rows).set_index("iteration")


def extract_kpis(table: pd.DataFrame):
    table = table.copy()
    table["v_multipliers"] = table["multipliers"].transform(lambda m: series_dict_to_array(**m))
    table["v_subgradient"] = table["relaxation_dual"].transform(lambda h: series_dict_to_array(**h.bundle.subgradient))
    table["v_infeasibilities"] = table["relaxation_primal"].transform(
        lambda h: series_dict_to_array(**h.infeasibilities)
    )

    v_multipliers = pd.DataFrame(
        dict(v_multipliers=table["v_multipliers"], prev_multipliers=table["v_multipliers"].shift())
    )
    table["multipliers_change"] = v_multipliers.apply(lambda r: np.linalg.norm(r[0] - r[1]), axis=1)
    norms = table.transform(
        dict(v_multipliers=np.linalg.norm, v_subgradient=np.linalg.norm, v_infeasibilities=np.linalg.norm)
    )
    norms.columns = [c + "_norm" for c in norms.columns]

    table = pd.concat([table, norms], axis=1)

    v_subgradients = pd.DataFrame(
        dict(v_multipliers=table["v_subgradient"], prev_multipliers=table["v_subgradient"].shift())
    )
    v_subgradients.iloc[0, 1] = v_subgradients.iloc[0, 0]
    table["subgradient_angle"] = v_subgradients.apply(lambda r: get_angle(r[0], r[1]), axis=1)

    return table.reset_index(drop=False)


def get_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    def normalise(v: np.ndarray) -> np.ndarray:
        return v / max(np.linalg.norm(v), 1e-3)

    angle = np.arccos(np.dot(normalise(v1), normalise(v2)))
    return angle * 180.0 / np.pi
