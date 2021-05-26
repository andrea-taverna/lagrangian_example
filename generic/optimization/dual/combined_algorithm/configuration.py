from typing import Dict, Callable, Any, List, Tuple

from dataclasses import dataclass, field

from generic.optimization.dual.lagrangian_decomposition import PrimalInformation, SeriesBundle
from generic.optimization.model import Solution

LARGE_VALUE = 1e80


@dataclass
class AlgorithmConfiguration:
    primal_feasibility_tolerance: float

    sgd_iterations: int
    cp_iterations: int
    heuristic_frequency: int

    max_iterations: int
    max_gap: float
    max_cp_gap: float
    subgradient_tolerance: float

    relaxation_solver_options: Dict[str, Any] = field(default_factory=dict())
    cp_solver_options: Dict[str, Any] = field(default_factory=dict())
    cp_name: str = "CP"
    sgd_name: str = "SubGd"


PrimalHeuristicAlgorithm = Callable[[List[Solution]], Tuple[PrimalInformation, SeriesBundle]]
