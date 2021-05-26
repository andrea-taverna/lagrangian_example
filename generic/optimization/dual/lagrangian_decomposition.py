import collections
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import OrderedDict, Tuple, NamedTuple
import pandas as pd
import numpy as np

from generic.optimization.model import Solution, OptimizationSense

SeriesBundle = NamedTuple("Bundle", [("intercept", float), ("subgradient", OrderedDict[str, pd.Series])])


@dataclass
class PrimalInformation:
    solution: Solution
    objective: float
    infeasibilities: OrderedDict[str, pd.Series]

    @staticmethod
    def dummy(sense: OptimizationSense) -> "PrimalInformation":
        return PrimalInformation(objective=-sense * np.infty, solution={}, infeasibilities=collections.OrderedDict({}))


@dataclass
class DualInformation:
    solution: OrderedDict[str, pd.Series]
    objective: float
    bundle: SeriesBundle

    @staticmethod
    def dummy(sense: OptimizationSense) -> "DualInformation":
        return DualInformation(
            objective=sense * np.infty,
            solution=collections.OrderedDict({}),
            bundle=SeriesBundle(np.nan, collections.OrderedDict({})),
        )


@dataclass
class LagrangianDecomposition(metaclass=ABCMeta):
    sense: OptimizationSense

    @abstractmethod
    def fill_multipliers(self, value: float = 0.0) -> OrderedDict[str, pd.Series]:
        pass

    @abstractmethod
    def multipliers_range(self) -> Tuple[OrderedDict[str, pd.Series], OrderedDict[str, pd.Series]]:
        pass

    @abstractmethod
    def _original_objective(self, solution: Solution) -> float:
        pass

    @abstractmethod
    def linearization_intercept(self, solution: Solution) -> float:
        pass

    def linearization(self, solution) -> Tuple[float, OrderedDict[str, pd.Series]]:
        return self.linearization_intercept(solution), self.subgradient(solution)

    @abstractmethod
    def violations(self, solution: Solution) -> OrderedDict[str, pd.Series]:
        pass

    @abstractmethod
    def infeasibilities(self, solution) -> OrderedDict[str, pd.Series]:
        pass

    def subgradient(self, solution: Solution) -> OrderedDict[str, pd.Series]:
        violations = self.violations(solution)
        return collections.OrderedDict([(k, -self.sense.value * v) for k, v in violations.items()])

    @abstractmethod
    def evaluate(self, multipliers: OrderedDict[str, pd.Series], **kwargs) -> Tuple[PrimalInformation, DualInformation]:
        pass

    def value(self, solution: Solution, multipliers: OrderedDict[str, pd.Series]) -> float:
        original_obj = self._original_objective(solution)
        lagrangian_penalty = sum(multipliers[k] @ v for k, v in self.violations(solution))
        return original_obj - self.sense.value * lagrangian_penalty

    @abstractmethod
    def information_from_primal_solution(self, solution, **kwargs) -> Tuple[PrimalInformation, SeriesBundle]:
        pass
