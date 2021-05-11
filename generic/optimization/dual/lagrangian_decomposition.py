import collections
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import OrderedDict, Tuple, NamedTuple
import pandas as pd
import numpy as np

from generic.optimization.dual.dualized_constraints import DualizedConstraint, ConstraintSense
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


def multipliers_range(
    dualized_cons: OrderedDict[str, DualizedConstraint]
) -> Tuple[OrderedDict[str, pd.Series], OrderedDict[str, pd.Series]]:
    lower_bounds = collections.OrderedDict([(k, dc.multipliers_lb) for k, dc in dualized_cons.items()])
    upper_bounds = collections.OrderedDict([(k, dc.multipliers_ub) for k, dc in dualized_cons.items()])
    return lower_bounds, upper_bounds


def fill_multipliers(
    dualized_constraints: OrderedDict[str, DualizedConstraint], value=0.0
) -> OrderedDict[str, pd.Series]:
    return collections.OrderedDict(
        [(k, pd.Series(np.full(len(v.index), value), index=v.index)) for k, v in dualized_constraints.items()]
    )


@dataclass
class LagrangianDecomposition(metaclass=ABCMeta):
    sense: OptimizationSense
    dualized_constraints: OrderedDict[str, DualizedConstraint]

    @abstractmethod
    def rhs(self) -> OrderedDict[str, pd.Series]:
        pass

    @abstractmethod
    def lhs(self, solution: Solution) -> OrderedDict[str, pd.Series]:
        pass

    def violations(self, solution: Solution) -> OrderedDict[str, pd.Series]:
        lhs = self.lhs(solution)
        rhs = self.rhs()
        return collections.OrderedDict(
            [(k, dc.sense.sign * (rhs[k] - lhs[k])) for k, dc in self.dualized_constraints.items()]
        )

    @abstractmethod
    def _original_objective(self, solution: Solution) -> float:
        pass

    @abstractmethod
    def evaluate(self, multipliers: OrderedDict[str, pd.Series], **kwargs) -> Tuple[PrimalInformation, DualInformation]:
        pass

    def value(self, solution: Solution, multipliers: OrderedDict[str, pd.Series]) -> float:
        original_obj = self._original_objective(solution)
        lagrangian_penalty = sum(multipliers[k] @ v for k, v in self.violations(solution))
        return original_obj - self.sense.value * lagrangian_penalty

    def infeasibilities(self, solution) -> OrderedDict[str, pd.Series]:
        violations = self.violations(solution)
        senses = {k: dc.sense for k, dc in self.dualized_constraints.items()}
        infeasibilities = collections.OrderedDict(
            [
                (k, v.transform(lambda x: max(0.0, x)) if senses[k] != ConstraintSense.EQUAL else abs(v))
                for k, v in violations.items()
            ]
        )
        return infeasibilities

    def subgradient(self, solution: Solution) -> OrderedDict[str, pd.Series]:
        violations = self.violations(solution)
        return collections.OrderedDict([(k, -self.sense.value * v) for k, v in violations.items()])

    @abstractmethod
    def linearization_intercept(self, solution: Solution) -> float:
        pass

    def linearization(self, solution) -> Tuple[float, OrderedDict[str, pd.Series]]:
        return self.linearization_intercept(solution), self.subgradient(solution)

    @abstractmethod
    def information_from_primal_solution(self, solution, **kwargs) -> Tuple[PrimalInformation, SeriesBundle]:
        pass

    @abstractmethod
    def multipliers_from_primal_solution(self, solution, **kwargs) -> OrderedDict[str, pd.Series]:
        pass
