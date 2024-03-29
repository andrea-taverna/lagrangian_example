import numpy as np
from typing import List, Tuple
from pulp import LpProblem, LpVariable, LpMaximize, LpMinimize, LpSenses, LpConstraint, lpSum, PULP_CBC_CMD, LpStatusOptimal

from generic.optimization.model import OptimizationSense


class CPLPModel:
    """
    Cutting Plane model as a Linear Program (CPLP)

    Attributes:
        model: LP model
        optimistic_bound: expected value of the actual function according to current cuts
        variables: model variables
        sense: either LpMinimize or LpMaximize
    """

    model: LpProblem
    optimistic_bound: LpVariable
    variables = List[LpVariable]
    sense: LpSenses

    def __init__(self, sense: OptimizationSense, var_lb: np.ndarray, var_ub: np.ndarray):
        """
        Creates a CPLP with empty cut set
        Args:
            sense: optimization sense
            var_lb: variables lower bound
            var_ub: variables upper bound

        """
        num_vars = len(var_lb)
        self.sense = sense
        self.model = LpProblem("CuttingPlane")
        self.optimistic_bound = LpVariable(name="z")
        self.variables = [LpVariable(f"lambda_{j}", lowBound=var_lb[j], upBound=var_ub[j]) for j in range(num_vars)]
        for v in self.variables:
            # see https://github.com/coin-or/pulp/issues/331#issuecomment-681965566
            v.setInitialValue(v.lowBound)

        self.model.setObjective(self.optimistic_bound)
        self.model.sense = LpMaximize if self.sense == OptimizationSense.MAX else LpMinimize

    def add_cut(self, intercept: float, subgradient: np.ndarray):
        """
        Adds a cut to the LP model
        Args:
            intercept: constant term of the cut
            subgradient: subgradient/coefficients of the cut

        """
        sign = int(self.sense.value)
        cons_expr = sign * self.optimistic_bound <= sign * (
            +lpSum(subgradient[j] * self.variables[j] for j in range(len(self.variables))) + intercept
        )
        self.model.addConstraint(cons_expr)

    def actual_solve(self, **kwargs) -> Tuple[int, float, np.ndarray]:
        """
        Calls the  solver on the LP model
        Args:
            **kwargs: additional arguments to the solver

        Returns:
            Tuple with:
                * status: solver's status
                * optimistic bound: expected value of the true function
                * variables' values: variables' values
        """
        status = self.model.solve(PULP_CBC_CMD(msg=0, **kwargs))
        var_values = np.array([self.variables[i].value() for i in range(len(self.variables))])
        return status, self.optimistic_bound.value(), var_values

    def solve(self, **kwargs) -> Tuple[float, np.ndarray]:
        status, optimistic_bound, variables = self.actual_solve(**kwargs)
        assert status == LpStatusOptimal
        return optimistic_bound, variables

