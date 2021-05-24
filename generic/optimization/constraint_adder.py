from pulp import LpProblem, LpAffineExpression

def add_constraint(model:LpProblem, expression: LpAffineExpression, name:str):
    """
    Adds a constraint to a model and returns the constraint object
    Rationale: PuLP's  `model.add_constraint` does not return the constraint object.

    Args:
        model: model to add the constraint to.
        expression: the mathematical expression of the constraint
        name: name of the constraint

    Returns:
        the LpConstraint object for the added constraint
    """
    model.addConstraint(expression, name=name)
    return model.constraints[name]