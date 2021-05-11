from pulp import LpProblem, LpAffineExpression

def add_constraint(model:LpProblem, expression: LpAffineExpression, name:str):
    model.addConstraint(expression, name=name)
    return model.constraints[name]