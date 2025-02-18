import numpy as np
import cvxpy as cp
from dataclasses import dataclass

@dataclass
class Constraint:
    description: str
    constraint: cp.Constraint

@dataclass
class OptimiserOutput:
    weights: np.ndarray
    expected_return: float
    risk: float

    def __iter__(self):
        return iter((self.weights, self.expected_return, self.risk))

