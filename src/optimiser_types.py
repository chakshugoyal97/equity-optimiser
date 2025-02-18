import numpy as np
from dataclasses import dataclass

@dataclass
class OptimiserOutput:
    weights: np.ndarray
    expected_return: float
    risk: float

    def __iter__(self):
        return iter((self.weights, self.expected_return, self.risk))

