from dataclasses import dataclass

import numpy as np


@dataclass
class OptimiserOutput:
    weights: np.ndarray
    expected_return: float
    risk: float

    def __iter__(self):
        return iter((self.weights, self.expected_return, self.risk))
