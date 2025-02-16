import numpy as np
from optimiser import EquityOptimiser

def test_optimiser():
    # GIVEN
    n = 15
    mu = 0.1
    sigma = 0.2
    expected_returns = sigma * np.random.randn(n, 1) + mu
    covariance_matrix = sigma * np.random.rand(n, n)
    covariance_matrix = covariance_matrix + covariance_matrix.T # for symmetrix matrix

    EqOp = EquityOptimiser(expected_returns, covariance_matrix)

    # WHEN
    w = EqOp.optimise()

    # THEN
    assert w.shape == (n, 1)
    print(w)

    