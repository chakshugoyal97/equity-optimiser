import numpy as np
import pytest
from optimiser import EquityOptimiser
import logging

logger = logging.getLogger(__name__)


def _get_return_and_covariance(n, mu, sigma):
    # normal sampling from N(mu, sigma)
    expected_returns = sigma * np.random.randn(n, 1) + mu

    # for symmetrix, PSD arrays
    covariance_matrix = np.random.rand(n, n)
    covariance_matrix = np.dot(covariance_matrix, covariance_matrix.T)

    return expected_returns, covariance_matrix


@pytest.mark.parametrize(
    "n,mu,sigma", [(3, 0.1, 0.2), (15, 0.07, 0.05), (20, 0.5, 0.8)]
)
def test_optimiser_basic(n, mu, sigma):
    # GIVEN
    n = 3
    expected_returns, covariance_matrix = _get_return_and_covariance(n, mu, sigma)
    eo = EquityOptimiser(expected_returns, covariance_matrix)

    # WHEN
    w = eo.optimise()

    # THEN
    logger.info(f"optimal weights: {w}")
    assert w.shape == (n,)


@pytest.mark.parametrize("w_min,w_max", [(-1, 1), (-5, None), (None, 3)])
def test_optimiser_weight_limits(w_min, w_max):
    # GIVEN
    n = 3
    expected_returns, covariance_matrix = _get_return_and_covariance(n, 0.1, 0.2)
    eo = EquityOptimiser(expected_returns, covariance_matrix)
    eo.add_criteria_weights(w_min, w_max)

    # WHEN
    w = eo.optimise()

    # THEN
    logger.info(f"optimal weights: {w}")
    assert w.shape == (n,)
    tol = 1e-8
    if w_min:
        assert np.all(w_min - tol <= w)
    if w_max:
        assert np.all(w <= w_max + tol)
