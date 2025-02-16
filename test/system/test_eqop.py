from typing import Optional
import numpy as np
import pytest
from optimiser import EquityOptimiser
import logging

logger = logging.getLogger(__name__)

tol = 1e-4

def _get_return_and_covariance(n: int, mu: float, sigma: float):
    # normal sampling from N(mu, sigma)
    expected_returns = sigma * np.random.randn(n, 1) + mu

    # for symmetrix, PSD arrays
    covariance_matrix = np.random.rand(n, n)
    covariance_matrix = np.dot(covariance_matrix, covariance_matrix.T)

    return expected_returns, covariance_matrix


@pytest.mark.parametrize(
    "n,mu,sigma", [(3, 0.1, 0.2), (15, 0.07, 0.05), (20, 0.5, 0.8)]
)
def test_optimiser_basic(n: int, mu: float, sigma: float):
    # GIVEN
    n = 3
    expected_returns, covariance_matrix = _get_return_and_covariance(n, mu, sigma)
    eo = EquityOptimiser(expected_returns, covariance_matrix)

    # WHEN
    w, mu, sigma = eo.optimise()

    # THEN
    logger.info(f"optimal weights: {w}")
    logger.info(f"mu: {mu}, sigma: {sigma}")

    assert w.shape == (n,)


@pytest.mark.parametrize("w_min,w_max", [(-1, 1), (-5, None), (None, 3)])
def test_optimiser_weight_limits(w_min: Optional[float], w_max: Optional[float]):
    # GIVEN
    n = 3
    expected_returns, covariance_matrix = _get_return_and_covariance(n, 0.1, 0.2)
    eo = EquityOptimiser(expected_returns, covariance_matrix)
    eo.add_criteria_weights(w_min, w_max)

    # WHEN
    w, mu, sigma = eo.optimise()

    # THEN
    # logger.info(f"optimal weights: {w}")
    # logger.info(f"mu: {mu}, sigma: {sigma}")

    assert w.shape == (n,)
    if w_min:
        assert np.all(w_min - tol <= w)
    if w_max:
        assert np.all(w <= w_max + tol)

@pytest.mark.parametrize(
    "n,mu,sigma,mu_min", [(15, 0.07, 0.05, 0.1), (20, 0.5, 0.8, 0.6)]
)
def test_optimiser_min_return(n: int, mu: float, sigma: float, mu_min: float):
    # GIVEN
    n = 3
    expected_returns, covariance_matrix = _get_return_and_covariance(n, mu, sigma)
    eo = EquityOptimiser(expected_returns, covariance_matrix)

    # WHEN
    eo.add_criteria_return_target(mu_min)
    w, mu, sigma = eo.optimise()

    # THEN
    # logger.info(f"optimal weights: {w}")
    # logger.info(f"mu: {mu}, sigma: {sigma}")
    assert w.shape == (n,)
    assert mu >= mu_min - tol

@pytest.mark.parametrize(
    "n,mu,sigma,sigma_max", [(15, 0.07, 0.05, 0.44), (20, 0.5, 0.8, 0.6)]
)
def test_optimiser_max_risk(n: int, mu: float, sigma: float, sigma_max: float):
    np.random.seed(67)
    # GIVEN
    n = 3
    expected_returns, covariance_matrix = _get_return_and_covariance(n, mu, sigma)
    eo = EquityOptimiser(expected_returns, covariance_matrix)

    # WHEN
    eo.add_criteria_risk_level(sigma_max)
    w, mu, sigma = eo.optimise()

    # THEN
    # logger.info(f"optimal weights: {w}")
    # logger.info(f"mu: {mu}, sigma: {sigma}")
    assert w.shape == (n,)
    assert sigma - tol <= sigma_max
