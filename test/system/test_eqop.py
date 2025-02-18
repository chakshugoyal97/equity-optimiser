import logging
from typing import Optional

import numpy as np
import pytest

from constants import TOL
from optimiser import EquityOptimiser

logger = logging.getLogger(__name__)

np.random.seed(73)  # for reproducibility


def _get_return_and_covariance(n: int, mu: float, sigma: float):
    # normal sampling from N(mu, sigma^2)
    expected_returns = sigma * np.random.randn(n, 1) + mu

    # for symmetrix, PSD arrays
    # needed because the risk term, (w^T @ Cov @ w^T) needs to be >= 0 always for convex optimsation
    covariance_matrix = np.random.rand(n, n)
    covariance_matrix = np.dot(covariance_matrix, covariance_matrix.T)

    return expected_returns, covariance_matrix


@pytest.mark.parametrize(
    "n,mu,sigma", [(3, 0.1, 0.2), (15, 0.07, 0.05), (20, 0.5, 0.8)]
)
def test_optimiser_basic(n: int, mu: float, sigma: float):
    # GIVEN
    expected_returns, covariance_matrix = _get_return_and_covariance(n, mu, sigma)

    # WHEN
    eo = EquityOptimiser(expected_returns, covariance_matrix)
    w_opt, mu_opt, sigma_opt = eo.optimise()

    # THEN
    assert w_opt.shape == (n,)
    assert np.isclose(np.sum(w_opt), 1, TOL)


@pytest.mark.parametrize(
    "n,mu,sigma,w_min,w_max",
    [(10, 0.1, 0.2, -1, 1), (10, 0.1, 0.2, -5, None), (10, 0.1, 0.2, None, 3)],
)
def test_optimiser_weight_limits(
    n: int, mu: float, sigma: float, w_min: Optional[float], w_max: Optional[float]
):
    # GIVEN
    expected_returns, covariance_matrix = _get_return_and_covariance(n, mu, sigma)

    # WHEN
    eo = EquityOptimiser(expected_returns, covariance_matrix)
    eo.add_criteria_weights(w_min, w_max)
    w_opt, mu_opt, sigma_opt = eo.optimise()

    # THEN
    assert w_opt.shape == (n,)
    assert np.isclose(np.sum(w_opt), 1, TOL)
    if w_min:
        assert np.all(w_min - TOL <= w_opt)
    if w_max:
        assert np.all(w_opt <= w_max + TOL)


@pytest.mark.parametrize(
    "n,mu,sigma,mu_min", [(15, 0.07, 0.05, 0.1), (20, 0.5, 0.8, 0.6)]
)
def test_optimiser_min_return(n: int, mu: float, sigma: float, mu_min: float):
    # GIVEN
    expected_returns, covariance_matrix = _get_return_and_covariance(n, mu, sigma)

    # WHEN
    eo = EquityOptimiser(expected_returns, covariance_matrix)
    eo.add_criteria_return_target(mu_min)
    w_opt, mu_opt, sigma_opt = eo.optimise()

    # THEN
    assert w_opt.shape == (n,)
    assert np.isclose(np.sum(w_opt), 1, TOL)
    assert mu_opt >= mu_min - TOL


@pytest.mark.parametrize(
    "n,mu,sigma,sigma_max", [(15, 0.07, 0.05, 0.44), (20, 0.5, 0.8, 0.6)]
)
def test_optimiser_max_risk(n: int, mu: float, sigma: float, sigma_max: float):
    # GIVEN
    expected_returns, covariance_matrix = _get_return_and_covariance(n, mu, sigma)

    # WHEN
    eo = EquityOptimiser(expected_returns, covariance_matrix)
    eo.add_criteria_risk_level(sigma_max)
    w_opt, mu_opt, sigma_opt = eo.optimise()

    # THEN
    assert w_opt.shape == (n,)
    assert np.isclose(np.sum(w_opt), 1, TOL)
    assert (sigma_opt - TOL) <= sigma_max


@pytest.mark.parametrize("n,mu,sigma,k,max_limit", [(10, 0.1, 0.5, 3, 0.6)])
def test_optimiser_top_k(n: int, mu: float, sigma: float, k: int, max_limit: float):
    # GIVEN
    expected_returns, covariance_matrix = _get_return_and_covariance(n, mu, sigma)

    # WHEN
    eo = EquityOptimiser(expected_returns, covariance_matrix)
    eo.add_criteria_limit_top_k_allocations(k, max_limit)
    eo.add_criteria_weights(0, 0.3)
    eo.add_criteria_return_target(0.25, 0.26)
    eo.add_criteria_risk_level(1.9)
    w_opt, mu_opt, sigma_opt = eo.optimise(0.05)

    # THEN
    assert w_opt.shape == (n,)
    assert np.isclose(np.sum(w_opt), 1, TOL)
    w_top_k = np.sum(np.sort(np.abs(w_opt))[-k:])
    assert w_top_k <= max_limit + TOL


@pytest.mark.parametrize(
    "n,mu,sigma,adv_multiple",
    [(3, 0.1, 0.2, 0.01), (15, 0.07, 0.05, 0.002), (20, 0.5, 0.8, 0.005)],
)
def test_optimiser_max_adv(n: int, mu: float, sigma: float, adv_multiple: float):
    # GIVEN
    expected_returns, covariance_matrix = _get_return_and_covariance(n, mu, sigma)
    w_prev = np.full(n, 1.0) / n
    volume = 1e6
    adv = np.random.uniform(1e6, 1e7, n)

    # WHEN
    eo = EquityOptimiser(expected_returns, covariance_matrix, w_prev)
    eo.add_criteria_max_adv_equity(adv_multiple, volume, adv)
    w_opt, mu_opt, sigma_opt = eo.optimise()

    # THEN
    assert w_opt.shape == (n,)
    assert np.isclose(np.sum(w_opt), 1, TOL)

    volume_traded = np.abs(w_opt - w_prev) * volume
    volume_allowed = adv_multiple * adv
    volume_buffer = volume_allowed - volume_traded
    assert np.all(volume_buffer + 1e-3 * volume >= 0)


@pytest.mark.parametrize(
    "n,mu,sigma,txn_cost",
    [
        (10, 0.1, 0.2, np.full(10, 0.01)),  # Small costs
        (15, 0.07, 0.05, np.linspace(0.005, 0.02, 15)),  # Varying costs
    ],
)
def test_optimiser_txn_costs(n: int, mu: float, sigma: float, txn_cost: np.ndarray):
    # GIVEN
    expected_returns, covariance_matrix = _get_return_and_covariance(n, mu, sigma)
    w_prev = np.full(n, 1.0) / n

    # WHEN
    eo = EquityOptimiser(expected_returns, covariance_matrix, w_prev)
    eo.set_txn_costs(txn_cost)
    w_opt, mu, sigma = eo.optimise()

    # THEN
    assert w_opt.shape == (n,)
    assert np.isclose(np.sum(w_opt), 1, TOL)

    # Ensure transaction costs impact the portfolio:
    txn_cost_impact = np.sum(txn_cost * np.abs(w_opt))
    assert txn_cost_impact > 0  # Transaction costs should penalize large weights
