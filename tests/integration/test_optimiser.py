import logging
from typing import Optional

import numpy as np
import pytest

from constants import TOL
from optimiser import EquityOptimiser
from workbooks.download_data import load_optimization_data

logger = logging.getLogger(__name__)

np.random.seed(73)  # for reproducibility


def _get_return_and_covariance(n: int, mu: float, sigma: float):
    """
    generate dummy expected_returns and covariance of stocks
        - assuming exp(stock_return) itself follows some N(mu, sigma^2) distribution
        - irl it might be some pareto/exponential distribution
    """
    # normal sampling from N(mu, sigma^2)
    expected_returns = sigma * np.random.randn(n) + mu

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


# verify with an obvious case that we are long bull stock, and short bear stock
def test_simple_case_two_assets():
    # GIVEN
    expected_returns = np.array([0.1, -0.1])
    covariance_matrix = np.array([[2.0, -2.0], [-2.0, 2.0]])

    # WHEN
    eo = EquityOptimiser(expected_returns, covariance_matrix)
    eo.set_weights_bound(-1, 2)
    w_opt, mu_opt, sigma_opt = eo.optimise(lambda_=0.0)

    # THEN
    # Check that better performing mu had better weight
    assert np.isclose(w_opt[0], 2, 1e-3)
    assert np.isclose(w_opt[1], -1, 1e-3)


# verify how we handle infinite case
def test_failure_infinite_case():
    # GIVEN
    expected_returns = np.array([0.1, -0.1])
    covariance_matrix = np.array([[2.0, -2.0], [-2.0, 2.0]])

    # WHEN
    eo = EquityOptimiser(expected_returns, covariance_matrix)
    # AND THEN
    with pytest.raises(ValueError) as e:
        w_opt, mu_opt, sigma_opt = eo.optimise(lambda_=0)

    assert "unbounded" in str(e)


# verify how we handle infeasible case
def test_failure_infeasible_case():
    # GIVEN
    expected_returns = np.array([0.05, 0.04])
    covariance_matrix = np.array([[1.0, 1.0], [1.0, 1.0]])

    # WHEN
    eo = EquityOptimiser(expected_returns, covariance_matrix)
    eo.set_weights_bound(0, 1)
    eo.set_min_return(0.5)

    # AND THEN
    with pytest.raises(ValueError) as e:
        w_opt, mu_opt, sigma_opt = eo.optimise()

    assert "infeasible" in str(e)


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
    eo.set_weights_bound(w_min, w_max)
    w_opt, mu_opt, sigma_opt = eo.optimise()

    # THEN
    assert w_opt.shape == (n,)
    assert np.isclose(np.sum(w_opt), 1, TOL)
    if w_min:
        assert (w_min - TOL) <= np.min(w_opt)
    if w_max:
        assert np.max(w_opt) <= (w_max + TOL)


@pytest.mark.parametrize(
    "n,mu,sigma,mu_min", [(15, 0.07, 0.05, 0.1), (20, 0.5, 0.8, 0.6)]
)
def test_optimiser_min_return(n: int, mu: float, sigma: float, mu_min: float):
    # GIVEN
    expected_returns, covariance_matrix = _get_return_and_covariance(n, mu, sigma)

    # WHEN
    eo = EquityOptimiser(expected_returns, covariance_matrix)
    eo.set_min_return(mu_min)
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
    eo.set_max_risk(sigma_max)
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
    eo.set_top_k_limit(k, max_limit)
    eo.set_weights_bound(0, 0.3)
    eo.set_min_return(0.25, 0.26)
    eo.set_max_risk(1.9)
    w_opt, mu_opt, sigma_opt = eo.optimise(0.05)

    # THEN
    assert w_opt.shape == (n,)
    assert np.isclose(np.sum(w_opt), 1, TOL)
    w_top_k = np.sum(np.sort(np.abs(w_opt))[-k:])
    assert w_top_k <= max_limit + TOL


@pytest.mark.parametrize(
    "n,mu,sigma,adv_multiple",
    [(3, 0.1, 0.2, 0.001), (15, 0.07, 0.05, 0.002), (20, 0.5, 0.8, 0.005)],
)
def test_optimiser_max_adv(n: int, mu: float, sigma: float, adv_multiple: float):
    # GIVEN
    expected_returns, covariance_matrix = _get_return_and_covariance(n, mu, sigma)
    logger.info(f"stocks mu: {expected_returns}")
    w_prev = np.full(n, 1.0) / n
    nav = 1e6
    adv = np.random.uniform(1e6, 1e7, n)

    # WHEN
    eo = EquityOptimiser(expected_returns, covariance_matrix, w_prev)
    eo.set_volume_adv_threshold(adv_multiple, nav, adv)
    w_opt, mu_opt, sigma_opt = eo.optimise()

    # THEN
    assert w_opt.shape == (n,)
    assert np.isclose(np.sum(w_opt), 1, TOL)

    volume_traded = np.abs(w_opt - w_prev) * nav
    volume_allowed = adv_multiple * adv
    volume_buffer_left = volume_allowed - volume_traded
    logger.info(f"{volume_traded}, {volume_allowed}, {volume_buffer_left}")
    assert np.all(volume_buffer_left/nav + TOL >= 0)


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


# verify that higher turnover penalty leads to less change in portfolio weights
@pytest.mark.parametrize("n,mu,sigma", [(5, 0.1, 0.2)])
def test_turnover_penalty(n, mu, sigma):
    expected_returns, covariance_matrix = _get_return_and_covariance(n, mu, sigma)
    prev_weights = np.ones(n) / n

    eo = EquityOptimiser(expected_returns, covariance_matrix, prev_weights)

    # Optimize with different turnover penalties
    w_opt_1, mu_opt_1, sigma_opt_1 = eo.optimise(turnover_f_=0.1)
    w_opt_2, mu_opt_2, sigma_opt_2 = eo.optimise(turnover_f_=1.0)

    # Check that higher turnover penalty leads to less change in weights
    turnover_1 = np.sum(np.abs(w_opt_1 - prev_weights))
    turnover_2 = np.sum(np.abs(w_opt_2 - prev_weights))

    assert turnover_2 <= turnover_1 + TOL


# verify that higher lambda results in lower risk and lower returns
@pytest.mark.parametrize("n,mu,sigma", [(5, 0.1, 0.2)])
def test_lambda_effect(n, mu, sigma):
    expected_returns, covariance_matrix = _get_return_and_covariance(n, mu, sigma)

    eo = EquityOptimiser(expected_returns, covariance_matrix)

    # Optimize with different lambda values
    w_opt_1, mu_opt_1, sigma_opt_1 = eo.optimise(lambda_=0.1)
    w_opt_2, mu_opt_2, sigma_opt_2 = eo.optimise(lambda_=1.0)

    # Check that higher lambda leads to lower risk
    assert sigma_opt_2 <= sigma_opt_1 + TOL
    # Check that higher lambda may result in lower return
    assert mu_opt_2 <= mu_opt_1 + TOL


def test_risk_given_return():
    # GIVEN: Assets with risk-return tradeoff
    mu = np.array([0.05, 0.15])
    cov = np.array([[0.1, 0.2], [0.2, 0.9]])
    eo = EquityOptimiser(mu, cov)

    # WHEN: Run both optimizers
    _, mu_low_risk, sigma_low_risk = eo.optimise_risk_given_return(min_ret=0.07)
    _, mu_high_risk, sigma_high_risk = eo.optimise_risk_given_return(min_ret=0.1)

    # THEN: Verify tradeoff
    assert np.isclose(mu_low_risk, 0.07, TOL)
    assert np.isclose(mu_high_risk, 0.1, TOL)
    assert mu_high_risk > mu_low_risk
    assert sigma_high_risk > sigma_low_risk
