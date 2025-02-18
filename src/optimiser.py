import logging
import math
from typing import Optional, Tuple

import cvxpy as cp
import numpy as np

import optimiser_validation_utils

logger = logging.getLogger(__name__)


class EquityOptimiser:
    def __init__(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        prev_weights: Optional[np.ndarray] = None,
    ):
        """
        :params:
            expected_returns:
                A vector (shape = (n,1)) of expected asset returns (mean vector).
                Assume the returns are sampled from a normal distribution with typical stock mean and volatilities.
            covariance_matrix:
                An (n,n) np array of covariance between asset returns. Should be positive semi-definite.
            prev_weights:
                Previous or existing weights. Required for certain constraints.
        """
        # validate input
        optimiser_validation_utils.validate_optimiser_inputs(
            expected_returns, covariance_matrix, prev_weights
        )

        # fill data
        self._n = expected_returns.shape[0]
        self._mu = expected_returns.reshape(self._n, 1)
        self._sigma = covariance_matrix.reshape(self._n, self._n)
        self._constraints: cp.Constraint = []

        self._w_prev = prev_weights if prev_weights is not None else np.zeros(self._n)
        self._w = cp.Variable(self._n)
        self._utility = cp.Objective

        # setup base objective function and criteria
        self.add_criteria_baseline()
        self.add_objective_baseline()

    # constraints
    def add_criteria_baseline(self):
        """
        Constraints:
            1^T * _w = 1
        """
        self._constraints += [cp.sum(self._w) == 1]

    def add_criteria_weights(
        self, w_min: Optional[float] = None, w_max: Optional[float] = None
    ):
        """
        Weight limits on individual assets (e.g., no more than 10% in any single asset).
        Bounds on asset weights.
        w_min <= w <= w_max
        """
        if w_min is not None:
            self._constraints += [w_min <= cp.min(self._w)]
        if w_max is not None:
            self._constraints += [cp.max(self._w) <= w_max]

    def add_criteria_return_target(self, mu_min: float, mu_max: Optional[float] = None):
        """
        A minimum return target for the portfolio.
        A target portfolio return.
        w^T * mu >= mu_min              (min asset return -> mu_max)
        """
        self._constraints += [mu_min <= self._return]
        if mu_max:
            self._constraints += [self._return <= mu_max]

    def add_criteria_risk_level(
        self, sigma_max: float, sigma_min: Optional[float] = None
    ):
        """
        A maximum risk level (maximum portfolio variance).
        As the constraint is not linear, this will trigger a convex solver.
        w^T * sigma * w <= sigma_max^2  (max variance -> sigma_max^2)
        :sigma_max:
            maximum standard deviation or sqrt(variance)
        """
        self._constraints += [self._risk <= sigma_max * sigma_max]
        if sigma_min:
            self._constraints += [sigma_min * sigma_min <= self._risk]

    def add_criteria_factor_exposure(
        self,
        factor_matrix: np.ndarray,
        min_exposure: Optional[np.ndarray],
        max_exposure: Optional[np.ndarray],
    ):
        """
        Control for industry/factor exposure.
        :params:
            factor_matrix: Factor exposures for each of the assets with shape = (f,n) where f is the # of factors
            min_exposure: minimum exposure vector to each factor, shape (f,)
            max_exposure: maximum exposure vector to each factor, shape (f,)
        """
        if min_exposure is not None:
            self._constraints += [min_exposure <= factor_matrix @ self._w]
        if max_exposure is not None:
            self._constraints += [factor_matrix @ self._w <= max_exposure]

    def add_criteria_max_adv_equity(self, limit: float, volume: float, adv: np.ndarray):
        """
        If a stock i, has an adv-i in the market overall (for eg. GOOG ADV is 2B USD),
        then we cannot trade more than max_adv * adv-i of that stock.
        |w_delta| * volume <= limit * max_adv
        :params:
            limit: adv multiple limit that is allowed to be traded, for ex 0.05 if setting to 5% of ADV
            w_prev: previous stock weights, should sum to 1 with shape (n,)
            volume: net asset value of the portfolio
            max_adv: market data - vector of adv traded per stock, should have shape (n,)
        """
        assert limit >= 0 and limit <= 1
        assert adv.shape == (self._n,)

        volume_traded = cp.abs(self._w - self._w_prev) * volume
        volume_permissible = limit * adv
        self._constraints += [volume_traded <= volume_permissible]

    def add_criteria_limit_top_k_allocations(self, k: int, max_limit: float):
        """
        Sum of the k largest long/short allocations should not exceed a given target number.

        Let t be the threshold cutoff for top-k allocations, then this can be represented by:
        SUM[ max(w_i - t, 0) ] + t*k <= max_limit, t >= 0

        Or use cp.sum_largest()
        """
        self._constraints += [cp.sum_largest(cp.abs(self._w), k) <= max_limit]

    # objectives
    def add_objective_baseline(self):
        """
        Objective Function:
            + [w^T * mu]                    ; maximise return
            - [lambda * w^T * sigma * w]    ; minimise risk, here, lambda is a +ve, risk parameter
            - [t * sum(|w_delta|)]          ; penalise turnover, here, t is a +ve turnover penalty parameter
        """
        # nonneg to ensure convexity
        self._lambda = cp.Parameter(nonneg=True)

        self._return = self._w.T @ self._mu
        self._risk = self._w.T @ self._sigma @ self._w

        # penalise high turnover: - t * sum(|w_delta|)
        self._t = cp.Parameter(nonneg=True)
        self.turnover_penalty = self._t * cp.sum(cp.abs(self._w - self._w_prev))

        # put it together
        self._utility = (
            self._return - self._lambda * self._risk - self._t * self.turnover_penalty
        )

    def set_txn_costs(self, c: np.ndarray):
        """
        Accomodate transaction costs (slippage, bid-ask spread, etc.).
        - c^T @ abs(w_delta) where, c is an (n,) vector representing txn costs effect
        """
        self._utility -= c.T @ cp.abs(self._w - self._w_prev)

    # solve
    def optimise(
        self, lambda_: float = 1.0, t_: float = 0
    ) -> Tuple[np.ndarray, float, float]:
        """
        Run optimiser and return optimal weights.
        :params:
            lambda_:   +ve parameter to adjust risk term
            t_:        +ve parameter to adjust portfolio-turnover term
        :returns:
            optimal_weights: weights vector shape=(n,) for each stock
            E(return): optimal portfolio expected return
            E(risk/variance): optimal portfolio standard_deviation (sqrt(variance))
        """
        optimiser_validation_utils.validate_lambda(lambda_)
        self._lambda.value = lambda_
        self._t.value = t_

        problem = cp.Problem(cp.Maximize(self._utility), self._constraints)
        # let cvxpy choose automatically or force interior-points by solver=cp.CLARABEL etc
        u = problem.solve()

        if self._w.value is None:
            raise ValueError("Optimization failed")

        logger.debug(f"Solver used: {problem.solver_stats.solver_name}, utililty: {u}")
        logger.info(f"optimal weights: {self._w.value}")
        logger.info(
            f"mu: {self._return.value[0]}, sigma: {math.sqrt(self._risk.value)}"
        )

        return self._w.value, self._return.value[0], math.sqrt(self._risk.value)
