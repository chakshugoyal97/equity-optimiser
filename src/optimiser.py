import logging
import math
from typing import Optional, Tuple

import cvxpy as cp
import numpy as np

import optimiser_validation_utils

logger = logging.getLogger(__name__)


class EquityOptimiser:
    def __init__(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray):
        """
        :params:
            expected_returns:
                A vector (shape = (n,1)) of expected asset returns (mean vector).
                Assume the returns are sampled from a normal distribution with typical stock mean and volatilities.
            covariance_matrix:
                An (n,n) np array of covariance between asset returns. Should be positive semi-definite.
        """
        # validate input
        optimiser_validation_utils.validate_optimiser_inputs(
            expected_returns, covariance_matrix
        )

        # fill data
        self._n = expected_returns.shape[0]
        self._mu = expected_returns.reshape(self._n, 1)
        self._sigma = covariance_matrix.reshape(self._n, self._n)
        self._constraints: cp.Constraint = []
        self._w = cp.Variable(self._n)
        self._utility = cp.Objective

        # setup base objective function and criteria
        self.add_criteria_baseline()
        self.add_utility_baseline()

    # constraints
    def add_criteria_baseline(self):
        """
        Constraints:
            1^T * _w = 1
        """
        one_vec = np.full(self._n, 1)
        self._constraints += [one_vec @ self._w.T == 1]

    def add_criteria_weights(
        self, w_min: Optional[float] = None, w_max: Optional[float] = None
    ):
        """
        Weight limits on individual assets (e.g., no more than 10% in any single asset).
        Bounds on asset weights.
        w_min <= w <= w_max
        """
        if w_min is not None:
            self._constraints += [w_min <= self._w]
        if w_max is not None:
            self._constraints += [self._w <= w_max]

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

    def add_criteria_factor_exposure():
        """
        Control for industry/factor exposure.
        :params:
            factor_matrix: Factor exposures for each of the assets. Assume a factor loading matrix of 5-10 factors with randomly sampled values.
            exposure_constraint: Maximum/Minimum exposure to a factor
        """
        pass

    def add_criteria_max_adv_equity(
        self, w_prev: np.ndarray, ADV: np.ndarray, max_adv: float
    ):
        """
        Eg - No more than 5% of the ADV traded for a single stock
        """
        optimiser_validation_utils.validate_adv(w_prev, ADV, self._n)
        pass

    def add_criteria_limit_top_k_allocations(self, k: int, max_limit: float):
        """
        Sum of the k largest long/short allocations should not exceed a given target number.

        Let t be the threshold cutoff for top-k allocations, then this can be represented by:
        SUM[ max(w_i - t, 0) ] + t*k <= max_limit, t >= 0

        Or use cp.sum_largest.
        """
        self._constraints += [cp.sum_largest(cp.abs(self._w), k) <= max_limit]

    # objectives
    def add_utility_baseline(self):
        """
        Objective Function:
            w^T * mu - lambda * w^T * sigma * w     ; where lambda is a +ve, risk parameter
        """
        self._lambda = cp.Parameter(
            nonneg=True
        )  # (to ensure concavity for maximisation problem, and calm down cvxpy)
        self._return = self._w.T @ self._mu
        self._risk = self._w.T @ self._sigma @ self._w
        self._utility = self._return - self._lambda * self._risk

    def modify_utility_txn_costs():
        """
        accomodate transaction costs (slippage, bid-ask spread, etc.).
        Fixed transaction costs per asset?
        """
        pass

    def modify_utility_reduce_turnover(self):
        """
        Reduction in turnover of the portfolio.
        """
        pass

    # solve
    def optimise(self, lambda_: float = 1.0) -> Tuple[np.ndarray, float, float]:
        """
        Run optimiser and return optimal weights.
        Optimization Approach:
            Quadratic Programming for Mean-Variance Optimization
            Interior Point Methods or Sequential Quadratic Programming (SQP)
        :params:
            expected_returns: an ndarray(n) of E(stock-i returns)
            covariance: an (nxn) numpy matrix of Cov(stock-i,stock-j)
        :returns:
            optimal_weights: weights vector shape=(n,) for each stock
            E(return): optimal portfolio expected return
            E(risk/variance): optimal portfolio standard_deviation (sqrt(variance))
        """
        optimiser_validation_utils.validate_lambda(lambda_)
        self._lambda.value = lambda_

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
