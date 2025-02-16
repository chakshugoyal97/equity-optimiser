from typing import Tuple
import numpy as np
import validation
import cvxpy as cp
import logging

logger = logging.getLogger(__name__)


class EquityOptimiser:
    _utility: cp.Objective

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
        validation.validate_optimiser_inputs(expected_returns, covariance_matrix)

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

    def add_criteria_baseline(self):
        """
        Constraints:
            sum(_w) = 1
        """
        self._constraints += [cp.sum(self._w) == 1]

    def add_criteria_weights(self, w_min: float = None, w_max: float = None):
        """
        Weight limits on individual assets (e.g., no more than 10% in any single asset).
        Bounds on asset weights.
        w_min <= w <= w_max
        """
        if w_min is not None:
            self._constraints += [w_min <= self._w]
        if w_max is not None:
            self._constraints += [self._w <= w_max]

    def add_criteria_return_target(mu_min: float):
        """
        A minimum return target for the portfolio.
        A target portfolio return.
        w^T * mu >= mu_min              (min asset return -> mu_max)
        """
        pass

    def add_criteria_risk_level():
        """
        A maximum risk level (maximum portfolio variance).
        A maximum allowable portfolio variance.
        w^T * sigma * w <= sigma_max^2  (max variance -> sigma_max^2)
        """
        pass

    def add_criteria_factor_exposure():
        """
        Control for industry/factor exposure.
        :params:
            factor_matrix: Factor exposures for each of the assets. Assume a factor loading matrix of 5-10 factors with randomly sampled values.
            exposure_constraint: Maximum/Minimum exposure to a factor
        """
        pass

    def add_criteria_max_adv_equity():
        """
        Eg - No more than 5% of the ADV traded for a single stock
        """
        pass

    def add_criteria_limit_top_k_allocations():
        """
        Sum of the k largest long/short allocations should not exceed a given target number
        """
        pass

    def add_utility_baseline(self):
        """
        Objective Function:
            w^T * mu - lambda * w^T * sigma * w     (lambda is risk parameter)
        """
        self._lambda = cp.Parameter(
            nonneg=True
        )  # (ensures concavity for maximisation problem)
        self._return = self._w.T @ self._mu
        self._risk = self._w.T @ self._sigma @ self._w
        self._risk = self._lambda * self._risk
        self._utility = self._return - self._risk

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
            optimal_weights:
            E(return):
            E(risk/variance):
        """
        validation.validate_lambda(lambda_)
        self._lambda = lambda_
        problem = cp.Problem(cp.Maximize(self._utility), self._constraints)
        problem.solve()
        if self._w.value is None:
            raise ValueError("Optimization failed")
        return self._w.value
