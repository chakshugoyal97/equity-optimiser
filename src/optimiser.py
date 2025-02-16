from typing import Tuple
import numpy as np
import validation
import cvxpy as cp
import logging

logger = logging.getLogger(__name__)

class EquityOptimiser:
    _utility: cp.Objective
    _constraints: cp.Constraint = []
    _w: cp.Variable

    def __init__(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray):
        """
        :params:
            expected_returns: A vector of expected asset returns (mean vector).
            Assume the returns are sampled from a normal distribution with typical stock mean and volatilities.
        """
        self._n = len(expected_returns)
        self._mu = expected_returns
        self._sigma = covariance_matrix
        validation.validate_optimiser_inputs(expected_returns, covariance_matrix)
        self.add_criteria_baseline()
        self.add_utility_baseline()
        

    def add_criteria_baseline(self):
        """
        Constraints:
            sum(_w) = 1
        """
        self._w = cp.Variable((self._n, 1))
        self._constraints += [cp.sum(self._w) == 1]


    def add_criteria_weights():
        """
        Weight limits on individual assets (e.g., no more than 10% in any single asset).
        Bounds on asset weights.
        w <= w_max
        """
        pass

    def add_criteria_return_target():
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
        self._lambda = cp.Parameter
        self._return = self._w.T @ self._mu
        self._risk = self._lambda * cp.quad_form(self._w, self._sigma)
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

    def optimise(self) -> Tuple[np.ndarray, float, float]:
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
        self._lambda = 1
        problem = cp.Problem(cp.Maximize(self._utility), self._constraints)
        return problem.solve()



