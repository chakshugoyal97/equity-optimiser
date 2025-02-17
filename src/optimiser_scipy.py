import functools
import logging
from typing import Optional, Tuple

import numpy as np
import scipy.optimize as sco

import lin_alg
import optimiser_validation_utils

logger = logging.getLogger(__name__)


class EquityOptimiser:
    def __init__(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray):
        """
        :params:
            expected_returns:
                A vector (shape = (n,)) of expected asset returns (mean vector).
            covariance_matrix:
                An (n,n) np array of covariance between asset returns. Should be positive semi-definite.
        """
        # validate input
        optimiser_validation_utils.validate_optimiser_inputs(expected_returns, covariance_matrix)

        # fill data
        self._n = expected_returns.shape[0]
        self._mu = expected_returns.reshape(self._n, 1)
        self._sigma = covariance_matrix.reshape(self._n, self._n)
        self._constraints = []
        self._w = np.ones(self._n) / self._n # equal weights to start
        self._bounds = None
        self._lambda = 1.0  # risk parameter
        self._txn_cost = np.full(self._n, 0).reshape(self._n, 1)  # txn cost

        # setup base objective function and criteria
        self.add_criteria_baseline()

    # constraints
    def add_criteria_baseline(self):
        """
            Constraints:
                1^T * _w = 1
        """
        def _net_exposure_constraint(n, w: np.ndarray):
            return np.ones(n).T @ w - 1

        # net_exposure_func = functools.partial(_net_exposure_constraint, self._n)
        self._constraints += [{'type': 'eq', 'fun': lambda w: _net_exposure_constraint(self._n, w)}]

    def _adv_constraint(self, w, max_adv):
        """ ADV Constraint: |w_i| ≤ max_adv * sum(|w|) for all i """
        total_volume = np.sum(np.abs(w))
        return max_adv * total_volume - np.max(np.abs(w))

    def _risk_constraint(self, w, sigma_max):
        """ Risk Constraint: Portfolio variance must be below a threshold """
        return sigma_max**2 - (w.T @ self._sigma @ w)

    def _top_k_allocations_constraint(self, w, k, max_limit):
        """ Sum of the k largest long/short allocations should not exceed max_limit """
        largest_allocations = np.partition(np.abs(w), -k)[-k:]  # Get k largest absolute weights
        return max_limit - np.sum(largest_allocations)

    def add_criteria_weights(self, w_min: Optional[float] = None, w_max: Optional[float] = None):
        """
        Set weight limits on individual assets (e.g., no more than 10% in any single asset).
        w_min <= w <= w_max
        """
        self._bounds = [(w_min, w_max)] * self._n

    def add_criteria_return_target(self, mu_min: float, mu_max: Optional[float] = None):
        """
        Ensure expected return meets constraints: mu_min ≤ w^T * mu ≤ mu_max
        """
        self._constraints.append({'type': 'ineq', 'fun': lambda w: np.dot(w, self._mu) - mu_min})
        if mu_max is not None:
            self._constraints.append({'type': 'ineq', 'fun': lambda w: mu_max - np.dot(w, self._mu)})

    def add_criteria_risk_level(self, sigma_max: float):
        """ Constraint to limit portfolio variance """
        self._constraints.append({'type': 'ineq', 'fun': lambda w: self._risk_constraint(w, sigma_max)})

    def add_criteria_max_adv_equity(self, max_adv: float):
        """ ADV Constraint Implementation """
        """
        For ex, no more than 5% of the ADV traded for a single stock. There could be multiple interpretations to this.
        Interpretation 1:
            If a stock i, has an adv-i in the market overall (for eg. GOOG ADV ~ 1.5B USD), then we cannot trade more than max_adv * adv-i of that stock.
                -> need to know total volume of portfolio, previous day/current prices, etc to correctly accomodate ...
        Interpration 2:
            As the portfolio is long/short, total volume that we trade is [|w_+| + |w_-|] * V, where V is the initial amount we began with.
            ADV for a stock is the amount of stock traded relative to the overall volume of the portfolio. Which means,
            |w_i|*V / ([|w_+| + |w_-|] * V) <= max_adv
            <=> |w_i| / ([|w_+| + |w_-|]) <= max_adv

        As the interpretation 2 is simpler and more reasonable based on the input data, in the assignment, this is the one we go with.
        Note, this will not be a convex constraint.
        """
        self._constraints.append({'type': 'ineq', 'fun': lambda w: self._adv_constraint(w, max_adv)})

    def add_criteria_limit_top_k_allocations(self, k: int, max_limit: float):
        """ Constraint to limit the sum of the k largest long/short allocations """
        self._constraints.append({'type': 'ineq', 'fun': lambda w: self._top_k_allocations_constraint(w, k, max_limit)})

    def add_criteria_factor_exposure():
        """
        Control for industry/factor exposure.
        :params:
            factor_matrix: Factor exposures for each of the assets. Assume a factor loading matrix of 5-10 factors with randomly sampled values.
            exposure_constraint: Maximum/Minimum exposure to a factor
        """
        pass

    # objective function
    def _objective(self, w):
        """
        Mean-Variance Objective with Transaction Costs:

        Maximize: w^T * mu - lambda * (w^T * sigma * w) - sum(c1_i * |w_i|))
            ; where lambda is a +ve, risk parameter 
        
        We return the negative of this as sco.minimize, needs a minimization problem.
        """
        expectation = lin_alg.expectation(self._w, self._mu)
        variance = lin_alg.variance(self._w, self._sigma)
        base_utility =  expectation - self._lambda * variance \
            - np.sum(self._txn_cost * np.abs(w))

        return -base_utility
    
    def modify_objective_txn_costs(self, txn_cost: np.ndarray):
        """
        Modify the objective function to include transaction costs.
        :params:
            txn_cost: An (n,) vector where each element represents some linear kind-of cost per asset.
        """
        optimiser_validation_utils.validate_txn_inputs(txn_cost, self._n)
        self._txn_cost = txn_cost.reshape(self._n, 1)

    def modify_objective_reduce_turnover():
        """
        Reduction in turnover of the portfolio.
        """
        pass

    # solve
    def optimise(self, lambda_: float = 1.0) -> Tuple[np.ndarray, float, float]:
        """
        Run optimizer using scipy.optimize.minimize with SLSQP.
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
        self._lambda = lambda_

        # Solve optimization
        result: sco.OptimizeResult = sco.minimize(
            self._objective, 
            self._w, 
            method='SLSQP',
            bounds=self._bounds, 
            constraints=self._constraints
        )

        if not result.success:
            logger.error(f"optimisation failed message: {result.message}")
            raise ValueError("Optimization failed!")

        # Extract results
        self._w = result.x
        optimal_return = lin_alg.expectation(self._w, self._mu)
        optimal_risk = lin_alg.std_dev(self._w, self._sigma)

        logger.info(f"optimal weights: {self._w}")
        logger.info(f"mu: {optimal_return}, sigma: {optimal_risk}")

        return self._w, optimal_return, optimal_risk
