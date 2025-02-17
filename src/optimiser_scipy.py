import logging
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

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
        # Validate input
        optimiser_validation_utils.validate_optimiser_inputs(expected_returns, covariance_matrix)

        # Store data
        self._n = expected_returns.shape[0]
        self._mu = expected_returns
        self._sigma = covariance_matrix
        self._constraints = []
        self._bounds = None
        self._lambda = 1.0  # Default risk aversion parameter
        self._txn_cost = None  # Transaction cost vector

    def _objective(self, w):
        """
        Mean-Variance Objective with Transaction Costs:
        Maximize: w^T * mu - lambda * w^T * sigma * w - sum(c1_i * |w_i|) - sum(c2_i * w_i^2)
        scipy.optimize.minimize minimizes functions, so we return the negative of this.
        """
        base_utility = w @ self._mu - self._lambda * w.T @ self._sigma @ w
        
        # Apply transaction costs if set
        if self._txn_cost is not None:
            base_utility -= np.sum(self._txn_cost * np.abs(w))

        return -base_utility

    def _net_exposure_constraint(self, w):
        """ Constraint: Sum of weights should be 1 (fully invested portfolio) """
        return np.sum(w) - 1

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
        Set upper and lower bounds on asset weights.
        """
        self._bounds = [(w_min if w_min is not None else -1, w_max if w_max is not None else 1)] * self._n

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
        self._constraints.append({'type': 'ineq', 'fun': lambda w: self._adv_constraint(w, max_adv)})

    def add_criteria_limit_top_k_allocations(self, k: int, max_limit: float):
        """ Constraint to limit the sum of the k largest long/short allocations """
        self._constraints.append({'type': 'ineq', 'fun': lambda w: self._top_k_allocations_constraint(w, k, max_limit)})

    def modify_utility_txn_costs(self, txn_cost: np.ndarray):
        """
        Modify the objective function to include transaction costs.
        :params:
            txn_cost: A (n,) vector where each element represents a linear cost per asset.
            txn_cost_quadratic: A (n,) vector where each element represents a quadratic cost per asset.
        """
        if txn_cost is not None:
            if len(txn_cost) != self._n:
                raise ValueError("Transaction cost linear vector must match asset count")
            self._txn_cost = txn_cost


    def optimise(self, lambda_: float = 1.0) -> Tuple[np.ndarray, float, float]:
        """
        Run optimizer using scipy.optimize.minimize with SLSQP.
        """
        optimiser_validation_utils.validate_lambda(lambda_)
        self._lambda = lambda_

        # Initial guess: Equal weights
        w0 = np.ones(self._n) / self._n

        # Solve optimization
        result = minimize(
            self._objective, 
            w0, 
            method='SLSQP', 
            bounds=self._bounds, 
            constraints=[{'type': 'eq', 'fun': self._net_exposure_constraint}] + self._constraints
        )

        if not result.success:
            raise ValueError("Optimization failed")

        # Extract results
        optimal_w = result.x
        optimal_return = np.dot(optimal_w, self._mu)
        optimal_risk = np.sqrt(optimal_w.T @ self._sigma @ optimal_w)

        logger.info(f"Optimal Weights: {optimal_w}")
        logger.info(f"Portfolio Expected Return: {optimal_return}, Risk: {optimal_risk}")

        return optimal_w, optimal_return, optimal_risk
