import cmath
import logging

import numpy as np

from constants import TOL

logger = logging.getLogger(__name__)


class InputError(Exception):
    pass


class UnboundedError(Exception):
    pass


def _is_PSD(A: np.ndarray, tol: float = TOL):
    """
    Check if a np array is positive semi-definite.
    A matrix M is PSD if it is symmetrix and x^T.M.x >= 0 for all x in R^n
    https://stackoverflow.com/questions/5563743/check-for-positive-definiteness-or-positive-semidefiniteness
    Checked if all eigenvalues are +ve or within -tolerance for machine errors
    """
    E = np.linalg.eigvalsh(A)
    return np.all(E > -tol)


def validate_optimiser_inputs(var: np.ndarray, covar: np.ndarray, w_prev: np.ndarray):
    N = var.shape[0]
    if N < 1 or var.shape != (N,):
        raise InputError(f"Incorrect shape of var, got {var.shape}")

    if covar.shape != (N, N):
        raise InputError(
            f"Incorrect shape of covar, expected {(N, N)}, got {covar.shape}"
        )

    # Check covariance matrix is PSD.
    # This is a criteria for convex optimisation
    if not _is_PSD(covar):
        raise InputError("Covariance matrix must be PSD.")

    if w_prev is not None:
        if not np.isclose(np.sum(w_prev), 1.0):
            raise InputError("previous weights do not sum to 1")
        if w_prev.shape != (N,):
            raise InputError(
                f"incorrect previous weights shape, expected, {(N, 1)}, got {w_prev.shape}"
            )


def validate_lambda(lambda_: float):
    if lambda_ < 0:
        raise InputError("lambda cannot be negative")


def validate_adv(w_prev: np.ndarray, adv: np.ndarray, n: int, limit):
    if not np.isclose(np.sum(w_prev), 1.0, TOL):
        raise InputError("prev weights do not sum to 1")
    if not w_prev.shape == (n,):
        raise InputError(f"Incorrect w_prev shape, expected {(n,)}, got {w_prev.shape}")
    if not adv.shape == (n,):
        raise InputError(f"Incorrect ADV shape, expected {(n,)}, got {adv.shape}")
    if not limit >= 0 and limit <= 1:
        raise InputError("limit not between 0 and 1")


def validate_txn_inputs(txn_cost: np.ndarray, n: int):
    if len(txn_cost) != n:
        raise ValueError("Transaction cost vector must match asset count")
