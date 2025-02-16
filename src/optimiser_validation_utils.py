import cmath
import logging

import numpy as np

from constants import TOL

logger = logging.getLogger(__name__)


class InputError(Exception):
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


def validate_optimiser_inputs(var: np.ndarray, covar: np.ndarray):
    N, a = var.shape
    if N < 1 or a != 1:
        raise InputError(f"Incorrect shape of var, got {var.shape}")

    if covar.shape != (N, N):
        raise InputError(
            f"Incorrect shape of covar, expected {(N, N)}, got {covar.shape}"
        )

    # Check covariance matrix is PSD.
    # This is a criteria for convex optimisation
    if not _is_PSD(covar):
        raise ValueError("Covariance matrix must be PSD.")


def validate_lambda(lambda_: float):
    if lambda_ < 0:
        raise InputError("lambda cannot be negative")


def validate_adv(w_prev: np.ndarray, adv: np.ndarray, n: int):
    if not cmath.isclose(np.sum(w_prev), 1.0, TOL):
        raise InputError("prev weights do not sum to 1")
    if not w_prev.shape == (n,):
        raise InputError(f"Incorrect w_prev shape, expected {(n,)}, got {w_prev.shape}")
    if not adv.shape == (n,):
        raise InputError(f"Incorrect ADV shape, expected {(n,)}, got {adv.shape}")
