import math
import numpy as np

def expectation(w: np.ndarray, mu: np.ndarray):
    return w.T @ mu

def variance(w: np.ndarray, sigma: np.ndarray):
    return w.T @ sigma @ w

def std_dev(w: np.ndarray, sigma: np.ndarray):
    return np.sqrt(w.T @ sigma @ w)
