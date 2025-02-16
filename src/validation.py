import logging

logger = logging.getLogger(__name__)

class InputError(Exception):
    pass

def validate_optimiser_inputs(var, covar):
    N = var.size
    if(N < 1): 
        raise InputError(f"Incorrect shape of var, got {N}")
    if(covar.shape != (N, N)):
        raise InputError(f"Incorrect shape of covar, expected {(N, N)}, got {covar.shape}")