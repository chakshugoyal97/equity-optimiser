# Equity Optimiser Library
Equity optimiser based upon Markowitz Modern Portfolio Theory (MPT) for long/short portfolios.

## Features
- Classic Mean-Variance Optimsiation
- Supports weight limits (min, max)
- Set maximum risk level, and/or minimum return expectation
- Dynamically add more complex constraints like limit volume/ADV ratio, top-k concentration, factor-exposure constraints, etc ...
- Adjust risk parameter, reduce turnover, txn costs etc

## Usage

### Initialising the Optimiser
```python
import numpy as np
from optimiser import EquityOptimiser

# Define expected returns and covariance matrix
expected_returns = np.array([0.1, 0.2, 0.15, 0.12, 0.18])
covariance_matrix = np.array([
    [0.05, 0.01, 0.02, 0.01, 0.02],
    [0.01, 0.06, 0.02, 0.01, 0.02],
    [0.02, 0.02, 0.07, 0.02, 0.03],
    [0.01, 0.01, 0.02, 0.04, 0.01],
    [0.02, 0.02, 0.03, 0.01, 0.08]
])

# Initialize optimizer
eo = EquityOptimiser(expected_returns, covariance_matrix)
```

### Adding Constraints
```python
# Add weight limits
eo.set_weights_bound(w_min=0.0, w_max=0.3)

# Add minimum return constraint
eo.set_min_return(mu_min=0.12)

# Add maximum risk constraint
eo.set_max_risk(sigma_max=0.15)
```

### Optimising The Portfolio
```python
# Optimize the portfolio
optimal_weights, expected_return, expected_risk = eo.optimise(lambda_=0.5, t_=0.1)

print("Optimal Weights:", optimal_weights)
print("Expected Return:", expected_return)
print("Expected Risk:", expected_risk)
```

# Contribution
Here, we use [uv](https://docs.astral.sh/uv/), installable via `brew install uv` on MacOS to setup environment, and run tests.

```bash
git clone https://github.com/chakshugoyal97/equity-optimiser.git
cd equity-optimiser
uv venv
uv sync
uv run pytest tests
```

Linting stuff
```bash
uv run ruff format
uv run ruff check --select I --fix .
uv run ruff check --select PL --fix .
```

## Pre-Requisites:
- cvxpy
- numpy
- pytest
- uv