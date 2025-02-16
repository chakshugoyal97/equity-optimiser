# Equity Optimiser Library
Equity Optimiser based upon Markowitz Modern Portfolio Theory (MPT)

## Features
- Classic Mean-Variance Optimsiation
- Supports weight limits (min, max)
- Dynamically adjust risk parameter and add criterias

## Examples
#TODO

# Contribution
You can try running the tests for yourself. Requires [uv](https://docs.astral.sh/uv/), installable via `brew install uv` on MacOS.

```bash
git clone https://github.com/chakshugoyal97/equity-optimiser.git
cd equity-optimiser
uv venv
uv sync
uv run pytest test/system/
```


## Acknowledgement:
- cvxpy library for optimisation
- uv