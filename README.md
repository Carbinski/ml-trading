# ml-trading

Learning repo: price loaders, chronological splits, and logged ML experiments.

## Setup

Python 3.11+. Pandas is the only runtime dependency.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Data (stale)

Uses the old course CSVs mirrored at
[JeffreyJackovich/machine-learning-for-trading](https://github.com/JeffreyJackovich/machine-learning-for-trading/tree/master/Part%201-manipulating_financial_data_in_python/data)
(pinned commit `08763f80649604158040dd1559925b3b90c97d32`).

```bash
python -m ml_trading.fetch                       # starter symbols into data/ml4t/
python -m ml_trading.fetch --symbols SPY,FAKE1,FAKE2
python -m ml_trading.fetch --all                 # full data folder via GitHub zipball
```

Details (file layout, missing sessions, fetch flags): [docs/data.md](docs/data.md).

## Experiment data

Logged runs use a local Yahoo cache in `data/clean-yfinance` (gitignored). Download scripts are in `src/data/`. Do not commit or redistribute those CSVs.

Experiment log: [docs/ml/index.md](docs/ml/index.md).

## What to work on

Do these in order. Logged signed next-day return models already have holdout R² at or below zero. A deeper model on that same target repeats that result.

1. **SGD logistic, SPY direction.** NumPy log loss. Gradient is `(sigmoid(w·x) - y) x`, with a fixed or decaying step size and L2 as weight decay. Label is the sign of the next-day `Adj Close` return. Features are the stationary columns in `src/ml/regression/ai_scripts/parametric_experiment_common.py`. Cutoff `2024-09-01`. Score accuracy against always-up, and long-or-flat growth of $1 against holding SPY. A scikit-learn `LogisticRegression` fit is a gradient check after the loop has a number.
2. **Same SGD loop, cross-sectional median.** One row per symbol-day. Label is 1 when that name's next-day return beats the same day's median across `STARTING_STOCKS`. Score the equal-weight basket of names predicted to beat the median against the equal-weight basket of all 20.
3. **Linear Q-learning.** Replace the price/SMA bin table with those same stationary features. One weight vector per action. The update is the tabular semi-gradient with the feature vector in place of the one-hot state. Train on dates before the cutoff, act greedy on the test window, and score growth of $1 against holding the stock.
4. **One-hidden-layer net on absolute return.** SGD. Target is the absolute next-day `Adj Close` return. Same split and features as volatility ridge. Beat that run's test micro R² and correlation on a matched window.

The same list is the todo on [docs/ml/index.md](docs/ml/index.md).

## Loader, cleaning, time split

```python
from ml_trading import load_prices, load_panel, time_split

# SPY trading calendar, missing FAKE2 days stay NaN (no lookahead).
prices = load_prices(
    ["SPY", "FAKE1", "FAKE2"],
    data_dir="tests/fixtures/ml4t",   # or data/ml4t after fetching
    fill="none",                      # or "ffill" for past-only fill
)

train, test = time_split(prices, cutoff="2011-01-01")
assert train.index.max() < test.index.min()
```

- **Calendar:** dates SPY actually traded. Weekends/holidays are not invented.
- **Missing bars:** reindex onto that calendar; default leave `NaN`. Optional `ffill` copies the last **past** value only — never bfill / never fill from the future.
- **Cleaning:** snake_case columns, numeric dtypes, timezone-naive dates, sort ascending, drop/flag bad rows (invalid dates, duplicates, high < low, non-positive prices).
- **Split:** earlier dates → train, later dates → test. Configurable cutoff. No shuffle, no k-fold across time.

`load_panel(...)` is the same alignment with MultiIndex columns `(field, symbol)`.
