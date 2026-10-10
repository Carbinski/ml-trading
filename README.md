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
