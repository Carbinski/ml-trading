# ml-trading

Personal implementations of Machine Learning for Trading ideas (Georgia Tech / Udacity topics), not a 1:1 curriculum clone.

This first slice is **data in**: cache the temporary ML4T homework CSVs, load a date-aligned panel, clean bars, and split train/test **in time**. Parametric regression, kNN, and the SPY backtest are later work. There is no brokerage or order code here.

## Setup

Python 3.11+. Pandas is the only runtime dependency.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Homework data (stale, not a live vendor)

Phase 1 temporarily uses the old course CSVs mirrored at
[JeffreyJackovich/machine-learning-for-trading](https://github.com/JeffreyJackovich/machine-learning-for-trading/tree/master/Part%201-manipulating_financial_data_in_python/data)
(pinned commit `08763f80649604158040dd1559925b3b90c97d32`).

They include **FAKE1** / **FAKE2**, synthetic `ML4T-*` / `SINE_*` files, and dead tickers. Prices stop in **September 2012**. This is homework residue, not a market data subscription. The full dump is ~170MB and the upstream repo has no license — cache it locally; do not vendor the dump in git. Tests ship a small SPY + FAKE1 + FAKE2 excerpt under `tests/fixtures/ml4t/`.

```bash
python -m ml_trading.fetch                       # starter symbols into data/ml4t/
python -m ml_trading.fetch --symbols SPY,FAKE1,FAKE2
python -m ml_trading.fetch --all                 # full data folder via GitHub zipball
```

Details (file layout, missing sessions, fetch flags): [docs/data.md](docs/data.md).

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

## Yahoo / yfinance (personal research)

**Phase-1 vendor rec for anything beyond this homework dump: Yahoo via yfinance, for personal research.** Carson fetches that himself, caches it locally, and does not redistribute it. This repository does **not** run yfinance, hit Yahoo Finance, or ship a Yahoo download script — automated collection would violate Yahoo ToS for our bots. Do not add Tiingo. See [docs/data.md](docs/data.md).
