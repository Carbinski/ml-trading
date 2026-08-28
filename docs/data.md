# Data: Jackovich ML4T CSVs and later Yahoo pulls

## What phase 1 uses (temporary)

The CSVs in
[JeffreyJackovich/machine-learning-for-trading `.../data`](https://github.com/JeffreyJackovich/machine-learning-for-trading/tree/master/Part%201-manipulating_financial_data_in_python/data)
are an old Udacity / Georgia Tech **Machine Learning for Trading** homework dump.

They are **not** a live data vendor. Treat them as stale course files:

- Pinned commit: `08763f80649604158040dd1559925b3b90c97d32` (repo has no license).
- About **1005** per-symbol CSVs plus `Lists/*.txt` (~**170MB**). A duplicate tree lives at `ml4t/data/`; this repo fetches only `Part 1-manipulating_financial_data_in_python/data`.
- Layout (confirmed, not assumed): `{SYMBOL}.csv` with columns `Date, Open, High, Low, Close, Volume, Adj Close`. Dates are timezone-naive `YYYY-MM-DD`, newest row first, weekdays only. SPY in the dump runs **2000-02-01 → 2012-09-12**.
- Includes **FAKE1** / **FAKE2**, course synthetics (`ML4T-000` … `ML4T-399`, `SINE_*`), and **dead tickers** (examples: `WWY`, `ABI`) with long stretches of volume `000`.
- **FAKE1** matches the SPY calendar in its 2009–2012 window. **FAKE2** does not: it is missing SPY sessions **2007-10-01..2007-12-31**, **2008-11-13..2008-11-25**, and **2010-06-15..2010-12-31**.
- Adjustments are frozen as of ~2012 (e.g. AAPL still prints pre-split hundreds).

Do not commit the full dump (size + no license). Cache it locally.

### Fetch / cache

```bash
python -m ml_trading.fetch                  # starter: SPY, FAKE1, FAKE2, AAPL, IBM, GOOG
python -m ml_trading.fetch --symbols SPY,FAKE1,FAKE2
python -m ml_trading.fetch --all            # zipball of the data folder (~170MB)
python -m ml_trading.fetch --all --skip-synthetic
python -m ml_trading.fetch --list           # remote symbol names
```

Files land in `data/ml4t/` (gitignored). Tests use `tests/fixtures/ml4t/` (SPY + FAKE1 + FAKE2 excerpt) and do not need the network.

Manual equivalent: download
`https://github.com/JeffreyJackovich/machine-learning-for-trading/archive/08763f80649604158040dd1559925b3b90c97d32.zip`
and copy `Part 1-manipulating_financial_data_in_python/data/` into `data/ml4t/`.

## Missing-bar policy

1. **Trading calendar = SPY dates** in the chosen window (not a weekday range). Weekends and NYSE holidays are already absent from SPY.
2. Other symbols are **reindexed** onto that calendar. Omitted sessions become `NaN`.
3. Default fill is **none**. The only other option is **`ffill`** (last known **past** bar). Leading NaNs stay NaN.
4. **Never** backward-fill or interpolate. That would copy a later price onto an earlier date.

Cleaning drops/flags unusable rows (bad dates, duplicate dates, high < low, non-positive prices, unparseable/missing required fields). Volume `000` becomes `0`. Dates are timezone-naive. Cleaning does **not** invent missing sessions.

## Time split

`time_split(frame, cutoff="2011-01-01")` puts every row with `date < cutoff` in train and `date >= cutoff` in test. No shuffle, no k-fold across time. `assert_no_time_leak` checks `train.index.max() < test.index.min()`.

## Yahoo / yfinance (personal research only)

**Phase-1 vendor recommendation for live or post-2012 research: Yahoo Finance via `yfinance`, fetched by Carson, cached locally.**

- Carson runs `yfinance` himself on his machine. This repo must **not** pip-install yfinance for data pulls, call Yahoo / `query1.finance.yahoo.com`, or ship a fetch script that executes a Yahoo download (automated collection would violate Yahoo ToS for our bots).
- Cache locally. Do **not** redistribute Yahoo extracts.
- Do not buy or integrate Tiingo. Do not scrape logins. No brokerage or order code in this slice.
