# Decision tree regression

**Script:** `src/ml/regression/decision-tree.py`
**Family:** trees
**Current:** v2 (`current`)

`DecisionTreeRegressor` on lag-1 bar-shape plus 1-day return and overnight. Next-day `Adj Close` return. SPY only. Depth chosen by train-only time-series CV.

Shared: cutoff `2024-09-01`. Loads 20 names, trains SPY. Same-day derived columns are features for that return. Overnight is split-adjusted open vs prior `Adj Close`. v1 used 10y SPY (head 2016-09-01). v2 uses 25y SPY (head 2001-09-04).

## Versions

### v2 - same recipe, 25y SPY

- **Date:** 2026-09-01
- **Status:** current
- **Script:** `src/ml/regression/decision-tree.py`
- **Git:** `795c896` (dirty; script untracked)

**What this version is:**
v1 columns, lag, cutoff, and `max_depth` grid on a 25y Yahoo SPY cache.

**What changed vs v1:**
- Data only: `data/clean-yfinance/SPY.csv` 10y (2016-09-01 to 2026-08-31, 2512 rows) → 25y (2001-09-04 to 2026-09-01, 6285 rows). Other tickers left at 10y; they are loaded and unused.
- Features, target, cutoff, model, and CV grid are unchanged.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance` SPY period 25y (head 2001-09-04) |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all `STARTING_STOCKS` (20 names in `src/ml/config.py`) |
| Symbols trained | SPY |
| Features | `1_Day_Return`, `Overnight`, `Range`, `Upper_Wick`, `Lower_Wick` at lag `_0` (5 cols). Built by `ut.process_OHLCV_all`; split uses `BASE_FEATURE_COLS` |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `sklearn.tree.DecisionTreeRegressor(random_state=42)` |
| Other | no scaler; `GridSearchCV` scoring `neg_mean_squared_error` with `TimeSeriesSplit(n_splits=5, gap=5)` on train; `max_depth` grid `{2, 4, 6, 8, 10, 15, 20, None}`; best `max_depth=2` (grid floor); 5783 train / 500 test rows; `plot_data` tree + time series + scatter; `DISPLAY_PLOTS=True`; unused derived columns still computed |

**Metrics**

| Metric | Train | Test |
|--------|-------|------|
| MSE | 0.000139001082539492 | 0.00010886806563246096 |
| R² | 0.047298297870795425 | -0.0015651249328039185 |
| Best `max_depth` | 2 | (same fit) |

Depth-2 leaves (mean next-day return of the train rows in that leaf):

| Leaf prediction | Train n | Test n | What it is |
|-----------------|---------|--------|------------|
| +0.000332 | 5775 | 500 | ordinary days |
| +0.052386 | 7 | 0 | 2008-09-29, 2008-10-15, 2008-11-20, 2008-12-01, 2020-03-09, 2020-03-12, 2020-03-16 |
| +0.145198 | 1 | 0 | 2008-10-10 |

Splits: `Range_0 <= 0.115698`, then `1_Day_Return_0 <= -0.072037`. Overnight and both wicks have importance 0. Unique test predictions: `{0.000332}` only.

Train-mean baseline on this holdout: test MSE 0.000108803, R² -0.000968.

**Read of the run:**
CV still picked depth 2. The tree is still a crash-day splitter. Extra history added 2008 leaves and moved the range threshold from 8.5% (v1) to 11.6%, so 2025-04-09 no longer hits a crash leaf. Every test day gets the bulk mean. Test R² is near a constant predictor, not a daily forecast. `src/ml/config.py` `PERIOD` is still `"10y"`; only the SPY cache was replaced.

**vs previous:**

| | Previous (v1) | This (v2) |
|--|-----------------|-----------|
| Change | 10y SPY, 2011 train / 499 test | 25y SPY, 5783 train / 500 test |
| Train MSE | 0.00012100921362891755 | 0.000139001082539492 (+0.00001799, worse) |
| Train R² | 0.0900063402409047 | 0.047298297870795425 (-0.0427, worse) |
| Test MSE | 0.00011351718270836929 | 0.00010886806563246096 (-0.00000465, better) |
| Test R² | -0.043386618590599335 | -0.0015651249328039185 (+0.0418, better) |
| Best `max_depth` | 2 | 2 |

Holdout cutoff is the same. Test looks better because the one v1 crash prediction on 2025-04-09 is gone, not because the tree found a daily rule. Test R² is still negative and slightly worse than predicting the training mean.

**Terminal excerpt**

```
Best params: {'decisiontreeregressor__max_depth': 2}
MSE: 0.000139001082539492
R^2: 0.047298297870795425
Best params: {'decisiontreeregressor__max_depth': 2}
MSE: 0.00010886806563246096
R^2: -0.0015651249328039185
```

### v1 - lag-1 bar-shape baseline, SPY only

- **Date:** 2026-09-01
- **Status:** superseded
- **Script:** `src/ml/regression/decision-tree.py`
- **Git:** `795c896` (dirty; script untracked)

**What this version is:**
First logged tree: five lag-0 columns, `GridSearchCV` over `max_depth`.

**What changed vs vN-1:**
First logged run.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`, period 10y |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all `STARTING_STOCKS` (20 names in `src/ml/config.py`) |
| Symbols trained | SPY |
| Features | `1_Day_Return`, `Overnight`, `Range`, `Upper_Wick`, `Lower_Wick` at lag `_0` (5 cols). Built by `ut.process_OHLCV_all`; split uses `BASE_FEATURE_COLS` |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `sklearn.tree.DecisionTreeRegressor(random_state=42)` |
| Other | no scaler; `GridSearchCV` scoring `neg_mean_squared_error` with `TimeSeriesSplit(n_splits=5, gap=5)` on train; `max_depth` grid `{2, 4, 6, 8, 10, 15, 20, None}`; best `max_depth=2` (grid floor); `plot_data` tree + time series + scatter; `DISPLAY_PLOTS=True`; unused derived columns still computed (volume/SMA windows drop extra early-train rows) |

**Metrics**

| Metric | Train | Test |
|--------|-------|------|
| MSE | 0.00012100921362891755 | 0.00011351718270836929 |
| R² | 0.0900063402409047 | -0.043386618590599335 |
| Best `max_depth` | 2 | (same fit) |

**Read of the run:**
CV picked the shallowest tree. Train R² is 0.09; test R² is negative. Same-day 1-day return, overnight, and bar-shape vs a next-day return. Earlier invocations in this session printed the same metrics then crashed in `plot_tree` (`GridSearchCV`, then `Pipeline`, then `class_names='Adj Close'`). Later runs completed both plots.

**vs previous:**
First logged run.

**Latest run notes:**
Several invocations of the same recipe. Metrics identical. Plot crashes until the fitted `DecisionTreeRegressor` was passed to `plot_tree`.

**Terminal excerpt**

```
Best params: {'decisiontreeregressor__max_depth': 2}
MSE: 0.00012100921362891755
R^2: 0.0900063402409047
Best params: {'decisiontreeregressor__max_depth': 2}
MSE: 0.00011351718270836929
R^2: -0.043386618590599335
```
