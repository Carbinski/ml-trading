# Voting regressor

**Script:** `src/ml/ensemble-learning/model-ensemble.py`
**Family:** ensemble
**Current:** v2 (`current`)

Equal-weight `VotingRegressor` of Ridge, RBF SVR, a depth-2 tree, and k=51 kNN. Next-day `Adj Close` return. SPY only. v1 fit `asinh` of that return and inverted before metrics; v2 fits the raw return.

Shared: cutoff `2024-09-01`. Loads 20 names, trains SPY. Same-day derived columns are features for that return. Overnight is split-adjusted open vs prior `Adj Close`.

## Versions

### v2 - raw next-day return, no asinh

- **Date:** 2026-09-01
- **Status:** current
- **Script:** `src/ml/ensemble-learning/model-ensemble.py`
- **Git:** `795c896` (dirty; script untracked)

**What this version is:**
v1 members, features, lag, and cutoff, trained and scored on the raw next-day return.

**What changed vs v1:**
- Training procedure / target scale: fit on `z = arcsinh(y / 0.01)` and invert with `0.01 * sinh` → fit and score on `y` directly.
- Data, cutoff, features, members, and equal-weight vote are unchanged.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance` SPY period 25y (head 2001-09-04). Other 19 names period 10y, loaded and unused |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all `STARTING_STOCKS` (20 names in `src/ml/config.py`) |
| Symbols trained | SPY |
| Features | `1_Day_Return`, `Overnight`, `Range`, `Close_Location`, `Rel_Vol` at lag `_0` (5 cols). Built by `ut.process_OHLCV_all`; split uses `BASE_FEATURE_COLS` |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.ensemble.VotingRegressor`, equal weights, no `weights=` |
| Other | members: `StandardScaler` + `RidgeCV` (alphas `np.logspace(-4, 4, 30)`, `TimeSeriesSplit(n_splits=5, gap=5)` on train, scoring `neg_mean_squared_error`); `StandardScaler` + `SVR(kernel="rbf", C=1.0, epsilon=0.01, cache_size=2000, max_iter=16000)`; `DecisionTreeRegressor(max_depth=2, random_state=42)`; `StandardScaler` + `KNeighborsRegressor(n_neighbors=51)`. No joint `GridSearchCV`. `plot_data` time series + scatter with member overlays; `DISPLAY_PLOTS=True`. Unused derived columns still computed. `src/ml/config.py` `PERIOD` is still `"10y"` |

**Metrics**

| Metric | Train | Test |
|--------|-------|------|
| MSE | 0.00012204418620107614 | 0.00011173366566303749 |
| R² | 0.15843863922376333 | -0.027928090380552772 |
| Ridge best alpha | 788.0462815669904 | (same fit) |

No member-level holdout numbers. No intercept print (`VotingRegressor`). SVR hit `ConvergenceWarning` (`max_iter=16000`).

**Read of the run:**
User: drop the asinh squash and log the raw-return ensemble. Train R² rose to 0.158; test R² moved the other way. Ridge CV picked `alpha=788` on raw `y` (v1 was 2807 on `z`). Same-day 1-day return, overnight, range, close location, and rel vol vs a next-day return from `Adj Close`. Test is still worse than a constant mean and worse than linear v8 / tree v2 on this window.

**vs previous:**

| | Previous (v1) | This (v2) |
|--|-----------------|-----------|
| Change | fit `asinh(y / 0.01)`, invert before MSE / R² | fit and score raw `y` |
| Train MSE | 0.00013968090709078995 | 0.00012204418620107614 (−0.00001764, better) |
| Train R² | 0.03682381025416048 | 0.15843863922376333 (+0.1216, better) |
| Test MSE | 0.00010969259508270594 | 0.00011173366566303749 (+0.00000204, worse) |
| Test R² | -0.009150636230794262 | -0.027928090380552772 (−0.0188, worse) |
| Ridge best alpha | 2807.2162039411755 | 788.0462815669904 |

Same data, features, cutoff, and members. Removing the squash bought train fit and lost holdout. The sklearn scaler warning on SVR is a false lead; that member already has `StandardScaler`.

**Terminal excerpt**

```
Ridge best alpha: 788.0462815669904
Training model summary:
Estimators: ridge, svr, tree, knn
MSE: 0.00012204418620107614
R^2: 0.15843863922376333
Test model summary:
Estimators: ridge, svr, tree, knn
MSE: 0.00011173366566303749
R^2: -0.027928090380552772
```

### v1 - asinh target, 5-col lag-1, SPY only

- **Date:** 2026-09-01
- **Status:** superseded
- **Script:** `src/ml/ensemble-learning/model-ensemble.py`
- **Git:** `795c896` (dirty; script untracked)

**What this version is:**
First logged ensemble: four members, no joint search, train on `asinh(y / 0.01)`, score after `0.01 * sinh`.

**What changed vs vN-1:**
First logged run.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance` SPY period 25y (head 2001-09-04). Other 19 names period 10y, loaded and unused |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all `STARTING_STOCKS` (20 names in `src/ml/config.py`) |
| Symbols trained | SPY |
| Features | `1_Day_Return`, `Overnight`, `Range`, `Close_Location`, `Rel_Vol` at lag `_0` (5 cols). Built by `ut.process_OHLCV_all`; split uses `BASE_FEATURE_COLS` |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1`. Fit uses `z = arcsinh(y / 0.01)`. Predictions invert with `y_hat = 0.01 * sinh(z_hat)` before MSE / R² |
| Model | `sklearn.ensemble.VotingRegressor`, equal weights, no `weights=` |
| Other | members: `StandardScaler` + `RidgeCV` (alphas `np.logspace(-4, 4, 30)`, `TimeSeriesSplit(n_splits=5, gap=5)` on `z`, scoring `neg_mean_squared_error`); `StandardScaler` + `SVR(kernel="rbf", C=1.0, epsilon=0.01, cache_size=2000, max_iter=16000)`; `DecisionTreeRegressor(max_depth=2, random_state=42)`; `StandardScaler` + `KNeighborsRegressor(n_neighbors=51)`. No joint `GridSearchCV`. Vote averages `z`, then `sinh`. `plot_data` time series + scatter with inverted member overlays; `DISPLAY_PLOTS=True`. Unused derived columns still computed. `src/ml/config.py` `PERIOD` is still `"10y"` |

**Metrics**

| Metric | Train | Test |
|--------|-------|------|
| MSE | 0.00013968090709078995 | 0.00010969259508270594 |
| R² | 0.03682381025416048 | -0.009150636230794262 |
| Ridge best alpha | 2807.2162039411755 | (same fit) |

No member-level holdout numbers. No intercept print (`VotingRegressor`).

**Read of the run:**
User: squash extreme returns so they are less drastic. Training is on `asinh` space; reported MSE / R² are on the raw return after invert, so they sit on the same scale as the other method pages. Ridge CV picked `alpha=2807` on `z` (not the `10000` ceiling from linear v8). SVR still uses `C=1.0`; `epsilon=0.01` is tight relative to `z` (std about 1). Same-day 1-day return, overnight, range, close location, and rel vol vs a next-day return from `Adj Close`. Test R² is negative and in the same band as linear v8 (test R² -0.008) and tree v2 (test R² -0.002) on this SPY window.

**vs previous:**
First version.

**Terminal excerpt**

```
Ridge best alpha: 2807.2162039411755
Training model summary:
Estimators: ridge, svr, tree, knn
MSE: 0.00013968090709078995
R^2: 0.03682381025416048
Test model summary:
Estimators: ridge, svr, tree, knn
MSE: 0.00010969259508270594
R^2: -0.009150636230794262
```
