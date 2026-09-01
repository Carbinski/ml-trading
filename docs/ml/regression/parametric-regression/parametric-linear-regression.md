# Parametric linear regression

**Script:** `src/ml/regression/parametric-regression.py`
**Family:** regression
**Current:** v8 (`current`)

Logged v8: `RidgeCV` (`alpha=10000`) after `StandardScaler` on lag-1 Open/High/Low/`Adj Close`, Range, Overnight, Relative_Volume, 5-day return, 20-day return, and 20-day SMA distance. Next-day `Adj Close` return. SPY only. `PolynomialFeatures(degree=1)` is a pass-through.

Shared across versions: `data/clean-yfinance` 10y, cutoff `2024-09-01`, 20 symbols loaded / SPY trained. Holdout reused since v1. Last train row's label uses the first test-period close. Overnight is raw `Open / Close.shift(1) - 1`, not the split-adjusted formula in `utility.py`. Same-day `_0` prices, including `Adj Close_0`, are features for that return. Weights print at `.2f`.

v8 fit `StandardScaler` on all of train, then `RidgeCV` split that already-scaled matrix. The working tree comments the scaler out; that recipe is not a logged version.

## Versions

### v8 - RidgeCV, lag=1, momentum features, SPY only

- **Date:** 2026-08-31
- **Status:** current
- **Script:** `src/ml/regression/parametric-regression.py`
- **Git:** `d352ecd` (dirty)

**What changed vs v7:**
- Model / procedure: unregularized OLS → `PolynomialFeatures` + `StandardScaler` + `RidgeCV` (best `alpha=10000.0`).
- Feature set: added `5_Day_Return=(Adj Close / Adj Close.shift(5))-1`, `20_Day_Return=(Adj Close / Adj Close.shift(20))-1`, `Dist_From_SMA=(Adj Close - lagged 20-day SMA) / lagged 20-day std`.
- Lag: 5 → 1 (10 same-day columns instead of 35).
- Plots off (`display_plots=False`).
- Data, cutoff, target, degree=1, and SPY-only train are unchanged.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`, period 10y (SPY head starts 2016-09-01) |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all `STARTING_STOCKS` (20 names in `src/ml/config.py`) |
| Symbols trained | SPY |
| Features | `Open`/`High`/`Low`/`Relative_Volume`/`Range`/`Overnight`/`5_Day_Return`/`20_Day_Return`/`Dist_From_SMA`/`Adj Close` at lag `_0` only (10 cols); `PolynomialFeatures(degree=1, include_bias=False)` |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `PolynomialFeatures` + `StandardScaler` + `sklearn.linear_model.RidgeCV` |
| Other | `RidgeCV` alphas `np.logspace(-4, 4, 30)`; CV `TimeSeriesSplit(n_splits=5, gap=5)` on train; best alpha `10000.0` (grid ceiling); summary reads `named_steps["ridgecv"]` (coefs are on scaled features); 20-day windows drop extra 2016 train rows |

**Metrics**

| Metric | Train | Test |
|--------|-------|------|
| MSE | 0.000132676798887128 | 0.00010970328112697492 |
| R² | 0.00578175106948875 | -0.008331362816047738 |
| Intercept | 0.0006038330863263826 | (same fit) |

Weights: all 10 printed as `-0.00` or `0.00` at `.2f`. No higher-precision dump this run.

**Read of the run:**
Alpha sat at the grid ceiling. Printed weights all ±0.00. Same session had a lag `_2` print (30 cols) before this lag=1 re-run.

**vs previous:**

| | Previous (v7) | This (v8) |
|--|-----------------|-----------|
| Change | OLS, lag=5, 35 cols | RidgeCV `alpha=10000` + scaler, lag=1, 10 cols incl. momentum |
| Train MSE | 0.0001231410169295402 | 0.000132676798887128 (+0.00000954, worse) |
| Train R² | 0.07889675643532512 | 0.00578175106948875 (-0.0731, worse) |
| Test MSE | 0.00012932636313988074 | 0.00010970328112697492 (-0.00001962, better) |
| Test R² | -0.18869578606262905 | -0.008331362816047738 (+0.1804, better) |

Holdout looks better than v7, but estimator, lag, and features all changed at once. Numbers sit near v6 (same RidgeCV + scaler, different columns/lag). Test R² is still negative.

**Terminal excerpt**

```
Best alpha: 10000.0
Weights: -0.00 Open_0, -0.00 High_0, -0.00 Low_0, -0.00 Relative_Volume_0, 0.00 Range_0, -0.00 Overnight_0, -0.00 5_Day_Return_0, -0.00 20_Day_Return_0, -0.00 Dist_From_SMA_0, -0.00 Adj Close_0
Intercept: 0.0006038330863263826
MSE: 0.000132676798887128
R^2: 0.00578175106948875
Intercept: 0.0006038330863263826
MSE: 0.00010970328112697492
R^2: -0.008331362816047738
```

### v7 - LinearRegression on v4 features, SPY only

- **Date:** 2026-08-31
- **Status:** superseded
- **Script:** `src/ml/regression/parametric-regression.py`
- **Git:** `d352ecd` (dirty)

**What changed vs v6:**
- Model / procedure: `PolynomialFeatures` + `StandardScaler` + `RidgeCV` (best `alpha=10000.0`) → `PolynomialFeatures` + `LinearRegression`.
- User: changed RidgeCV to LinearRegression.
- Data, cutoff, target, lag=5 builder, derived columns, degree=1, and SPY-only train are unchanged.
- Printed weights have no `5_Day_Return_*` and no `Best alpha`, so this is not the later RidgeCV + `5_Day_Return` recipe sitting in the dirty script.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`, period 10y (SPY head starts 2016-09-01) |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all `STARTING_STOCKS` (20 names in `src/ml/config.py`) |
| Symbols trained | SPY |
| Features | `Open`/`High`/`Low`/`Relative_Volume`/`Range`/`Overnight`/`Adj Close` at lags `_0`..`_4` (35 cols); `PolynomialFeatures(degree=1, include_bias=False)` |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `PolynomialFeatures` + `sklearn.linear_model.LinearRegression` |
| Other | no scaling, no regularization; summary reads `named_steps["linearregression"]`; recipe matches v4 exactly |

**Metrics**

| Metric | Train | Test |
|--------|-------|------|
| MSE | 0.0001231410169295402 | 0.00012932636313988074 |
| R² | 0.07889675643532512 | -0.18869578606262905 |
| Intercept | 0.008145983327903119 | (same fit) |

Non-zero weights match v4 (Range/Overnight lags; Overnight_1 = 0.92).

**Read of the run:**
Byte-for-byte the v4 recipe. OLS without the scaler/Ridge penalty restores the Range/Overnight fit and the worse holdout.

**vs previous:**

| | Previous (v6) | This (v7) |
|--|-----------------|-----------|
| Change | RidgeCV `alpha=10000` + `StandardScaler` | OLS, no scaler (same 35 cols) |
| Train MSE | 0.00013200575051082363 | 0.0001231410169295402 (-0.00000886, better) |
| Train R² | 0.012587941885507381 | 0.07889675643532512 (+0.0663, better) |
| Test MSE | 0.0001106144699413259 | 0.00012932636313988074 (+0.00001871, worse) |
| Test R² | -0.016706502096465625 | -0.18869578606262905 (-0.1720, worse) |

Return to the v4 recipe. Train fit comes back; holdout gets worse than v6.

### v6 - RidgeCV + StandardScaler on v4 features, SPY only

- **Date:** 2026-08-31
- **Status:** superseded
- **Script:** `src/ml/regression/parametric-regression.py`
- **Git:** `d352ecd` (dirty)

**What changed vs v5:**
- Training procedure: `Ridge(alpha=0.05)` → `PolynomialFeatures(degree=1)` + `StandardScaler` + `RidgeCV` (train-only time-series CV, `scoring="neg_mean_squared_error"`).
- Regularization strength: fixed `0.05` → selected `10000.0`.
- Data, cutoff, target, lag=5 builder, derived columns, degree=1, and SPY-only train are unchanged.
- An earlier invocation in the same session printed degree-2 interaction weights (test MSE 0.000120, R² -0.106, intercept 0.000606) before this degree-1 re-run. That leftover is not this version.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`, period 10y (SPY head starts 2016-09-01) |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all `STARTING_STOCKS` (20 names in `src/ml/config.py`) |
| Symbols trained | SPY |
| Features | `Open`/`High`/`Low`/`Relative_Volume`/`Range`/`Overnight`/`Adj Close` at lags `_0`..`_4` (35 cols); `PolynomialFeatures(degree=1, include_bias=False)` |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `PolynomialFeatures` + `StandardScaler` + `sklearn.linear_model.RidgeCV` |
| Other | `RidgeCV` alphas `np.logspace(-4, 4, 30)`; CV `TimeSeriesSplit(n_splits=5, gap=5)` on train; best alpha `10000.0` (grid ceiling); summary reads `named_steps["ridgecv"]` (coefs are on scaled features) |

**Metrics**

| Metric | Train | Test |
|--------|-------|------|
| MSE | 0.00013200575051082363 | 0.0001106144699413259 |
| R² | 0.012587941885507381 | -0.016706502096465625 |
| Intercept | 0.0006062457285380145 | (same fit) |

Weights: all 35 printed as `-0.00` or `0.00` at `.2f`. No higher-precision dump this run.

**Read of the run:**
Alpha sat at the grid ceiling. Printed weights all ±0.00. `plt.show()` was still open when this was logged.

**vs previous:**

| | Previous (v5) | This (v6) |
|--|-----------------|-----------|
| Change | Ridge `alpha=0.05`, no scaler | RidgeCV `alpha=10000` + `StandardScaler` |
| Train MSE | 0.0001266665138047068 | 0.00013200575051082363 (+0.00000534, worse) |
| Train R² | 0.052525798261809586 | 0.012587941885507381 (-0.0399, worse) |
| Test MSE | 0.00012158900756333152 | 0.0001106144699413259 (-0.00001097, better) |
| Test R² | -0.11757832983938177 | -0.016706502096465625 (+0.1009, better) |

Regularization plus scaling traded train fit for a better holdout. Test R² is still negative and a bit worse than v1.

**Terminal excerpt**

```
Best alpha: 10000.0
Weights: (35 cols, all ±0.00 at .2f)
Intercept: 0.0006062457285380145
MSE: 0.00013200575051082363
R^2: 0.012587941885507381
Intercept: 0.0006062457285380145
MSE: 0.0001106144699413259
R^2: -0.016706502096465625
```

### v5 - Ridge (alpha=0.05) on v4 features, SPY only

- **Date:** 2026-08-31
- **Status:** superseded
- **Script:** `src/ml/regression/parametric-regression.py`
- **Git:** `d352ecd` (dirty)

**What changed vs v4:**
- Model class / hyperparameter only: `LinearRegression` → `Ridge(alpha=0.05)`.
- Data, cutoff, target, lag=5 builder, derived columns, degree=1, and SPY-only train are unchanged.
- `print_model_summary` still prints train then test. `plot_data` now takes a title; the first Ridge attempt crashed after metrics (`plot_data() missing 1 required positional argument: 'y_pred'`). A second run of the same recipe printed the same numbers and reached both plots.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`, period 10y (SPY head starts 2016-09-01) |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all `STARTING_STOCKS` (20 names in `src/ml/config.py`) |
| Symbols trained | SPY |
| Features | `Open`/`High`/`Low`/`Relative_Volume`/`Range`/`Overnight`/`Adj Close` at lags `_0`..`_4` (35 cols); `PolynomialFeatures(degree=1, include_bias=False)` |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `PolynomialFeatures` + `sklearn.linear_model.Ridge(alpha=0.05)` |
| Other | no scaling; `Relative_Volume` uses a 20-day rolling mean shifted by 1, so more 2016 train rows drop than in v3; summary reads `named_steps["ridge"]` |

**Metrics**

| Metric | Train | Test |
|--------|-------|------|
| MSE | 0.0001266665138047068 | 0.00012158900756333152 |
| R² | 0.052525798261809586 | -0.11757832983938177 |
| Intercept | 0.009916617455613038 | (same fit) |

Weights that print off zero at `.2f` (Open/High/Low/Relative_Volume/`Adj Close` lags all ±0.00):

| Feature | Weight |
|---------|--------|
| Range_0 | 0.02 |
| Overnight_0 | -0.05 |
| Range_1 | -0.01 |
| Overnight_1 | 0.12 |
| Range_2 | 0.02 |
| Overnight_2 | 0.03 |
| Range_3 | 0.02 |
| Overnight_3 | 0.00 |
| Range_4 | 0.02 |
| Overnight_4 | 0.04 |

**Read of the run:**
Ridge shrinks v4's Overnight_1 from 0.92 to 0.12. First attempt printed these metrics then crashed in `plot_data` (missing title); the re-run matched.

**vs previous:**

| | Previous (v4) | This (v5) |
|--|-----------------|-----------|
| Change | OLS on 35 derived/lagged cols | Ridge `alpha=0.05`, same cols |
| Train MSE | 0.0001231410169295402 | 0.0001266665138047068 (+0.00000353, worse) |
| Train R² | 0.07889675643532512 | 0.052525798261809586 (-0.0264, worse) |
| Test MSE | 0.00012932636313988074 | 0.00012158900756333152 (-0.00000774, better) |
| Test R² | -0.18869578606262905 | -0.11757832983938177 (+0.0711, better) |

Regularization traded train fit for a better holdout. Test R² is still negative and worse than v1/v2.

**Latest run notes:**
Two invocations of the Ridge recipe. Metrics identical. First died in `plot_data` (title argument); second completed both panels.

**Terminal excerpt**

```
Weights: -0.00 Open_0, ... 0.02 Range_0, -0.05 Overnight_0, ... 0.12 Overnight_1, ... 0.04 Overnight_4, 0.00 Adj Close_4
Intercept: 0.009916617455613038
MSE: 0.0001266665138047068
R^2: 0.052525798261809586
Intercept: 0.009916617455613038
MSE: 0.00012158900756333152
R^2: -0.11757832983938177
```

### v4 - degree 1, derived Range/Overnight/Relative_Volume, SPY only

- **Date:** 2026-08-31
- **Status:** superseded
- **Script:** `src/ml/regression/parametric-regression.py`
- **Git:** `d352ecd` (dirty)

**What changed vs v3:**
- Feature set: lagged raw OHLCV (`Close`/`Volume` included) → `Open`/`High`/`Low`/`Adj Close` plus `Range=(High-Low)/Close`, `Overnight=Open/Close.shift(1)-1`, `Relative_Volume=Volume/(20-day mean).shift(1)`.
- Polynomial degree 2 → 1 (35 columns, no squares or pairwise products).
- Script now prints train metrics then test metrics.
- Data, cutoff, target, lag=5, and SPY-only train are unchanged.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`, period 10y (SPY head starts 2016-09-01) |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all `STARTING_STOCKS` (20 names in `src/ml/config.py`) |
| Symbols trained | SPY |
| Features | `Open`/`High`/`Low`/`Relative_Volume`/`Range`/`Overnight`/`Adj Close` at lags `_0`..`_4` (35 cols); `PolynomialFeatures(degree=1, include_bias=False)` |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `PolynomialFeatures` + `sklearn.linear_model.LinearRegression` |
| Other | no scaling, no regularization; `Relative_Volume` rolling window drops extra 2016 train rows |

**Metrics**

| Metric | Train | Test |
|--------|-------|------|
| MSE | 0.0001231410169295402 | 0.00012932636313988074 |
| R² | 0.07889675643532512 | -0.18869578606262905 |
| Intercept | 0.008145983327903119 | (same fit) |

Weights that print off zero at `.2f` (price/volume-style lags all ±0.00):

| Feature | Weight |
|---------|--------|
| Range_0 | 0.31 |
| Overnight_0 | -0.50 |
| Range_1 | -0.39 |
| Overnight_1 | 0.92 |
| Range_2 | 0.18 |
| Overnight_2 | 0.21 |
| Range_3 | 0.11 |
| Overnight_3 | -0.21 |
| Range_4 | -0.01 |
| Overnight_4 | 0.04 |

**Read of the run:**
Range and Overnight take the non-zero weights; Overnight_1 is 0.92. Those columns are return-like; raw Open/High/Low/`Adj Close` lags still print ±0.00.

**vs previous:**

| | Previous (v3) | This (v4) |
|--|-----------------|-----------|
| Change | degree-2 on 30 raw OHLCV (495 terms) | degree-1 on 35 derived/lagged cols |
| Test MSE | 0.0001890460581161295 | 0.00012932636313988074 (-0.00005972, better) |
| Test R² | -0.7376059080184323 | -0.18869578606262905 (+0.5489, better) |

Better than v3 on the holdout. Still worse than v1/v2. v3 did not print train metrics, so those are not compared.

**Terminal excerpt**

```
Weights: 0.00 Open_0, ... 0.31 Range_0, -0.50 Overnight_0, ... 0.92 Overnight_1, ... 0.04 Overnight_4, -0.00 Adj Close_4
Intercept: 0.008145983327903119
MSE: 0.0001231410169295402
R^2: 0.07889675643532512
Intercept: 0.008145983327903119
MSE: 0.00012932636313988074
R^2: -0.18869578606262905
```

### v3 - degree-2 polynomial on lagged raw OHLCV, SPY only

- **Date:** 2026-08-31
- **Status:** superseded
- **Script:** `src/ml/regression/parametric-regression.py`
- **Git:** `b2911f2` (dirty)

**What changed vs v2:**
- Model class / procedure: plain `LinearRegression` → `make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())`.
- Feature representation after the transformer: 30 lagged columns expanded to squares and pairwise products (495 weights).
- `plot_data` now draws the two-panel actual-vs-predicted figure (was a no-op TODO in v2).
- Data, cutoff, target, lag=5 builder, and SPY-only train are unchanged.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`, period 10y (SPY head starts 2016-09-01) |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all `STARTING_STOCKS` (20 names in `src/ml/config.py`) |
| Symbols trained | SPY |
| Features | `Open`/`High`/`Low`/`Close`/`Volume`/`Adj Close` at lags `_0`..`_4`, then degree-2 polynomial (no bias column) |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `PolynomialFeatures` + `sklearn.linear_model.LinearRegression` |
| Other | no scaling, no regularization; first four rows dropped for lag NaNs (2016, train only); summary reads `named_steps["linearregression"]` |

**Metrics**

| Metric | Value |
|--------|--------|
| MSE | 0.0001890460581161295 |
| R² | -0.7376059080184323 |
| Intercept | 0.000747398407886868 |

Weights: 495 terms printed, all `-0.00` or `0.00` at `.2f`. No higher-precision dump this run.

**Read of the run:**
First print crashed (`AttributeError: 'Pipeline' object has no attribute 'coef_'`). After reading `coef_` / `intercept_` from the inner regressor, the run completed. Quadratic expansion of raw prices and unscaled volume made test R² far more negative than v2. `plt.show()` warned under Agg; the figure code ran.

**vs previous:**

| | Previous (v2) | This (v3) |
|--|-----------------|-----------|
| Change | linear OLS on 30 lagged OHLCV | degree-2 polynomial (495 terms) then OLS |
| MSE | 0.00011111346340266835 | 0.0001890460581161295 (+0.00007793, worse) |
| R² | -0.021292971632680846 | -0.7376059080184323 (-0.7163, worse) |

Worse on both metrics. Expanding raw price/volume products did not help.

**Terminal excerpt**

```
Weights: -0.00 Open_0, ... (495 terms, all ±0.00 at .2f)
Intercept: 0.000747398407886868
MSE: 0.0001890460581161295
R^2: -0.7376059080184323
```

### v2 - same model, 5-bar lagged raw OHLCV, SPY only

- **Date:** 2026-08-31
- **Status:** superseded
- **Script:** `src/ml/regression/parametric-regression.py`
- **Git:** `b2911f2` (dirty)

**What changed vs v1:**
- Feature set only: 6 same-day columns → 30 lagged price/volume columns (`lag=5`). Data, cutoff, target, model, and SPY-only train are unchanged.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`, period 10y (SPY head starts 2016-09-01) |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all `STARTING_STOCKS` (20 names in `src/ml/config.py`) |
| Symbols trained | SPY |
| Features | `Open`/`High`/`Low`/`Close`/`Volume`/`Adj Close` at lags `_0`..`_4` (`_0` is same-day) |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.linear_model.LinearRegression` |
| Other | no scaling, no regularization; `plot_data` is a no-op TODO; first four rows dropped for lag NaNs (2016, train only) |

**Metrics**

| Metric | Value |
|--------|--------|
| MSE | 0.00011111346340266835 |
| R² | -0.021292971632680846 |
| Intercept | 0.0012111266945216284 |

Weights: all 30 printed as `-0.00` or `0.00` at `.2f`. Volume lags `_0`, `_2`, `_3`, `_4` printed `0.00`; the rest printed `-0.00`. No higher-precision dump this run.

**Read of the run:**
Near-constant predictor. Extra lags add collinear raw prices. An earlier attempt this session hit `IndexingError` until `X` was aligned with the valid mask.

**vs previous:**

| | Previous (v1) | This (v2) |
|--|-----------------|-----------|
| Change | 6 same-day OHLCV | 30 lagged OHLCV (`lag=5`) |
| MSE | 0.00010934437687776823 | 0.00011111346340266835 (+0.00000177, worse) |
| R² | -0.005032515169875795 | -0.021292971632680846 (-0.01626, worse) |

Worse on both metrics. Five days of raw prices did not beat a mean-return guess.

**Terminal excerpt**

```
Weights: -0.00 Open_0, ... (30 cols, all ±0.00 at .2f)
Intercept: 0.0012111266945216284
MSE: 0.00011111346340266835
R^2: -0.021292971632680846
```

### v1 - raw OHLCV baseline, SPY only

- **Date:** 2026-08-31
- **Status:** superseded
- **Script:** `src/ml/regression/parametric-regression.py`
- **Git:** `b2911f2` (dirty)

**What changed vs vN-1:**
First logged run.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`, period 10y (SPY head starts 2016-09-01) |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all `STARTING_STOCKS` (20 names in `src/ml/config.py`) |
| Symbols trained | SPY |
| Features | same-day `Open`, `High`, `Low`, `Close`, `Volume`, `Adj Close` |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.linear_model.LinearRegression` |
| Other | no scaling, no regularization; `plot_data` is a no-op TODO; script file also contains a `lag=5` lagged-OHLCV builder that was not what this run printed |

**Metrics**

| Metric | Value |
|--------|--------|
| MSE | 0.00010934437687776823 |
| R² | -0.005032515169875795 |
| Intercept | 0.0011004553088680416 |

Weights (higher precision from the same recipe earlier in the session; `.2f` print rounds all to `0.00`):

| Feature | Weight |
|---------|--------|
| Open | -4.21311833e-07 |
| High | -4.24253657e-07 |
| Low | -4.18020243e-07 |
| Close | -4.21216135e-07 |
| Volume | 2.84552128e-12 |
| Adj Close | -4.44990511e-07 |

**Read of the run:**
Coefficients near zero; intercept ~0.0011 is a mean-return guess. Test R² slightly negative. Price level vs return target; unscaled volume next to prices. Earlier attempts failed (NaN in `y`, stub `plot_data` arity, print bugs) before this run.

**Latest run notes:**
Same intercept, MSE, and R² across the successful prints in this session. Only the weight formatting changed.

**Terminal excerpt**

```
Weights: -0.00 Open, -0.00 High, -0.00 Low, -0.00 Close, 0.00 Volume, -0.00 Adj Close
Intercept: 0.0011004553088680416
MSE: 0.00010934437687776823
R^2: -0.005032515169875795
```
