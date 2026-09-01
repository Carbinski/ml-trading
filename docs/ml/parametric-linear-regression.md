# Parametric linear regression

**Script:** `src/ml/regression/parametric-regression.py`
**Family:** regression
**Current:** v3 (`current`)

Degree-2 polynomial expansion of lagged raw OHLCV, then unregularized OLS, to predict next-day `Adj Close` return. Still SPY only.

## Versions

### v3 - degree-2 polynomial on lagged raw OHLCV, SPY only

- **Date:** 2026-08-31
- **Status:** current
- **Script:** `src/ml/regression/parametric-regression.py`
- **Git:** `b2911f2` (dirty)

**What this version is:**
Same 30 lagged raw OHLCV columns and next-day return target as v2, but the estimator is a `Pipeline` of `PolynomialFeatures(degree=2, include_bias=False)` then `LinearRegression` (495 expanded terms).

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
First print crashed (`AttributeError: 'Pipeline' object has no attribute 'coef_'`). After reading `coef_` / `intercept_` from the inner regressor and names from `PolynomialFeatures.get_feature_names_out`, the train completed. Intercept ~0.00075; printed weights still round to zero. Test R² is far more negative than v2, so the quadratic expansion of raw prices and unscaled volume did not help. Same-day `_0` prices, including `Close_0` and `Adj Close_0`, remain features for a next-day return from `Adj Close`. Twenty symbols loaded; only SPY trained. `plt.show()` warned under a non-interactive Agg backend; the figure code ran.

**vs previous:**

| | Previous (v2) | This (v3) |
|--|-----------------|-----------|
| Change | linear OLS on 30 lagged OHLCV | degree-2 polynomial (495 terms) then OLS |
| MSE | 0.00011111346340266835 | 0.0001890460581161295 (+0.00007793, worse) |
| R² | -0.021292971632680846 | -0.7376059080184323 (-0.7163, worse) |

Worse on both metrics. Test window, cutoff, and target scale match v2, so this is a real step back, not a scoring artifact. Expanding raw price/volume products increased fit flexibility without a useful representation of next-day return.

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

**What this version is:**
Same unregularized `LinearRegression` and next-day return target as v1, but features are five bars of raw OHLCV (`shift(0)` through `shift(4)`), 30 columns.

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
Still a near-constant predictor (intercept ~0.0012, printed weights zero). Test R² is more negative than v1. Same-day `_0` prices, including `Close_0` and `Adj Close_0`, are still features for a next-day return from `Adj Close`, so price level and target stay on incompatible scales. Extra lags add collinear raw prices, not a new representation. Twenty symbols loaded; only SPY trained. An earlier attempt this session hit `IndexingError` until `X` was aligned with the valid mask.

**vs previous:**

| | Previous (v1) | This (v2) |
|--|-----------------|-----------|
| Change | 6 same-day OHLCV | 30 lagged OHLCV (`lag=5`) |
| MSE | 0.00010934437687776823 | 0.00011111346340266835 (+0.00000177, worse) |
| R² | -0.005032515169875795 | -0.021292971632680846 (-0.01626, worse) |

Worse on both metrics. Test window is comparable (lag NaNs only drop 2016 train rows; cutoff and target scale are the same), so this is a real step back, not a scoring artifact. Adding five days of raw prices did not help the model beat a mean-return guess.

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

**What this version is:**
Unregularized `LinearRegression` on raw same-day prices and volume for SPY, chronological 80/20 split on the 10y clean-yfinance window.

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
Coefficients are near zero. The intercept is about 0.0011, in line with a constant mean-return guess, and test R² is slightly negative, so the fit did not beat predicting the mean on the holdout window. Features are same-day prices (including `Close` and `Adj Close`) for a next-day return built from `Adj Close`, so price level and target live on incompatible scales. Volume is unscaled next to prices. Twenty symbols are loaded; only SPY is trained. Earlier attempts in the same session failed (NaN in `y`, stub `plot_data` arity, print bugs) before this metrics-producing run.

**Latest run notes:**
Same intercept, MSE, and R² across the successful prints in this session. Only the weight formatting changed.

**Terminal excerpt**

```
Weights: -0.00 Open, -0.00 High, -0.00 Low, -0.00 Close, 0.00 Volume, -0.00 Adj Close
Intercept: 0.0011004553088680416
MSE: 0.00010934437687776823
R^2: -0.005032515169875795
```
