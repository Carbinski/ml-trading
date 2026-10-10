# Kernel SVR

**Script:** `src/ml/regression/ai_scripts/ai-svm-regression.py`
**Family:** regression
**Current:** v6 (`current`)

Per-ticker RBF `SVR`. v6 fits frozen v4/v5 lag-1 columns at `C=0.01`, `gamma=0.01`, `epsilon=0.01` on all 20 tickers (no `GridSearchCV`).

Shared: cutoff `2024-09-01`. Same-day derived columns are features for that return. Overnight is split-adjusted open vs prior `Adj Close`. SPY cache is the 25y file (head 2001-09-04); the other 19 names are 10y.

## Versions

### v6 - frozen v4/v5 features, all 20 tickers, C=0.01 / gamma=0.01

- **Date:** 2026-09-01
- **Status:** current
- **Script:** `src/ml/regression/ai_scripts/ai-svm-regression.py`
- **Git:** `795c896` (dirty; script untracked)

**What this version is:**
Frozen SPY v4/v5 column sets and frozen `C`/`gamma`/`epsilon`, one RBF SVR per ticker on every CSV in `data/clean-yfinance`.

**What changed vs v5:**
- Universe: SPY only → all 20 tickers.
- Script: `src/ml/regression/svm.py` → `src/ml/regression/ai_scripts/ai-svm-regression.py`.
- No `GridSearchCV`. `C=0.01`, `gamma=0.01`, `epsilon=0.01` frozen from the SPY v4/v5 train-only CV.
- Both the 6-col v4 set and the 10-col v5 set (includes same-day `Adj Close`) are fit on each ticker.
- Data dir, cutoff, target, scaler, RBF kernel, and lag are unchanged.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`. 19 names period 10y (head 2016-09-01). SPY period 25y (head 2001-09-04) |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all 20 CSVs in `data/clean-yfinance` |
| Symbols trained | AAPL, AMZN, CAT, GOOG, IBM, JNJ, JPM, KO, LIN, META, MSFT, NEE, NVDA, PG, PLD, SPY, UNH, UNP, V, XOM (one model each, each recipe) |
| Features | two frozen lag `_0` sets from SPY SVM v4/v5. v4: `1_Day_Return`, `Overnight`, `Range`, `Signed_Rel_Vol`, `Upper_Wick`, `Lower_Wick`. v5: those plus `Adj Close`, `20_Day_Return`, `ATR_20`, `Dist_From_SMA` |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `StandardScaler` + `sklearn.svm.SVR(kernel="rbf", C=0.01, gamma=0.01, epsilon=0.01, cache_size=500, max_iter=16000, tol=1e-3)` |
| Other | no CV; ~1992 train / ~499 test rows per 10y ticker; SPY 5764 / 500; mean `n_support` ≈ 850; finished in 5.0s; `--output` JSON optional |

**Metrics**

Universe (unweighted mean of per-ticker metrics):

| Recipe | Macro train R² | Macro test R² | Median test R² | Test R² > 0 | Beat train-mean | Macro test corr | Dir acc |
|--------|----------------|---------------|----------------|-------------|-----------------|-----------------|---------|
| v4 | 0.0256 | -0.0074 | -0.0044 | 5/20 | 6/20 | -0.0009 | 0.520 |
| v5 | 0.0353 | -0.0137 | -0.0088 | 4/20 | 4/20 | 0.0191 | 0.514 |

v5 vs v4 test R²: better on 7/20 (AAPL, IBM, LIN, NVDA, PG, SPY, XOM). Train R² is higher on v5 for every ticker, so the "lower train, better test" pattern is 0/20. Macro test R² Δ = −0.0063.

Per-ticker test R²:

| Symbol | v4 | v5 | Better |
|--------|-----|-----|--------|
| AAPL | -0.0441 | -0.0361 | v5 |
| AMZN | -0.0040 | -0.0069 | v4 |
| CAT | -0.0167 | -0.0298 | v4 |
| GOOG | -0.0017 | -0.0276 | v4 |
| IBM | -0.0065 | -0.0038 | v5 |
| JNJ | -0.0041 | -0.0409 | v4 |
| JPM | 0.0027 | -0.0504 | v4 |
| KO | -0.0104 | -0.0163 | v4 |
| LIN | 0.0078 | 0.0134 | v5 |
| META | -0.0048 | -0.0103 | v4 |
| MSFT | -0.0134 | -0.0148 | v4 |
| NEE | -0.0047 | -0.0074 | v4 |
| NVDA | -0.0021 | -0.0009 | v5 |
| PG | -0.0036 | 0.0048 | v5 |
| PLD | 0.0036 | 0.0008 | v4 |
| SPY | -0.0404 | -0.0172 | v5 |
| UNH | -0.0089 | -0.0293 | v4 |
| UNP | 0.0041 | 0.0004 | v4 |
| V | 0.0086 | -0.0005 | v4 |
| XOM | -0.0101 | -0.0020 | v5 |

v4 positive test R²: JPM, LIN, PLD, UNP, V. v5: LIN, PG, PLD, UNP. LIN is the only name positive on both. v4 beat the train-mean baseline on GOOG plus those five.

SPY rows match the logged v4/v5 SPY-only refits (v4 test R² −0.0404, `n_support=1504`; v5 −0.0172, `n_support=1492`).

**Read of the run:**
The small-`C` / small-`gamma` surface is close to a mean predictor across the book. A few names have tiny positive holdout R² (largest LIN v5 0.0134) on ~499 test days. Same-day features for a next-day return from `Adj Close`. The v5 recipe still uses raw same-day `Adj Close` as a feature. Hyperparameters were not re-tuned per ticker.

**vs previous:**

| | Previous (v5) | This (v6) |
|--|-----------------|-----------|
| Change | SPY only, 10-col grid search | all 20 tickers, frozen C/gamma, v4 and v5 recipes |
| Symbols trained | SPY | 20 |
| Macro test R² | n/a (one ticker) | v4 −0.0074; v5 −0.0137 |
| SPY test R² (v5 cols) | -0.017194530258363372 | -0.0172 (same frozen refit) |
| SPY test R² (v4 cols) | -0.04035546307829496 (v4) | -0.0404 (same frozen refit) |
| v5 vs v4 on holdout | SPY only, v5 better | v5 better on 7/20; worse on the book |

Universe expansion, not a new kernel. The SPY-only v5 lift (−0.040 → −0.017) does not hold as a universe result: extra columns raised train R² everywhere and hurt macro test R². Still below a constant mean and in the same range as kNN v8 (v5 k=51 macro test R² −0.0111) and linear v8 / tree v2 on SPY.

**Terminal excerpt**

```
v4: macro train R2=0.0256, macro test R2=-0.0074, median test R2=-0.0044, positive test R2=5/20, beat train-mean baseline=6/20, macro test corr=-0.0009, macro dir acc=0.5200.
v5: macro train R2=0.0353, macro test R2=-0.0137, median test R2=-0.0088, positive test R2=4/20, beat train-mean baseline=4/20, macro test corr=0.0191, macro dir acc=0.5136.
v5 vs v4: better test R2=7/20 ['AAPL', 'IBM', 'LIN', 'NVDA', 'PG', 'SPY', 'XOM']; lower train R2 and better test R2=0/20 []; macro test R2 Δ=-0.0063.
Finished in 5.0s.
```

### v5 - same recipe, 10 features including Adj Close

- **Date:** 2026-09-01
- **Status:** superseded
- **Script:** `src/ml/regression/svm.py`
- **Git:** `795c896` (dirty; script untracked)

**What this version is:**
v4 lag, cutoff, RBF, and `C`/`gamma` grid with four added columns: `Adj Close`, `20_Day_Return`, `ATR_20`, `Dist_From_SMA`.

**What changed vs v4:**
- Feature set only: 6 cols → 10. Added `Adj Close`, `20_Day_Return`, `ATR_20`, `Dist_From_SMA`.
- Data, target, split, model, and grid are unchanged.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance` SPY period 25y (head 2001-09-04). Script loads `["SPY"]` only |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | SPY |
| Symbols trained | SPY |
| Features | `Adj Close`, `1_Day_Return`, `20_Day_Return`, `Overnight`, `Range`, `Signed_Rel_Vol`, `ATR_20`, `Dist_From_SMA`, `Upper_Wick`, `Lower_Wick` at lag `_0` (10 cols). Built by `ut.process_OHLCV_all`; split uses `BASE_FEATURE_COLS` |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `StandardScaler` + `sklearn.svm.SVR(kernel="rbf", epsilon=0.01, cache_size=500, max_iter=16000, tol=1e-3)` |
| Other | `GridSearchCV` scoring `neg_mean_squared_error` with `TimeSeriesSplit(n_splits=3, gap=5)` on train; `C` grid `{0.01, 0.1, 1}`; `gamma` grid `{0.01, 0.1, "scale"}`; best `C=0.01`, `gamma=0.01`; CV MSE 0.00016107; 27 fits in 2.2s (`n_jobs=-1`); 5764 train / 500 test rows; `n_support=1492`; several `C=1` folds hit `ConvergenceWarning` (`max_iter=16000`); `plot_data` time series + scatter; `DISPLAY_PLOTS=True`; unused derived columns still computed |

**Metrics**

| Metric | Train | Test |
|--------|-------|------|
| MSE | 0.00014030881114616 | 0.00011056694979128539 |
| R² | 0.032494068643990515 | -0.017194530258363372 |
| Best `C` | 0.01 | (same fit) |
| Best `gamma` | 0.01 | (same fit) |
| Intercept | 0.00094219 | (same fit) |
| `n_support` | 1492 | (same fit) |

**Read of the run:**
Same best params as v4 (`C=0.01`, `gamma=0.01`). Train n stayed 5764 because the 20-day windows were already implied by `Signed_Rel_Vol`. Test R² moved from −0.040 to −0.017; CV MSE ticked the other way (0.000160 → 0.000161). Intercept dropped to 0.00094. Same-day `Adj Close` is now a feature for a next-day return from that price (price level vs return target). Test is still worse than a constant mean.

**vs previous:**

| | Previous (v4) | This (v5) |
|--|-----------------|-----------|
| Change | 6 cols, no raw `Adj Close` | 10 cols; added `Adj Close`, `20_Day_Return`, `ATR_20`, `Dist_From_SMA` |
| Train MSE | 0.00014095507142133267 | 0.00014030881114616 (−0.00000065, better) |
| Train R² | 0.02803775086664162 | 0.032494068643990515 (+0.0045, better) |
| Test MSE | 0.00011308449547212016 | 0.00011056694979128539 (−0.00000252, better) |
| Test R² | -0.04035546307829496 | -0.017194530258363372 (+0.023, better) |
| `n_support` | 1504 | 1492 |
| CV MSE | 0.000160461 | 0.00016107 |
| Best params | `C=0.01`, `gamma=0.01` | `C=0.01`, `gamma=0.01` |

Same data, cutoff, and grid. Small holdout lift; CV did not agree. Still below a constant mean and in the same range as linear v8 / tree v2.

**Terminal excerpt**

```
Best params: {'svr__C': 0.01, 'svr__gamma': 0.01}  |  CV MSE: 0.00016107
n_support: 1492
Best params: {'svr__C': 0.01, 'svr__gamma': 0.01}
Intercept: [0.00094219]
MSE: 0.00014030881114616
R^2: 0.032494068643990515
Best params: {'svr__C': 0.01, 'svr__gamma': 0.01}
Intercept: [0.00094219]
MSE: 0.00011056694979128539
R^2: -0.017194530258363372
```

### v4 - C+gamma grid, epsilon 0.01

- **Date:** 2026-09-01
- **Status:** superseded
- **Script:** `src/ml/regression/svm.py`
- **Git:** `795c896` (dirty; script untracked)

**What this version is:**
v3 columns, lag, cutoff, and RBF, with a smaller `C` grid, `gamma` in the search, `epsilon=0.01`, and `max_iter` back to 16000.

**What changed vs v3:**
- `C` grid `{1, 10}` → `{0.01, 0.1, 1}`
- `gamma` now searched `{0.01, 0.1, "scale"}` (was fixed `"scale"`)
- `epsilon` 0.001 → 0.01
- `max_iter` 128000 → 16000; `cache_size` 2000 → 500
- Features, target, split, and SPY cache are unchanged

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance` SPY period 25y (head 2001-09-04). Script loads `["SPY"]` only |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | SPY |
| Symbols trained | SPY |
| Features | `1_Day_Return`, `Overnight`, `Range`, `Signed_Rel_Vol`, `Upper_Wick`, `Lower_Wick` at lag `_0` (6 cols). Built by `ut.process_OHLCV_all`; split uses `BASE_FEATURE_COLS` |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `StandardScaler` + `sklearn.svm.SVR(kernel="rbf", epsilon=0.01, cache_size=500, max_iter=16000, tol=1e-3)` |
| Other | `GridSearchCV` scoring `neg_mean_squared_error` with `TimeSeriesSplit(n_splits=3, gap=5)` on train; `C` grid `{0.01, 0.1, 1}`; `gamma` grid `{0.01, 0.1, "scale"}`; best `C=0.01`, `gamma=0.01`; CV MSE 0.000160461; 27 fits in 2.2s (`n_jobs=-1`); 5764 train / 500 test rows; `n_support=1504`; one fold (`C=1`, `gamma=0.1`, CV 3/3) hit `ConvergenceWarning` (`max_iter=16000`); `plot_data` time series + scatter; `DISPLAY_PLOTS=True`; unused derived columns still computed |

**Metrics**

| Metric | Train | Test |
|--------|-------|------|
| MSE | 0.00014095507142133267 | 0.00011308449547212016 |
| R² | 0.02803775086664162 | -0.04035546307829496 |
| Best `C` | 0.01 | (same fit) |
| Best `gamma` | 0.01 | (same fit) |
| Intercept | 0.00248172 | (same fit) |
| `n_support` | 1504 | (same fit) |

**Read of the run:**
CV picked the smallest `C` and `gamma` in the grid. Train R² fell from 0.340 to 0.028; test R² moved from −0.665 to −0.040. Support vectors dropped from 5095 to 1504. The surface is flatter (wider epsilon tube, small C, small gamma), so holdout is closer to a mean predictor and no longer the large negative of the `C=1` fits. Test is still worse than a constant mean. Intercept 0.0025 is smaller than v3 but still large versus daily SPY returns. Same-day features for a next-day return from `Adj Close`.

**vs previous:**

| | Previous (v3) | This (v4) |
|--|-----------------|-----------|
| Change | `C` in `{1, 10}`, `gamma="scale"`, `epsilon=0.001`, `max_iter=128000` | `C` in `{0.01, 0.1, 1}`, `gamma` in `{0.01, 0.1, "scale"}`, `epsilon=0.01`, `max_iter=16000` |
| Train MSE | 9.569099097714842e-05 | 0.00014095507142133267 (+0.0000453, worse) |
| Train R² | 0.3401583222647149 | 0.02803775086664162 (−0.312, worse) |
| Test MSE | 0.00018103422120214853 | 0.00011308449547212016 (−0.0000679, better) |
| Test R² | -0.6654797834616775 | -0.04035546307829496 (+0.625, better) |
| `n_support` | 5095 | 1504 |
| CV MSE | 0.000238857 | 0.000160461 |
| Best params | `C=1` | `C=0.01`, `gamma=0.01` |

Same data, features, and cutoff. The holdout lift is less overfit (smallest C/gamma plus a wider tube), not a new signal. Train got worse as expected. Test is still below a constant mean and in the same range as linear v8 and tree v2 on this window.

**Terminal excerpt**

```
Best params: {'svr__C': 0.01, 'svr__gamma': 0.01}  |  CV MSE: 0.000160461
n_support: 1504
Best params: {'svr__C': 0.01, 'svr__gamma': 0.01}
Intercept: [0.00248172]
MSE: 0.00014095507142133267
R^2: 0.02803775086664162
Best params: {'svr__C': 0.01, 'svr__gamma': 0.01}
Intercept: [0.00248172]
MSE: 0.00011308449547212016
R^2: -0.04035546307829496
```

### v3 - same recipe, max_iter 128000

- **Date:** 2026-09-01
- **Status:** superseded
- **Script:** `src/ml/regression/svm.py`
- **Git:** `795c896` (dirty; script untracked)

**What this version is:**
v2 columns, lag, cutoff, RBF, and `C` grid with SMO allowed 8× more iterations (`max_iter` 16000 → 128000).

**What changed vs v2:**
- Training procedure only: `max_iter` 16000 → 128000. Features, target, split, cache, and grid are unchanged.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance` SPY period 25y (head 2001-09-04). Script loads `["SPY"]` only |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | SPY |
| Symbols trained | SPY |
| Features | `1_Day_Return`, `Overnight`, `Range`, `Signed_Rel_Vol`, `Upper_Wick`, `Lower_Wick` at lag `_0` (6 cols). Built by `ut.process_OHLCV_all`; split uses `BASE_FEATURE_COLS` |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `StandardScaler` + `sklearn.svm.SVR(kernel="rbf", epsilon=0.001, gamma="scale", cache_size=2000, max_iter=128000, tol=1e-3)` |
| Other | `GridSearchCV` scoring `neg_mean_squared_error` with `TimeSeriesSplit(n_splits=3, gap=5)` on train; `C` grid `{1, 10}`; best `C=1`; CV MSE 0.000238857; 6 fits in 6.9s (`n_jobs=-1`); 5764 train / 500 test rows; `n_support=5095`; `C=10` folds hit `ConvergenceWarning` (`max_iter=128000`); `plot_data` time series + scatter; `DISPLAY_PLOTS=True`; unused derived columns still computed |

**Metrics**

| Metric | Train | Test |
|--------|-------|------|
| MSE | 9.569099097714842e-05 | 0.00018103422120214853 |
| R² | 0.3401583222647149 | -0.6654797834616775 |
| Best `C` | 1 | (same fit) |
| Intercept | 0.00784427 | (same fit) |
| `n_support` | 5095 | (same fit) |

**Read of the run:**
More SMO steps added support vectors (5095 / 5764 train rows) and a small train R² lift (0.340 vs 0.328). Test R² moved the other way (−0.665 vs −0.652). Intercept 0.0078 is still large versus daily SPY returns. `C=10` still hit `max_iter`. A same-session `max_iter=64000` run printed the same `n_support`, intercept, and train/test metrics, so the extra budget did not change the `C=1` refit. Same-day features for a next-day return from `Adj Close`.

**vs previous:**

| | Previous (v2) | This (v3) |
|--|-----------------|-----------|
| Change | `max_iter=16000` | `max_iter=128000` |
| Train MSE | 9.745715880095838e-05 | 9.569099097714842e-05 (−0.00000177, better) |
| Train R² | 0.32797962991212803 | 0.3401583222647149 (+0.012, better) |
| Test MSE | 0.00017961585739273878 | 0.00018103422120214853 (+0.00000142, worse) |
| Test R² | -0.6524311110367667 | -0.6654797834616775 (−0.013, worse) |
| `n_support` | 4614 | 5095 |
| CV MSE | 0.000238434 | 0.000238857 |

Same data, features, cutoff, and `C=1`. Extra SMO bought a slightly tighter train fit and more SVs, not a better holdout. Test is still worse than a constant mean.

**Terminal excerpt**

```
Best params: {'svr__C': 1}  |  CV MSE: 0.000238857
n_support: 5095
Best params: {'svr__C': 1}
Intercept: [0.00784427]
MSE: 9.569099097714842e-05
R^2: 0.3401583222647149
Best params: {'svr__C': 1}
Intercept: [0.00784427]
MSE: 0.00018103422120214853
R^2: -0.6654797834616775
```

### v2 - same recipe, max_iter 16000

- **Date:** 2026-09-01
- **Status:** superseded
- **Script:** `src/ml/regression/svm.py`
- **Git:** `795c896` (dirty; script untracked)

**What this version is:**
v1 columns, lag, cutoff, RBF, and `C` grid with SMO allowed 8× more iterations (`max_iter` 2000 → 16000). Kernel cache set to 2000 MB.

**What changed vs v1:**
- Training procedure only: `max_iter` 2000 → 16000, `cache_size` 500 → 2000. Features, target, split, and grid are unchanged.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance` SPY period 25y (head 2001-09-04). Script loads `["SPY"]` only |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | SPY |
| Symbols trained | SPY |
| Features | `1_Day_Return`, `Overnight`, `Range`, `Signed_Rel_Vol`, `Upper_Wick`, `Lower_Wick` at lag `_0` (6 cols). Built by `ut.process_OHLCV_all`; split uses `BASE_FEATURE_COLS` |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `StandardScaler` + `sklearn.svm.SVR(kernel="rbf", epsilon=0.001, gamma="scale", cache_size=2000, max_iter=16000, tol=1e-3)` |
| Other | `GridSearchCV` scoring `neg_mean_squared_error` with `TimeSeriesSplit(n_splits=3, gap=5)` on train; `C` grid `{1, 10}`; best `C=1`; CV MSE 0.000238434; 6 fits in 3.1s (`n_jobs=-1`); 5764 train / 500 test rows; `n_support=4614`; every CV fold and the refit hit `ConvergenceWarning` (`max_iter=16000`); `plot_data` time series + scatter; `DISPLAY_PLOTS=True`; unused derived columns still computed |

**Metrics**

| Metric | Train | Test |
|--------|-------|------|
| MSE | 9.745715880095838e-05 | 0.00017961585739273878 |
| R² | 0.32797962991212803 | -0.6524311110367667 |
| Best `C` | 1 | (same fit) |
| Intercept | 0.01040324 | (same fit) |
| `n_support` | 4614 | (same fit) |

**Read of the run:**
More SMO steps let the model fit the training set (train R² 0.33 vs v1 −1.48). It still did not converge. 4614 / 5764 train rows are support vectors, so the RBF surface is using most days, not a small set of prototypes. Test R² is still below 0: the extra fit did not carry to the holdout. Intercept 0.010 is large versus daily SPY returns. Same-day features for a next-day return from `Adj Close`. The script on disk may already have a higher `max_iter` than this run; this version is the `max_iter=16000` terminal paste.

**vs previous:**

| | Previous (v1) | This (v2) |
|--|-----------------|-----------|
| Change | `max_iter=2000`, cache 500 MB | `max_iter=16000`, cache 2000 MB |
| Train MSE | 0.000359013129779199 | 9.745715880095838e-05 (−0.000262, better) |
| Train R² | -1.4755917298324719 | 0.32797962991212803 (+1.80, better) |
| Test MSE | 0.0004107968303698627 | 0.00017961585739273878 (−0.000231, better) |
| Test R² | -2.7792513014827853 | -0.6524311110367667 (+2.13, better) |
| `n_support` | 1686 | 4614 |
| CV MSE | 0.000429523 | 0.000238434 |

Same data, features, cutoff, and `C=1`. The lift is more complete SMO, not a new signal. Train improved a lot more than test; test is still worse than a constant mean. v1's terrible scores were a truncated solver, not a fair read of RBF SVR.

**Terminal excerpt**

```
Best params: {'svr__C': 1}  |  CV MSE: 0.000238434
n_support: 4614
Best params: {'svr__C': 1}
Intercept: [0.01040324]
MSE: 9.745715880095838e-05
R^2: 0.32797962991212803
Best params: {'svr__C': 1}
Intercept: [0.01040324]
MSE: 0.00017961585739273878
R^2: -0.6524311110367667
```

### v1 - RBF SVR baseline, SPY only, max_iter 2000

- **Date:** 2026-09-01
- **Status:** superseded
- **Script:** `src/ml/regression/svm.py`
- **Git:** `795c896` (dirty; script untracked)

**What this version is:**
First logged kernel SVR: six lag-0 columns, `StandardScaler`, RBF, `GridSearchCV` over `C` in `{1, 10}`.

**What changed vs vN-1:**
First logged run.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance` SPY period 25y (head 2001-09-04). Script loads `["SPY"]` only |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | SPY |
| Symbols trained | SPY |
| Features | `1_Day_Return`, `Overnight`, `Range`, `Signed_Rel_Vol`, `Upper_Wick`, `Lower_Wick` at lag `_0` (6 cols). Built by `ut.process_OHLCV_all`; split uses `BASE_FEATURE_COLS` |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `StandardScaler` + `sklearn.svm.SVR(kernel="rbf", epsilon=0.001, gamma="scale", cache_size=500, max_iter=2000, tol=1e-3)` |
| Other | `GridSearchCV` scoring `neg_mean_squared_error` with `TimeSeriesSplit(n_splits=3, gap=5)` on train; `C` grid `{1, 10}`; best `C=1`; CV MSE 0.000429523; 6 fits in 1.6s (`n_jobs=-1`); 5764 train / 500 test rows; `n_support=1686`; every CV fold and the refit hit `ConvergenceWarning` (`max_iter=2000`); `plot_data` time series + scatter; `DISPLAY_PLOTS=True`; unused derived columns still computed |

**Metrics**

| Metric | Train | Test |
|--------|-------|------|
| MSE | 0.000359013129779199 | 0.0004107968303698627 |
| R² | -1.4755917298324719 | -2.7792513014827853 |
| Best `C` | 1 | (same fit) |
| Intercept | 0.0041148 | (same fit) |
| `n_support` | 1686 | (same fit) |

**Read of the run:**
Usable baseline numbers, but the fit is not a finished SMO: `max_iter=2000` stopped every fold. Train and test R² are far below 0, so the model is worse than predicting the mean. Intercept 0.004 is large versus daily SPY returns. Same-day features for a next-day return from `Adj Close`. `Signed_Rel_Vol` needs a 20-day volume window, which is why train n (5764) is a bit below the 25y tree run (5783) on one fewer column. The sklearn warning about scaling is a false lead; the pipeline already has `StandardScaler`.

**vs previous:**
First version.

**Terminal excerpt**

```
Best params: {'svr__C': 1}  |  CV MSE: 0.000429523
n_support: 1686
Best params: {'svr__C': 1}
Intercept: [0.0041148]
MSE: 0.000359013129779199
R^2: -1.4755917298324719
Best params: {'svr__C': 1}
Intercept: [0.0041148]
MSE: 0.0004107968303698627
R^2: -2.7792513014827853
```
