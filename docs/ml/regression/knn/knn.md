# kNN regression

**Script:** `src/ml/regression/ai_scripts/ai-knn-regression.py`
**Family:** neighbors
**Current:** v8 (`current`)

Per-ticker scaled `KNeighborsRegressor`. v8 fits frozen v4/v5 lag-1 columns at k=15, 21, and 51 on all 20 tickers (no subset search, no `GridSearchCV`).

Shared across versions: `data/clean-yfinance` 10y, cutoff `2024-09-01`, next-day `Adj Close` return. kNN has no coefficients. Same-day derived columns are features for that return. Overnight is split-adjusted open vs prior `Adj Close`. SPY's post-2024 window reused since v1; other tickers first appear in v6/v8.

## Versions

### v8 - frozen v4/v5 features, all 20 tickers, k=15/21 vs 51

- **Date:** 2026-09-01
- **Status:** current
- **Script:** `src/ml/regression/ai_scripts/ai-knn-regression.py`
- **Git:** `d352ecd` (dirty)

**What changed vs v7:**
- Universe: SPY only → all 20 tickers in `data/clean-yfinance`.
- k set `{3, 5, 7, 11, 15, 21}` → fixed fits at `{15, 21, 51}`. 51 is back as the v6 CV ceiling. 3/5/7/11 are dropped.
- No `GridSearchCV`. Each k is fit once on that ticker's train.
- Data, cutoff, target, scaler, lag, and feature lists are unchanged.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`, period 10y |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all 20 CSVs in `data/clean-yfinance` |
| Symbols trained | AAPL, AMZN, CAT, GOOG, IBM, JNJ, JPM, KO, LIN, META, MSFT, NEE, NVDA, PG, PLD, SPY, UNH, UNP, V, XOM (one model each, each k) |
| Features | two frozen lag `_0` sets from SPY v4/v5. v4: `1_Day_Return`, `Overnight`, `Range`, `Close Location`, `Upper Wick`, `Lower Wick`. v5: `1_Day_Return`, `Overnight`, `Range`, `Close Location`, `Rel_Vol`, `20_Day_Return`, `Dist_From_SMA` |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `StandardScaler` + `sklearn.neighbors.KNeighborsRegressor(n_neighbors=k)` |
| Other | no subset search; no CV; ~1992 train / ~499 test rows per ticker |

**Metrics**

Universe (unweighted mean of per-ticker metrics):

| Recipe | k | Macro train R² | Macro test R² | Median test R² | Test R² > 0 | Beat train-mean |
|--------|---|----------------|---------------|----------------|-------------|-----------------|
| v4 | 15 | 0.0836 | -0.0500 | -0.0525 | 1/20 | 1/20 |
| v4 | 21 | 0.0621 | -0.0367 | -0.0398 | 1/20 | 1/20 |
| v4 | 51 | 0.0278 | -0.0127 | -0.0140 | 7/20 | 7/20 |
| v5 | 15 | 0.0869 | -0.0515 | -0.0529 | 1/20 | 1/20 |
| v5 | 21 | 0.0637 | -0.0325 | -0.0336 | 2/20 | 2/20 |
| v5 | 51 | 0.0286 | -0.0111 | -0.0099 | 6/20 | 6/20 |

k=15 vs k=51 test R²: v4 better on 1/20 (SPY); v5 better on 1/20 (SPY). k=21 vs k=51: v4 better on 0/20; v5 better on 3/20 (CAT, NVDA, SPY). Train R² is higher at 15 and 21 than at 51 on every ticker, so the "lower train, better test" pattern vs k=51 is 0/20.

v4 per-ticker test R² (best k among 15/21/51):

| Symbol | k=15 | k=21 | k=51 | Best k |
|--------|------|------|------|--------|
| AAPL | -0.0930 | -0.0421 | 0.0006 | 51 |
| AMZN | -0.0758 | -0.0628 | -0.0383 | 51 |
| CAT | -0.0850 | -0.0529 | -0.0142 | 51 |
| GOOG | -0.0581 | -0.0480 | -0.0173 | 51 |
| IBM | -0.0213 | -0.0111 | 0.0053 | 51 |
| JNJ | -0.0473 | -0.0196 | 0.0018 | 51 |
| JPM | -0.0334 | -0.0180 | 0.0013 | 51 |
| KO | -0.0300 | -0.0194 | -0.0078 | 51 |
| LIN | -0.0708 | -0.0599 | -0.0338 | 51 |
| META | -0.0785 | -0.0559 | -0.0290 | 51 |
| MSFT | -0.0594 | -0.0624 | -0.0288 | 51 |
| NEE | -0.0386 | -0.0374 | -0.0236 | 51 |
| NVDA | -0.0252 | -0.0189 | 0.0098 | 51 |
| PG | -0.0615 | -0.0585 | -0.0232 | 51 |
| PLD | -0.0578 | -0.0285 | -0.0138 | 51 |
| SPY | 0.0106 | 0.0086 | 0.0095 | 15 |
| UNH | -0.0426 | -0.0451 | -0.0271 | 51 |
| UNP | -0.0312 | -0.0291 | -0.0066 | 51 |
| V | -0.0232 | -0.0183 | 0.0076 | 51 |
| XOM | -0.0786 | -0.0555 | -0.0264 | 51 |

v5 per-ticker test R²:

| Symbol | k=15 | k=21 | k=51 | Best k |
|--------|------|------|------|--------|
| AAPL | -0.1163 | -0.0814 | -0.0266 | 51 |
| AMZN | -0.1125 | -0.0844 | -0.0475 | 51 |
| CAT | -0.0234 | -0.0136 | -0.0225 | 21 |
| GOOG | -0.0572 | -0.0296 | 0.0059 | 51 |
| IBM | -0.0476 | -0.0353 | -0.0183 | 51 |
| JNJ | -0.0562 | -0.0385 | -0.0110 | 51 |
| JPM | -0.0476 | -0.0280 | -0.0014 | 51 |
| KO | -0.0265 | -0.0028 | -0.0001 | 51 |
| LIN | -0.0748 | -0.0385 | -0.0157 | 51 |
| META | -0.0693 | -0.0378 | 0.0080 | 51 |
| MSFT | -0.1021 | -0.0541 | -0.0265 | 51 |
| NEE | -0.0534 | -0.0505 | -0.0323 | 51 |
| NVDA | -0.0147 | 0.0029 | -0.0012 | 21 |
| PG | -0.0216 | -0.0246 | 0.0047 | 51 |
| PLD | -0.0303 | -0.0080 | 0.0035 | 51 |
| SPY | 0.0341 | 0.0328 | 0.0100 | 15 |
| UNH | -0.0593 | -0.0377 | -0.0285 | 51 |
| UNP | -0.0430 | -0.0320 | -0.0088 | 51 |
| V | -0.0553 | -0.0292 | 0.0057 | 51 |
| XOM | -0.0525 | -0.0592 | -0.0195 | 51 |

SPY k=15/21 rows match v7. v4 k=51 rows match v6. v5 k=51 differs from v6 only on META: v6 CV chose k=21 (test R² -0.0378); forcing k=51 here gives test R² +0.0080, so v5 k=51 is 6/20 positive instead of v6's 5/20.

**Read of the run:**
Ranking k on this holdout peeks; SPY's window has been reused since v1, the other 19 names have not. Twenty separate models, not one pooled neighbor space.

**vs previous:**

| | Previous (v7) | This (v8) |
|--|-----------------|-----------|
| Change | SPY only, k in {3, 5, 7, 11, 15, 21} | all 20 tickers, k in {15, 21, 51} |
| SPY v5 k=15 test R² | 0.0341 | 0.0341 (same) |
| SPY v5 k=21 test R² | 0.0328 | 0.0328 (same) |
| Names where k=15 beats k=51 (v5 test R²) | SPY (only ticker) | 1/20 (SPY) |
| Names where k=21 beats k=51 (v5 test R²) | SPY (only ticker) | 3/20 (CAT, NVDA, SPY) |
| Macro test R² | n/a (one ticker) | v5 k=15 -0.0515; k=21 -0.0325; k=51 -0.0111 |

SPY is a v7 reproduction. On the other 19 names, smaller k usually hurts. k=51 is the least-bad of the three on average.

**Terminal excerpt**

```
v4 k=15 vs k=51: better test R2=1/20 ['SPY']; lower train R2 and better test R2=0/20 []; macro test R2 Δ=-0.0373.
v4 k=21 vs k=51: better test R2=0/20 []; lower train R2 and better test R2=0/20 []; macro test R2 Δ=-0.0241.
v5 k=15 vs k=51: better test R2=1/20 ['SPY']; lower train R2 and better test R2=0/20 []; macro test R2 Δ=-0.0404.
v5 k=21 vs k=51: better test R2=3/20 ['CAT', 'NVDA', 'SPY']; lower train R2 and better test R2=0/20 []; macro test R2 Δ=-0.0214.
SPY v5 k=15 test R²=0.0341; k=21=0.0328; k=51=0.0100
```

### v7 - frozen v4/v5 features, SPY, small-k sweep

- **Date:** 2026-09-01
- **Status:** superseded
- **Script:** `src/ml/regression/KNN.py` (one-off runner; `select_features` not called)
- **Git:** `d352ecd` (dirty)

**What changed vs v6:**
- Universe back to SPY only. No 20-ticker loop.
- k grid `{3, 5, 11, 21, 51}` → fixed fits at `{3, 5, 7, 11, 15, 21}`, then CV over that small grid. 51 is excluded.
- No 512-subset search. Feature lists are the v4 and v5 SPY winners.
- Data, cutoff, target, scaler, lag, and chronological split match v4–v6. Plots off.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`, period 10y |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all `STARTING_STOCKS` (20 names in `src/ml/config.py`) |
| Symbols trained | SPY |
| Features | two frozen lag `_0` sets. v4: `1_Day_Return`, `Overnight`, `Range`, `Close Location`, `Upper Wick`, `Lower Wick` (6 cols). v5: `1_Day_Return`, `Overnight`, `Range`, `Close Location`, `Rel_Vol`, `20_Day_Return`, `Dist_From_SMA` (7 cols) |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `StandardScaler` + `sklearn.neighbors.KNeighborsRegressor` |
| Other | no subset search; 1992 train / 499 test rows (split on the full candidate list, then column-subset, same as v4/v5); `DISPLAY_PLOTS` off |

**Metrics**

v4 recipe (bar shape), vs documented k=51 test R² `0.009511846611423236`:

| k | Train R² | Test MSE | Test R² |
|---|---------|----------|---------|
| 3 | 0.3814 | 0.000146175 | -0.3436 |
| 5 | 0.2438 | 0.000132769 | -0.2203 |
| 7 | 0.1775 | 0.000119414 | -0.0976 |
| 11 | 0.1294 | 0.000111828 | -0.0279 |
| 15 | 0.0959 | 0.000107648 | 0.0106 |
| 21 | 0.0696 | 0.000107865 | 0.0086 |
| 51 (v4 doc) | 0.0263 | 0.000107762 | 0.0095 |

v5 recipe (close location / rel vol / 20-day / SMA), vs documented k=51 test R² `0.01001269150711992`:

| k | Train R² | Test MSE | Test R² |
|---|---------|----------|---------|
| 3 | 0.3117 | 0.000130442 | -0.1989 |
| 5 | 0.2099 | 0.000118048 | -0.0850 |
| 7 | 0.1274 | 0.000114083 | -0.0486 |
| 11 | 0.0930 | 0.000108177 | 0.0057 |
| 15 | 0.0764 | 0.000105084 | 0.0341 |
| 21 | 0.0681 | 0.000105227 | 0.0328 |
| 51 (v5 doc) | 0.0362 | 0.000107708 | 0.0100 |

`GridSearchCV` on `{3, 5, 7, 11, 15, 21}` with `TimeSeriesSplit(n_splits=5, gap=5)` chose **k=21** for both recipes (grid ceiling). CV-chosen test R²: v4 `0.0086`; v5 `0.0328`.

**Read of the run:**
k=3 to k=11 overfit train and lose on test. v5 columns move at k=15/21 (test R² ~0.034 / 0.033 vs ~0.010 at k=51); v4 barely moves. CV still picks 21, not 15. Ranking k=15 by this reused SPY holdout peeks.

**vs previous:**

| | Previous (v6) | This (v7) |
|--|-----------------|-----------|
| Change | frozen v4/v5 sets, all 20 tickers, k grid to 51 | frozen v4/v5 sets, SPY only, k in {3, 5, 7, 11, 15, 21} |
| SPY v5 k=51 test R² | 0.0100 | n/a (51 not fit) |
| SPY v5 k=15 test R² | n/a | 0.0341 |
| SPY v5 k=21 test R² | n/a (META used 21 in v6; SPY used 51) | 0.0328 |
| SPY v4 k=15 test R² | n/a | 0.0106 |

On SPY, smaller-than-51 k helps only for the v5 columns, and only around 15–21. Ranking k=15 by test R² peeks at the holdout; CV still wants 21.

**Terminal excerpt**

```
=== v4 features (6 cols) ===
  k=15  train R²= 0.0959  test MSE=0.000107648  test R²= 0.0106  vs k=51 Δ=+0.00105
  k=21  train R²= 0.0696  test MSE=0.000107865  test R²= 0.0086  vs k=51 Δ=-0.00094
  CV best params: {'kneighborsregressor__n_neighbors': 21}
=== v5 features (7 cols) ===
  k=15  train R²= 0.0764  test MSE=0.000105084  test R²= 0.0341  vs k=51 Δ=+0.02412
  k=21  train R²= 0.0681  test MSE=0.000105227  test R²= 0.0328  vs k=51 Δ=+0.02280
  CV best params: {'kneighborsregressor__n_neighbors': 21}
```

### v6 - frozen v4/v5 features, all 20 tickers

- **Date:** 2026-09-01
- **Status:** superseded
- **Script:** `src/ml/regression/ai_scripts/ai-knn-regression.py`
- **Git:** `d352ecd` (dirty)

**What changed vs v5:**
- Universe: SPY only → all 20 tickers in `data/clean-yfinance`.
- Feature selection skipped. Both frozen recipes are trained: v4 bar-shape (6 cols) and v5 close-location / rel-vol / 20-day return / SMA-distance (7 cols).
- New script: `ai-knn-regression.py`. `KNN.py` is unchanged.
- Data, cutoff, target, scaler, k grid, lag, and chronological split are unchanged. Plots off.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`, period 10y |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all 20 CSVs in `data/clean-yfinance` |
| Symbols trained | AAPL, AMZN, CAT, GOOG, IBM, JNJ, JPM, KO, LIN, META, MSFT, NEE, NVDA, PG, PLD, SPY, UNH, UNP, V, XOM (one model each) |
| Features | two frozen lag `_0` sets from SPY v4/v5. v4: `1_Day_Return`, `Overnight`, `Range`, `Close Location`, `Upper Wick`, `Lower Wick`. v5: `1_Day_Return`, `Overnight`, `Range`, `Close Location`, `Rel_Vol`, `20_Day_Return`, `Dist_From_SMA` |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `StandardScaler` + `sklearn.neighbors.KNeighborsRegressor` |
| Other | no subset search; `GridSearchCV` scoring `neg_mean_squared_error` with `TimeSeriesSplit(n_splits=5, gap=5)` on each ticker's train; k grid `{3, 5, 11, 21, 51}`; ~1992 train / ~499 test rows per ticker; `DISPLAY_PLOTS` not used |

**Metrics**

Universe (unweighted mean of per-ticker metrics):

| | v4 recipe | v5 recipe |
|--|-----------|-----------|
| Macro train MSE | 0.000315166 | 0.000313540 |
| Macro train R² | 0.0278 | 0.0309 |
| Macro test MSE | 0.000343964 | 0.000345408 |
| Macro test R² | -0.0127 | -0.0134 |
| Median test R² | -0.0140 | -0.0133 |
| Test R² > 0 | 7/20 | 5/20 |
| Beat train-mean baseline (test MSE) | 7/20 | 5/20 |
| Macro test correlation | 0.0190 | 0.0181 |
| Macro directional accuracy | 0.5120 | 0.4984 |
| Best `n_neighbors` | 51 on all 20 | 51 on 19; META=21 |

Per-ticker holdout:

| Symbol | v4 test MSE | v4 test R² | v5 test MSE | v5 test R² |
|--------|-------------|------------|-------------|------------|
| AAPL | 0.000328338 | 0.000583 | 0.000337254 | -0.026556 |
| AMZN | 0.000481072 | -0.038251 | 0.000485375 | -0.047537 |
| CAT | 0.000498640 | -0.014218 | 0.000502705 | -0.022485 |
| GOOG | 0.000398392 | -0.017294 | 0.000389293 | 0.005940 |
| IBM | 0.000632085 | 0.005313 | 0.000647090 | -0.018299 |
| JNJ | 0.000143798 | 0.001793 | 0.000145637 | -0.010975 |
| JPM | 0.000248964 | 0.001270 | 0.000249628 | -0.001394 |
| KO | 0.000129405 | -0.007800 | 0.000128417 | -0.000106 |
| LIN | 0.000144718 | -0.033781 | 0.000142182 | -0.015666 |
| META | 0.000583052 | -0.029038 | 0.000588018 | -0.037802 |
| MSFT | 0.000339454 | -0.028786 | 0.000338711 | -0.026533 |
| NEE | 0.000259630 | -0.023640 | 0.000261829 | -0.032308 |
| NVDA | 0.000783654 | 0.009825 | 0.000792385 | -0.001207 |
| PG | 0.000147781 | -0.023200 | 0.000143745 | 0.004749 |
| PLD | 0.000264659 | -0.013769 | 0.000260161 | 0.003462 |
| SPY | 0.000107762 | 0.009512 | 0.000107708 | 0.010013 |
| UNH | 0.000726733 | -0.027101 | 0.000727717 | -0.028491 |
| UNP | 0.000213519 | -0.006554 | 0.000214004 | -0.008837 |
| V | 0.000196923 | 0.007612 | 0.000197297 | 0.005728 |
| XOM | 0.000250689 | -0.026376 | 0.000249006 | -0.019486 |

v4 positive test R²: AAPL, IBM, JNJ, JPM, NVDA, SPY, V. v5: GOOG, PG, PLD, SPY, V. Overlap: SPY, V. Worst on both recipes: AMZN. Best v4: NVDA. Best v5: SPY.

SPY rows match v4 and v5 exactly (same recipe, same split).

**Read of the run:**
SPY-selected columns do not transfer as a positive-R² rule. Macro and median test R² are negative. Feature lists were chosen on SPY train, then applied to the other 19 names (not re-selected). Directional accuracy sits near 0.5. Twenty separate models, not one pooled neighbor space.

**vs previous:**

| | Previous (v5) | This (v6) |
|--|-----------------|-----------|
| Change | exhaustive 512-subset search, SPY only, 7 cols | frozen v4 and v5 sets, all 20 tickers, no search |
| Symbols trained | SPY | 20 |
| SPY test MSE (v5 recipe) | 0.00010770750570766955 | 0.00010770750570766955 (same) |
| SPY test R² (v5 recipe) | 0.01001269150711992 | 0.01001269150711992 (same) |
| Macro test R² | n/a (one ticker) | v4 -0.0127; v5 -0.0134 |

SPY is a reproduction. The new information is the other 19 names: v4 beat a mean predictor on 7/20, v5 on 5/20.

**Terminal excerpt**

```
v4 summary: macro train R2=0.0278, macro test R2=-0.0127, median test R2=-0.0140, positive test R2=7/20, beat train-mean baseline=7/20, k counts={'51': 20}.
v5 summary: macro train R2=0.0309, macro test R2=-0.0134, median test R2=-0.0133, positive test R2=5/20, beat train-mean baseline=5/20, k counts={'21': 1, '51': 19}.
SPY v4 | n_neighbors=51 | train MSE=0.000129941 R²=0.0262818 | test MSE=0.000107762 R²=0.00951185
SPY v5 | n_neighbors=51 | train MSE=0.000128612 R²=0.0362391 | test MSE=0.000107708 R²=0.0100127
```

### v5 - exhaustive 512-subset search on train, lag=1, SPY only

- **Date:** 2026-09-01
- **Status:** superseded
- **Script:** `src/ml/regression/KNN.py`
- **Git:** `d352ecd` (dirty)

**What changed vs v4:**
- Feature selection: all `2^9 = 512` subsets of `CANDIDATE_GROUPS` (flattened) plus the three-column base. Still scored on `X_train` only by nested `TimeSeriesSplit` + `GridSearchCV` fold MSE. Ties keep the smaller set (empty-to-full order).
- Trained columns: dropped `Upper Wick` and `Lower Wick`; added `Rel_Vol`, `20_Day_Return`, `Dist_From_SMA`. Still lag `_0` only (7 cols).
- Data, cutoff, target, scaler, k grid, split-then-subset, and SPY-only train are unchanged.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`, period 10y |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all `STARTING_STOCKS` (20 names in `src/ml/config.py`) |
| Symbols trained | SPY |
| Features | selected `1_Day_Return`, `Overnight`, `Range`, `Close Location`, `Rel_Vol`, `20_Day_Return`, `Dist_From_SMA` at lag `_0` (7 cols). Winner is trial 176/512: candidates `Close Location`, `Rel_Vol`, `20_Day_Return`, `Dist_From_SMA` on top of the base |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `StandardScaler` + `sklearn.neighbors.KNeighborsRegressor` |
| Other | selection and final fit both use `GridSearchCV` scoring `neg_mean_squared_error` with `TimeSeriesSplit(n_splits=5, gap=5)` on train; best `n_neighbors=51` (grid ceiling); `plot_data` 2x2 (time series, scatter, return histogram, residuals); `DISPLAY_PLOTS=True` |

**Metrics**

Selection fold means (train only):

| Trial | Fold MSE | Fold R² |
|-------|----------|---------|
| 176/512 `Close Location`,`Rel_Vol`,`20_Day_Return`,`Dist_From_SMA` (kept) | 0.000156663 | -0.0115 |

Full 512-line print not stored. Later trials in the terminal tail (432–512) had higher fold MSE than 176.

| Metric | Train (final fit) | Test |
|--------|-------------------|------|
| MSE | 0.00012861232036723516 | 0.00010770750570766955 |
| R² | 0.03623906350661876 | 0.01001269150711992 |
| Best `n_neighbors` | 51 | (same fit) |

**Read of the run:**
Exhaustive search kept a mix greedy v4 never assembled: close location, rel vol, 20-day return, SMA distance, no wicks. Winner fold R² is still negative. Searching 512 subsets on the same folds can pick a lucky CV winner.

**vs previous:**

| | Previous (v4) | This (v5) |
|--|-----------------|-----------|
| Change | greedy per-group subsets; kept base plus full bar shape (6 cols) | exhaustive 512 subsets; kept base plus close location, rel vol, 20-day return, SMA distance (7 cols) |
| Train MSE | 0.00012994110739807677 | 0.00012861232036723516 (-0.00000133, better) |
| Train R² | 0.02628175125544807 | 0.03623906350661876 (+0.00996, better) |
| Test MSE | 0.00010776199605719133 | 0.00010770750570766955 (-0.00000005, better) |
| Test R² | 0.009511846611423236 | 0.01001269150711992 (+0.00050, better) |

Train improved. Holdout is essentially flat vs v4. Winner fold R² is still negative, so the small positive test R² is not what CV predicted.

**Terminal excerpt**

```
176/512 | Close Location,Rel_Vol,20_Day_Return,Dist_From_SMA           | MSE=0.000156663  R²=-0.0115
Selected features: ['1_Day_Return', 'Overnight', 'Range', 'Close Location', 'Rel_Vol', '20_Day_Return', 'Dist_From_SMA']
Best params: {'kneighborsregressor__n_neighbors': 51}
MSE: 0.00012861232036723516
R^2: 0.03623906350661876
Best params: {'kneighborsregressor__n_neighbors': 51}
MSE: 0.00010770750570766955
R^2: 0.01001269150711992
```

### v4 - greedy subset search on train, lag=1, SPY only

- **Date:** 2026-09-01
- **Status:** superseded
- **Script:** `src/ml/regression/KNN.py`
- **Git:** `d352ecd` (dirty)

**What changed vs v3:**
- Feature selection: `select_features` on `X_train` only. Base is `1_Day_Return`, `Overnight`, `Range`. Each `CANDIDATE_GROUPS` non-empty subset is scored by mean fold MSE from nested `TimeSeriesSplit` + `GridSearchCV`. A subset is kept only if it beats the running fold MSE.
- Trained columns: v3's `Rel_Vol` and `Dist_From_SMA` dropped; `Close Location`, `Upper Wick`, `Lower Wick` added. Still lag `_0` only (6 cols).
- `split_data` is built from base plus all candidate names first, then column-subset after selection. Row drop still follows the longest rolling windows (volume and SMA), including unused candidates.
- Data, cutoff, target, scaler, k grid, and SPY-only train are unchanged.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`, period 10y |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all `STARTING_STOCKS` (20 names in `src/ml/config.py`) |
| Symbols trained | SPY |
| Features | selected `1_Day_Return`, `Overnight`, `Range`, `Close Location`, `Upper Wick`, `Lower Wick` at lag `_0` (6 cols). Candidates also included volume, 5/10/20-day returns, and `Dist_From_SMA`; none of those beat bar-shape fold MSE |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `StandardScaler` + `sklearn.neighbors.KNeighborsRegressor` |
| Other | selection and final fit both use `GridSearchCV` scoring `neg_mean_squared_error` with `TimeSeriesSplit(n_splits=5, gap=5)` on train; best `n_neighbors=51` (grid ceiling); `plot_data` 2x2 (time series, scatter, return histogram, residuals); `DISPLAY_PLOTS=True` |

**Metrics**

Selection fold means (train only):

| Trial | Fold MSE | Fold R² |
|-------|----------|---------|
| baseline (`1_Day_Return`+`Overnight`+`Range`) | 0.000157893 | -0.0197 |
| bar_shape all three (kept) | 0.000156921 | -0.0107 |
| volume `Shock_Vol` (best unused) | 0.000157152 | -0.0145 |

| Metric | Train (final fit) | Test |
|--------|-------------------|------|
| MSE | 0.00012994110739807677 | 0.00010776199605719133 |
| R² | 0.02628175125544807 | 0.009511846611423236 |
| Best `n_neighbors` | 51 | (same fit) |

**Read of the run:**
Greedy search kept the full bar-shape group and rejected volume, multi-horizon returns, and `Dist_From_SMA`. Every selection fold R² was negative, including the winner. First kNN version here with test R² > 0; CV did not predict that sign flip.

**vs previous:**

| | Previous (v3) | This (v4) |
|--|-----------------|-----------|
| Change | hand-picked lag-1: 1-day return, overnight, range, rel vol, SMA distance (5 cols) | greedy subset search on train; kept base plus bar shape (6 cols) |
| Train MSE | 0.00012913171980701173 | 0.00012994110739807677 (+0.00000081, worse) |
| Train R² | 0.03234692557563468 | 0.02628175125544807 (-0.00607, worse) |
| Test MSE | 0.00010896107404856351 | 0.00010776199605719133 (-0.00000120, better) |
| Test R² | -0.0015094093869603142 | 0.009511846611423236 (+0.01102, better) |

Test improved and is slightly above a mean predictor. Train got a bit worse. Selection folds still had negative R², so the holdout sign flip is not what CV predicted.

**Terminal excerpt**

```
Selected features: ['1_Day_Return', 'Overnight', 'Range', 'Close Location', 'Upper Wick', 'Lower Wick']
Best params: {'kneighborsregressor__n_neighbors': 51}
MSE: 0.00012994110739807677
R^2: 0.02628175125544807
Best params: {'kneighborsregressor__n_neighbors': 51}
MSE: 0.00010776199605719133
R^2: 0.009511846611423236
```

### v3 - thin 5-feature set, lag=1, SPY only

- **Date:** 2026-09-01
- **Status:** superseded
- **Script:** `src/ml/regression/KNN.py`
- **Git:** `d352ecd` (dirty)

**What changed vs v2:**
- Feature set: 11 derived columns lagged `_0`..`_4` (55 cols) → `1_Day_Return`, `Overnight`, `Range`, `Rel_Vol`, `Dist_From_SMA` at `_0` only (5 cols).
- `LAG` 5 → 1. History is encoded once in `1_Day_Return` and `Dist_From_SMA`, not by lagging rolling stats.
- Added `1_Day_Return`. Dropped bar-shape extras (`Close Location`, wicks), `Shock_Vol`, and 5/10/20-day returns from the trained matrix.
- `process_OHLCV_all` still computes the unused derived columns; `split_data` uses `BASE_FEATURE_COLS`. Volume column renamed `Rel Vol` → `Rel_Vol`.
- Data, cutoff, target, scaler, k grid, and SPY-only train are unchanged.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`, period 10y |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all `STARTING_STOCKS` (20 names in `src/ml/config.py`) |
| Symbols trained | SPY |
| Features | `1_Day_Return`, `Overnight`, `Range`, `Rel_Vol`, `Dist_From_SMA` at lag `_0` (5 cols). Built by `ut.process_OHLCV_all`; split uses `BASE_FEATURE_COLS` |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `StandardScaler` + `sklearn.neighbors.KNeighborsRegressor` |
| Other | `GridSearchCV` scoring `neg_mean_squared_error`; CV `TimeSeriesSplit(n_splits=5, gap=5)` on train; best `n_neighbors=51` (grid ceiling); `plot_data` 2x2 (time series, scatter, return histogram, residuals; recent-return scatter skipped because `Adj Close_0`/`_1` are absent); `DISPLAY_PLOTS=True` |

**Metrics**

| Metric | Train | Test |
|--------|-------|------|
| MSE | 0.00012913171980701173 | 0.00010896107404856351 |
| R² | 0.03234692557563468 | -0.0015094093869603142 |
| Best `n_neighbors` | 51 | (same fit) |

**Read of the run:**
CV still picked 51. Cutting from 55-d to 5-d did not change the test story: MSE is flat, R² remains negative. Rolling windows on volume and SMA still drop extra early-train rows.

**vs previous:**

| | Previous (v2) | This (v3) |
|--|-----------------|-----------|
| Change | lag-5 derived bar/volume/return/SMA cols (55 cols) | lag-1 thin set: 1-day return, overnight, range, rel vol, SMA distance (5 cols) |
| Train MSE | 0.00013013554631566388 | 0.00012913171980701173 (-0.00000100, better) |
| Train R² | 0.02657719740120279 | 0.03234692557563468 (+0.00577, better) |
| Test MSE | 0.0001091143271367456 | 0.00010896107404856351 (-0.00000015, better) |
| Test R² | -0.002918026282234143 | -0.0015094093869603142 (+0.00141, better) |

Holdout is still slightly worse than predicting the mean. The train R² bump is larger than the test change.

**Terminal excerpt**

```
Best params: {'kneighborsregressor__n_neighbors': 51}
MSE: 0.00012913171980701173
R^2: 0.03234692557563468
Best params: {'kneighborsregressor__n_neighbors': 51}
MSE: 0.00010896107404856351
R^2: -0.0015094093869603142
```

### v2 - derived bar/volume/return features, lag=5, SPY only

- **Date:** 2026-09-01
- **Status:** superseded
- **Script:** `src/ml/regression/KNN.py`
- **Git:** `d352ecd` (dirty)

**What changed vs v1:**
- Feature set: 25 lagged raw `Open`/`High`/`Low`/`Volume`/`Adj Close` columns → 11 derived columns lagged `_0`..`_4` (55 cols).
- Script now calls `process_OHLCV_all` via `process_data`. Module-level `FEATURE_COLS` still lists raw OHLCV plus the derived names but is unused.
- Data, cutoff, target, lag, scaler, k grid, and SPY-only train are unchanged.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`, period 10y |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all `STARTING_STOCKS` (20 names in `src/ml/config.py`) |
| Symbols trained | SPY |
| Features | `Range`, `Close Location`, `Upper Wick`, `Lower Wick`, `Rel Vol`, `Shock Vol`, `Overnight`, `5_Day_Return`, `10_Day_Return`, `20_Day_Return`, `Dist_From_SMA` at lags `_0`..`_4` (55 cols). Built by `ut.process_OHLCV_all`; raw OHLCV not in the trained matrix |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `StandardScaler` + `sklearn.neighbors.KNeighborsRegressor` |
| Other | `GridSearchCV` scoring `neg_mean_squared_error`; CV `TimeSeriesSplit(n_splits=5, gap=5)` on train; best `n_neighbors=51` (grid ceiling); `plot_data` 2x2 (time series, scatter, return histogram, residuals; recent-return scatter skipped because `Adj Close_0`/`_1` are absent); `DISPLAY_PLOTS=True`; `FEATURE_COLS` unused |

**Metrics**

| Metric | Train | Test |
|--------|-------|------|
| MSE | 0.00013013554631566388 | 0.0001091143271367456 |
| R² | 0.02657719740120279 | -0.002918026282234143 |
| Best `n_neighbors` | 51 | (same fit) |

**Read of the run:**
CV still picked 51. Raw price levels are gone from the trained columns, which removes the v1 price-vs-return scale mix. Same-day derived bar/volume/return/SMA columns remain. Rolling windows drop extra early-train rows vs v1.

**vs previous:**

| | Previous (v1) | This (v2) |
|--|-----------------|-----------|
| Change | lag-5 raw OHLCV (25 cols) | lag-5 derived bar/volume/return/SMA cols (55 cols) |
| Train MSE | 0.0001316423924681682 | 0.00013013554631566388 (-0.00000151, better) |
| Train R² | 0.011467107630209195 | 0.02657719740120279 (+0.0151, better) |
| Test MSE | 0.00010924284405188783 | 0.0001091143271367456 (-0.00000013, better) |
| Test R² | -0.00409928207384791 | -0.002918026282234143 (+0.00118, better) |

Holdout is almost the same as v1: test MSE is flat and test R² is still slightly negative. The train R² bump is more visible than any test change.

**Terminal excerpt**

```
Best params: {'kneighborsregressor__n_neighbors': 51}
MSE: 0.00013013554631566388
R^2: 0.02657719740120279
Best params: {'kneighborsregressor__n_neighbors': 51}
MSE: 0.0001091143271367456
R^2: -0.002918026282234143
```

### v1 - raw OHLCV baseline, SPY only

- **Date:** 2026-09-01
- **Status:** superseded
- **Script:** `src/ml/regression/KNN.py`
- **Git:** `d352ecd` (dirty)

**What changed vs vN-1:**
First logged run.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`, period 10y |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | all `STARTING_STOCKS` (20 names in `src/ml/config.py`) |
| Symbols trained | SPY |
| Features | `Open`/`High`/`Low`/`Volume`/`Adj Close` at lags `_0`..`_4` (25 cols) |
| Target | next-day `Adj Close` return: `Adj Close.shift(-1) / Adj Close - 1` |
| Model | `sklearn.pipeline.Pipeline`: `StandardScaler` + `sklearn.neighbors.KNeighborsRegressor` |
| Other | `GridSearchCV` scoring `neg_mean_squared_error`; CV `TimeSeriesSplit(n_splits=5, gap=5)` on train; best `n_neighbors=51` (grid ceiling); scaler is inside the searched pipeline; `plot_data` 2x2 (time series, scatter, return histogram, recent-return scatter); `DISPLAY_PLOTS=True` |

**Metrics**

| Metric | Train | Test |
|--------|-------|------|
| MSE | 0.0001316423924681682 | 0.00010924284405188783 |
| R² | 0.011467107630209195 | -0.00409928207384791 |
| Best `n_neighbors` | 51 | (same fit) |

**Read of the run:**
CV picked 51, so neighbors are averaged heavily. Test R² slightly negative. Same-day `_0` prices, including `Adj Close_0`, vs a return target; volume z-scored with prices.

**vs previous:**
First logged run.

**Terminal excerpt**

```
Best params: {'kneighborsregressor__n_neighbors': 51}
MSE: 0.0001316423924681682
R^2: 0.011467107630209195
Best params: {'kneighborsregressor__n_neighbors': 51}
MSE: 0.00010924284405188783
R^2: -0.00409928207384791
```
