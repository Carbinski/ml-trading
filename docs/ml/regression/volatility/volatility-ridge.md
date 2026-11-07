# Volatility ridge

**Script:** `src/ml/regression/ai_scripts/ai-volatility-regression.py`
**Family:** regression
**Current:** v1 (`current`)

One pooled degree-1 Ridge fit predicts the absolute next-day adjusted close-to-close return for every CSV in `data/clean-yfinance`. Alpha is chosen on train CV only. Validation is reported and not used to pick alpha.

## Versions

### v1 - pooled Ridge, next-day absolute return

- **Date:** 2026-10-30
- **Status:** current
- **Script:** `src/ml/regression/ai_scripts/ai-volatility-regression.py`
- **Git:** `8834a22` (dirty)

**What this version is:**
One model shared across symbols. The script names the experiment `next-day volatility target`. The target is the absolute next-day adjusted close-to-close return.

**What changed:**
First logged run.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`, every CSV in that directory. AAPL, AMZN, CAT, GOOG, IBM, JNJ, JPM, KO, LIN, META, MSFT, NEE, NVDA, PG, PLD, UNH, UNP, V, XOM: 2512 rows, 2016-09-01 through 2026-08-31. SPY: 6285 rows, 2001-09-04 through 2026-09-01. |
| Calendar / split | `VAL_CUTOFF` is `2024-09-01` (`CUTOFF` in `src/ml/config.py`). `TEST_CUTOFF` is `2025-09-01`. Train: feature date and target date both before 2024-09-01. Validation: feature date in [2024-09-01, 2025-09-01) and target date before 2025-09-01. Test: feature date on or after 2025-09-01. This is a different window from the single 2024-09-01 holdout on the status board. |
| Symbols loaded | AAPL, AMZN, CAT, GOOG, IBM, JNJ, JPM, KO, LIN, META, MSFT, NEE, NVDA, PG, PLD, SPY, UNH, UNP, V, XOM (20 files) |
| Symbols trained | all 20, one pooled model |
| Features | `horizon=1`, `lag=1`. `lag=1` keeps offset 0 only (`shift(0)`, suffix `_0`): `Ret_1_0`, `Overnight_0`, `Intraday_0`, `Range_0`, `Close_Location_0`, `Relative_Volume_0`, `Vol_20_0`, `5_Day_Return_0`, `20_Day_Return_0`, `Dist_From_SMA_0`. No raw price levels. Column definitions are `_engineer_features` in `src/ml/regression/ai_scripts/parametric_experiment_common.py`. |
| Target | `ForwardReturn.abs()`, with `ForwardReturn = Adj Close.shift(-1) / Adj Close - 1` |
| Model | `make_pipeline(PolynomialFeatures(degree=1, include_bias=False), StandardScaler(), Ridge())`. `GridSearchCV` on `ridge__alpha`, scoring `neg_mean_squared_error`. Alphas `np.logspace(-4, 4, 30)`. CV is `TimeSeriesSplit(n_splits=5, gap=5)` on unique train dates, then applied to every row on those dates. |
| Other | Best alpha `117.21022975334793`, fit on train rows only. `PolynomialFeatures(degree=1, include_bias=False)` keeps the 10 columns and adds no interactions. Negative predictions are counted, then clipped to 0 before `evaluate_regression`. Baseline is the per-symbol training mean of the absolute-return target. Run from the repo root with `.venv/bin/python`, `--n-jobs` left at the default `-1`. |

**Metrics**

MSE improvement is `100 * (baseline_mse - mse) / baseline_mse`. Correlation in the pooled row is `np.corrcoef` on all rows in that window. Macro R² is the unweighted mean of per-symbol R².

| Metric | Validation | Test |
|--------|------------|------|
| Observations | 4960 | 5001 |
| Stocks | 20 | 20 |
| Micro R² | 0.11453272583223684 | 0.04309196944831917 |
| Macro R² | 0.04364152816522181 | -0.027699534936528568 |
| Median R² | 0.03657142563082699 | -0.032890449193797244 |
| Micro MSE | 0.0001777472984005604 | 0.00017029680975067383 |
| Micro MAE | 0.008401954077332368 | 0.00841488380056801 |
| Baseline MSE (train-mean) | 0.00019131650648703064 | 0.00017289459212143873 |
| MSE improvement vs train mean (%) | 7.092544357844047 | 1.5025237856718217 |
| Correlation (pooled) | 0.3506545232818901 | 0.24408164226553422 |
| Macro correlation | 0.2499936878000913 | 0.10888533879758555 |
| Prediction std | 0.00622039195105558 | 0.004968522913147567 |
| Stocks beating train-mean baseline | 15 | 10 |
| Stocks with positive R² | 14 | 2 |
| Negative prediction rate before clipping | 0.0 | 0.0 |
| Mean actual absolute return | 0.012159663853072857 | 0.012466147032502083 |
| Mean predicted absolute return | 0.012510784142499325 | 0.012502410394789758 |
| Pooled MSE below train-mean baseline | yes | yes |

**Per-symbol test**

Validation rows are 248 for every symbol (4960). Test rows are 250 for every symbol except SPY (251), 5001 total.

| Symbol | Rows | R² | MSE | MSE improvement vs train mean (%) | Correlation | Beat train-mean baseline |
|--------|------|----|-----|-----------------------------------|-------------|--------------------------|
| AAPL | 250 | 0.030511614150898758 | 0.00012047907508660692 | 4.851432269776153 | 0.20210782214549808 | yes |
| AMZN | 250 | -0.02599703314691726 | 0.00024179209168323664 | -2.156404516963123 | 0.0877337999984615 | no |
| CAT | 250 | -0.04814887169594839 | 0.0003037223760076239 | 3.0362909902434305 | 0.074793920043041 | yes |
| GOOG | 250 | -0.026607020483455424 | 0.00021400769708471607 | -1.2265681785308848 | 0.06441181470621213 | no |
| IBM | 250 | -0.04369721271762139 | 0.0005959980881693439 | 8.795645423405924 | 0.12671578460202026 | yes |
| JNJ | 250 | -0.011731176546752708 | 6.094201944331698e-05 | 1.360755237364797 | 0.14494666624709404 | yes |
| JPM | 250 | -0.07153389826502954 | 8.940471450803104e-05 | -6.1685942400965885 | 0.05142673970028055 | no |
| KO | 250 | -0.04755385806570267 | 6.6362804759087e-05 | -2.952770007362059 | 0.05157002596898602 | no |
| LIN | 250 | -0.010324436855100538 | 5.8150438216135576e-05 | 1.116477825879978 | 0.14617680616912718 | yes |
| META | 250 | -0.0321950702943028 | 0.00032639992514607916 | -3.085512396553876 | 0.09431476787109558 | no |
| MSFT | 250 | -0.0026305963927313325 | 0.00023093705689182644 | 0.8145193831444906 | 0.14693866393848665 | yes |
| NEE | 250 | -0.035504922200450206 | 8.728916629685173e-05 | -2.7059286430230447 | 0.12836405525054653 | no |
| NVDA | 250 | -0.039371410532461626 | 0.00023398522426151084 | 3.944306054888182 | 0.08990142222876184 | yes |
| PG | 250 | -0.03358582809329169 | 5.7829652664188954e-05 | 2.797001810451473 | 0.08935037115730646 | yes |
| PLD | 250 | -0.02536148991466569 | 8.401747181010186e-05 | 2.098052716022215 | 0.11929738004627716 | yes |
| SPY | 251 | 0.07115781068296334 | 2.5755452459947326e-05 | 16.37707773670277 | 0.2971208128430008 | yes |
| UNH | 250 | -0.0713986334825163 | 0.0002987843869629232 | -1.6564262140770694 | 0.05571471983861425 | no |
| UNP | 250 | -0.023272712123986627 | 0.00010043891015741308 | -0.6916522613622275 | 0.10807167207010969 | no |
| V | 250 | -0.06185120982704606 | 0.00010057332451752207 | -5.863648778890041 | 0.020082167409987615 | no |
| XOM | 250 | -0.044894742926453235 | 0.00010964448431617662 | -4.48066867055781 | 0.07866736371680355 | no |

**Read of the run:**
Pooled test micro R² is 0.04309196944831917. Macro R² is -0.027699534936528568. Ten of 20 symbols have test MSE below their training-mean baseline. Two of 20 have positive test R²: AAPL and SPY. R² uses the evaluation-window mean. The baseline is the training-period per-symbol mean of the absolute return. Mean actual absolute return on the test rows is 0.012466147032502083. Mean predicted absolute return is 0.012502410394789758. The negative-prediction rate before clipping is 0 on validation and on test, so clipping left both scores unchanged.

Validation micro R² is 0.11453272583223684 and macro R² is 0.04364152816522181, with 15 of 20 symbols beating the training-mean baseline and 14 of 20 with positive R². Alpha is chosen on train CV only. Validation is reported and not used to pick alpha.

Features are same-day (lag 0, suffix `_0`) close-derived columns for a next-day absolute return.

The test block is feature dates on or after 2025-09-01, and the target is an absolute return. Parametric linear v8, kNN v8, decision tree v2, kernel SVR v6, and voting v2 on the status board score signed next-day returns with cutoff 2024-09-01. This R² is a separate measurement from those rows.

**Latest run notes:**
At `8834a22` the script inserted `Path(__file__).resolve().parents[3]` (the `src/` directory) and imported `src.ml.regression.parametric_experiment_common`. The first execution failed with `ModuleNotFoundError: No module named 'src'`. The helper lives at `src/ml/regression/ai_scripts/parametric_experiment_common.py` and imports `src.ml.config`, which needs the repo root (`parents[4]`) on `sys.path`. The logged run uses that two-line import fix only. Target, features, split, model, clipping, and metrics code were left as written.

The working tree was clean at `8834a22` before the import edit. This run is the dirty tree with that uncommitted fix. Command, from the repo root:

```
.venv/bin/python src/ml/regression/ai_scripts/ai-volatility-regression.py --output /tmp/volatility-ridge-results.json
```

Exit code 0. Stderr was empty. The JSON file was temporary and was not added to the repo. It was deleted after these numbers were copied.

**Terminal excerpt**

```
Volatility test: micro R2=0.043092, macro R2=-0.027700, MSE improvement vs stock means=1.503%, correlation=0.2441.
Wrote results to /tmp/volatility-ridge-results.json
```
