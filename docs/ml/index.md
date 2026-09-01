# ML experiment log

Living record of learning algorithms tried in this repo: current status, latest metrics, and links to version history.

## Status board

| Method | Version | Status | Data | Target | Symbols trained | Key metrics | Page |
|--------|---------|--------|------|--------|-----------------|-------------|------|
| Parametric linear regression | v8 | current | clean-yfinance, 10y daily | next-day Adj Close return | SPY | train MSE 0.000133, R² 0.006; test MSE 0.000110, R² -0.008 | [parametric-linear-regression](regression/parametric-regression/parametric-linear-regression.md) |
| kNN regression | v8 | current | clean-yfinance, 10y daily | next-day Adj Close return | all 20 | v5-recipe macro test R² k=15 -0.0515, k=21 -0.0325, k=51 -0.0111; k=15 beats k=51 on 1/20 | [knn](regression/knn/knn.md) |

## How to read this

One row per method, always the **current** version. Full history and vs-previous notes live on the method page.

Add or update rows only from a documented training run.
