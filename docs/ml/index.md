# ML experiment log

Living record of learning algorithms tried in this repo: current status, latest metrics, and links to version history.

## Status board

| Method | Version | Status | Data | Target | Symbols trained | Key metrics | Page |
|--------|---------|--------|------|--------|-----------------|-------------|------|
| Parametric linear regression | v8 | current | clean-yfinance, 10y daily | next-day Adj Close return | SPY | train MSE 0.000133, R² 0.006; test MSE 0.000110, R² -0.008 | [parametric-linear-regression](regression/parametric-regression/parametric-linear-regression.md) |
| kNN regression | v8 | current | clean-yfinance, 10y daily | next-day Adj Close return | all 20 | v5-recipe macro test R² k=15 -0.0515, k=21 -0.0325, k=51 -0.0111; k=15 beats k=51 on 1/20 | [knn](regression/knn/knn.md) |
| Decision tree regression | v2 | current | clean-yfinance, 25y SPY | next-day Adj Close return | SPY | train MSE 0.000139, R² 0.047; test MSE 0.000109, R² -0.002; `max_depth=2` | [decision-tree](regression/decision-tree/decision-tree.md) |
| Kernel SVR | v6 | current | clean-yfinance, 10y daily (SPY 25y) | next-day Adj Close return | all 20 | v4-recipe macro test R² -0.0074, 5/20 positive; v5 -0.0137, 4/20; v5 beats v4 on 7/20 | [svm](regression/svm/svm.md) |
| Voting regressor | v2 | current | clean-yfinance, 25y SPY | next-day Adj Close return | SPY | train MSE 0.000122, R² 0.158; test MSE 0.000112, R² -0.028 | [voting-regressor](ensemble-learning/voting-regressor.md) |
| Volatility ridge | v1 | current | clean-yfinance daily, all 20 CSVs | absolute next-day Adj Close return | all 20, pooled | best alpha 117.21022975334793; test micro R² 0.043092, macro R² -0.027700; MSE improvement vs stock means 1.503%; correlation 0.2441; 10/20 beat baseline; 2/20 positive R² | [volatility-ridge](regression/volatility/volatility-ridge.md) |
| Tabular Q-learning | v1 | current | clean-yfinance, 25y SPY | next-day Adj Close return while long, else 0 | SPY | best alpha 0.0001; test growth of $1 0.3085 vs stock 0.4124; train 12.6361 vs 7.0878; 0/5 alphas beat the stock on test | [q-learning](reinforcement-learning/q-learning.md) |
| Dyna-Q | v1 | current | clean-yfinance, 25y SPY | next-day Adj Close return while long, else 0 | SPY | best alpha 1e-5, 5 planning steps; test growth of $1 0.3957 vs stock 0.4124; train 17.3449 vs 7.0878; 0/5 alphas beat the stock on test | [dyna-q](reinforcement-learning/dyna-q.md) |

## How to read this

One row per method, always the **current** version. Full history and vs-previous notes live on the method page.

Add or update rows only from a documented training run.

## Todo

Do these in order. Same rules as the board: `data/clean-yfinance`, cutoff `2024-09-01`, chronological split, no shuffled rows. Features for items 1 to 4 are the stationary columns in `src/ml/regression/ai_scripts/parametric_experiment_common.py` (`Ret_1`, `Overnight`, `Intraday`, `Range`, `Close_Location`, `Relative_Volume`, `Vol_20`, `5_Day_Return`, `20_Day_Return`, `Dist_From_SMA`). No raw price levels. Signed next-day return regression is already on the board. Holdout R² there is at or below zero.

| Order | Experiment | Target | Model | Score against |
|-------|------------|--------|-------|----------------|
| 1 | Single-name direction | sign of the next-day `Adj Close` return. Start with SPY | NumPy SGD logistic: log loss, gradient `(sigmoid(w·x) - y) x`, fixed or decaying step size, L2 as weight decay. A scaled scikit-learn `LogisticRegression` is a gradient check after the loop has a number | always-up accuracy, and long/flat growth of $1 vs holding the stock |
| 2 | Cross-sectional median | 1 when that name's next-day return beats the same day's median across `STARTING_STOCKS`. One row per symbol-day | the same SGD logistic | equal-weight basket of those names. Long only the names predicted to beat the median, against the equal-weight basket of all 20 |
| 3 | Linear Q-learning | next-day `Adj Close` return while long, 0 while flat | one weight vector per action. Tabular semi-gradient with the feature vector in place of the price/SMA one-hot state. Train before the cutoff, greedy after | growth of $1 vs holding the stock |
| 4 | Absolute-return net | absolute next-day `Adj Close` return | one hidden layer, SGD. Same split and features as volatility ridge | volatility ridge test micro R² and correlation, on a matched window |

`src/ml/reinforcement-learning/dyna-q-cross-section.py` is a different experiment. It fits a separate long/flat policy per name. It does not rank names against each other on the same day.
