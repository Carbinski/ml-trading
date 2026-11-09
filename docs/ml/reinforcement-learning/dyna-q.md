# Dyna-Q

**Script:** `src/ml/reinforcement-learning/dyna-q.py`
**Family:** other
**Current:** v1 (`current`)

Dyna-Q on SPY. Same table, bins, reward, and alpha list as tabular Q-learning. After each real backup the agent stores that transition and takes 5 updates from the stored model.

## Versions

### v1 - price/SMA bins, SPY, 5 planning steps

- **Date:** 2026-11-08
- **Status:** current
- **Script:** `src/ml/reinforcement-learning/dyna-q.py`
- **Git:** `7f128ea` (dirty; docs only, training script unchanged)

**What this version is:**
One SPY run of the script's default alpha list. `DynaQ` subclasses `QLearning`. Each alpha trains a fresh table for 1000 epochs, then walks the test window greedy from flat.

**What changed:**
First logged run.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance/SPY.csv`, 6285 rows, 2001-09-04 through 2026-09-01 |
| Calendar / split | dates < `2024-09-01` train, else test. Train bars 5784 (2001-09-04 through 2024-08-30). Test bars 501 (2024-09-03 through 2026-09-01). The greedy test walk takes 500 steps because the last bar has no next close |
| Symbols loaded | SPY |
| Symbols trained | SPY |
| Features | Same as tabular Q-learning v1. `Price_to_SMA_Ratio` = `Adj Close` / 20-day mean of `Adj Close.shift(1)`. 22 bins. State id is `position * 22 + bin`. Printed train days with a ratio: 5764 |
| Target | Same reward as tabular Q-learning v1. Next-day `Adj Close` return if the new position is long, else 0. Flat may buy or hold. Long may hold or sell |
| Model | `DynaQ` in `src/ml/reinforcement-learning/dyna-q.py`. Table shape `(44, 3)`. The Q update matches `QLearning.update`. The model keeps the last seen `(reward, next state, next position, done)` for each `(state, action)` and is kept from epoch to epoch. `plan` draws a stored state, then a stored action from that state, and repeats the Q update. `PLANNING_STEPS = 5` after every real step, including the step that ends the train window |
| Other | `ALPHAS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)`. `EPOCHS = 1000`. Exploration falls linearly from 0.1 to 0.05. `SEED = 0`. `GAMMA` printed `0.999920` (`0.98 ** (1/252)`). Train and test both start flat. `DISPLAY_PLOTS=True`. This run set `MPLBACKEND=Agg`, so `plt.show()` did not open a window. No costs |

**Metrics**

Growth of $1 is compounded in `visualize`. "Sum of daily rewards" is what `evaluate` prints as `Policy equity change`. That sum is not the compounded growth. Stock growth matches tabular Q-learning v1 because the file and the cutoff are the same.

| Alpha | Train growth of $1 | Test growth of $1 | Test sum of daily rewards |
|-------|--------------------|-------------------|---------------------------|
| 1e-5 | 17.3449 | 0.3957 | 0.3558 |
| 3e-5 | 9.6651 | 0.1893 | 0.1946 |
| 0.0001 | 10.1428 | 0.2940 | 0.2836 |
| 0.0003 | 17.7454 | 0.2043 | 0.2086 |
| 0.001 | 7.2662 | 0.2706 | 0.2614 |
| Stock | 7.0878 | 0.4124 | |

Highest test growth is alpha `1e-5`: policy 0.3957, stock 0.4124. Every alpha is below the stock on the test window. At that alpha the printed greedy table buys in 16 of 22 flat bins and sells in 5 of 22 long bins.

**Read of the run:**
The best holdout policy stays long in most bins, so its test growth sits close to buy-and-hold and still finishes below it (0.3957 vs 0.4124). Train growth at that alpha is 17.3449 against a stock gain of 7.0878. The state uses today's `Adj Close` in the ratio, and the reward is the return from that same close to the next close.

Same SPY file, cutoff, bins, epochs, exploration schedule, gamma, seed, and alpha list as tabular Q-learning v1. The added piece is the 5 model updates. One seed. Planning changes the Q values during training, so the exploratory actions are not the same sequence as Q-learning.

| Alpha | Q-learning v1 test growth | Dyna-Q v1 test growth |
|-------|---------------------------|------------------------|
| 1e-5 | 0.1443 | 0.3957 |
| 3e-5 | 0.1931 | 0.1893 |
| 0.0001 | 0.3085 | 0.2940 |
| 0.0003 | 0.2512 | 0.2043 |
| 0.001 | 0.2436 | 0.2706 |

Dyna-Q is higher on 2 of 5 alphas. Its best test growth (0.3957) is above Q-learning's best (0.3085 at alpha `0.0001`). At the shared alpha `0.0001`, Dyna-Q's test growth is 0.2940, below Q-learning. Both best settings lose to the stock.

**Latest run notes:**
Command, from the repo root:

```
MPLBACKEND=Agg .venv/bin/python src/ml/reinforcement-learning/dyna-q.py SPY
```

Exit code 0. Elapsed about 47 minutes. The alpha grid prints `1e-5` and `3e-5` as `0.0000`. The table above uses the per-alpha lines, which print the step size.

**Terminal excerpt**

```
gamma=0.999920  value after 252 trading days=0.98
Train days=5764  bins=22

Training SPY for 1000 epochs, alpha=1e-05, planning_steps=5, exploration 0.1 to 0.05, cutoff 2024-09-01
Evaluate steps: 500
Policy equity change: 0.3558
Buy-and-hold equity change: 0.4124
Train growth of $1: policy 17.3449  stock 7.0878
Test growth of $1:  policy 0.3957  stock 0.4124

Stock growth of $1: train 7.0878  test 0.4124
Alpha grid
 alpha   train   test
0.0000 17.3449 0.3957
0.0000  9.6651 0.1893
0.0001 10.1428 0.2940
0.0003 17.7454 0.2043
0.0010  7.2662 0.2706
```
