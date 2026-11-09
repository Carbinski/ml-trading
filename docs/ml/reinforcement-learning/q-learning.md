# Tabular Q-learning

**Script:** `src/ml/reinforcement-learning/q-learning.py`
**Family:** other
**Current:** v1 (`current`)

Tabular Q-learning on SPY. State is flat or long, times a bin of today's `Adj Close` over the prior 20-day average. Reward is the next close-to-close return while long, and 0 while flat.

## Versions

### v1 - price/SMA bins, SPY alpha sweep

- **Date:** 2026-11-08
- **Status:** current
- **Script:** `src/ml/reinforcement-learning/q-learning.py`
- **Git:** `7f128ea` (dirty; docs only, training script unchanged)

**What this version is:**
One SPY run of the script's default alpha list. Each alpha trains a fresh zero table for 1000 epochs, then walks the test window greedy from flat.

**What changed:**
First logged run.

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance/SPY.csv`, 6285 rows, 2001-09-04 through 2026-09-01 |
| Calendar / split | dates < `2024-09-01` train, else test. Train bars 5784 (2001-09-04 through 2024-08-30). Test bars 501 (2024-09-03 through 2026-09-01). The greedy test walk takes 500 steps because the last bar has no next close |
| Symbols loaded | SPY |
| Symbols trained | SPY |
| Features | `Price_to_SMA_Ratio` = `Adj Close` / 20-day mean of `Adj Close.shift(1)`. Bins are 0.005 wide from 0.95 to 1.05 (`EDGES`), plus the two tails: 22 bins. State id is `position * 22 + bin`. The first 20 bars have no ratio (window starts 2001-10-08). Printed train days with a ratio: 5764 |
| Target | Reward after the action at t: next-day `Adj Close` return `Adj Close[t+1] / Adj Close[t] - 1` if the new position is long, else 0. Flat may buy or hold. Long may hold or sell |
| Model | `QLearning` in `src/ml/reinforcement-learning/q-learning.py`. Table shape `(44, 3)` for buy, sell, hold. Update is `Q ← (1 - alpha) * Q + alpha * target`, with `target = reward` on the last train bar and `reward + gamma * max legal Q(next)` otherwise |
| Other | `ALPHAS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)`. `EPOCHS = 1000`. Exploration falls linearly from 0.1 to 0.05. `SEED = 0`. `GAMMA = 0.98 ** (1/252)`, printed `0.999920`. `ANNUAL_RETAINED = 0.98`. Train and test both start flat. `DISPLAY_PLOTS=True`. This run set `MPLBACKEND=Agg`, so `plt.show()` did not open a window. No costs |

**Metrics**

Growth of $1 is compounded in `visualize`: equity starts at 1, and each step multiplies by `1 + reward`. Train stock growth uses the close at the window start (2001-10-08) through 2024-08-30. Test stock growth is the close on 2026-09-01 over the close on 2024-09-03, minus 1. "Sum of daily rewards" is what `evaluate` prints as `Policy equity change`. That sum is not the compounded growth. Buy-and-hold in `evaluate` matches the test stock growth below.

| Alpha | Train growth of $1 | Test growth of $1 | Test sum of daily rewards |
|-------|--------------------|-------------------|---------------------------|
| 1e-5 | 23.0996 | 0.1443 | 0.1540 |
| 3e-5 | 12.9842 | 0.1931 | 0.1898 |
| 0.0001 | 12.6361 | 0.3085 | 0.2846 |
| 0.0003 | 6.5007 | 0.2512 | 0.2315 |
| 0.001 | 6.3434 | 0.2436 | 0.2250 |
| Stock | 7.0878 | 0.4124 | |

Highest test growth is alpha `0.0001`: policy 0.3085, stock 0.4124. Every alpha is below the stock on the test window. At that alpha the printed greedy table buys in 11 of 22 flat bins and sells in 14 of 22 long bins.

**Read of the run:**
Smaller step sizes fit the training path harder (train growth 23.0996 at `1e-5` against a stock gain of 7.0878) and give back more of the holdout. The best holdout setting still finishes below buy-and-hold. The state uses today's `Adj Close` in the ratio, and the reward is the return from that same close to the next close. `evaluate` and `visualize` disagree in form: the printed policy equity change is a sum of 500 daily rewards, and the alpha grid is compounded growth of $1.

**Latest run notes:**
Command, from the repo root:

```
MPLBACKEND=Agg .venv/bin/python src/ml/reinforcement-learning/q-learning.py SPY
```

Exit code 0. Elapsed about 21 minutes. The alpha grid prints `1e-5` and `3e-5` as `0.0000`. The table above uses the per-alpha lines, which print the step size.

**Terminal excerpt**

```
gamma=0.999920  value after 252 trading days=0.98
Train days=5764  bins=22

Training SPY for 1000 epochs, alpha=0.0001, exploration 0.1 to 0.05, cutoff 2024-09-01
Evaluate steps: 500
Policy equity change: 0.2846
Buy-and-hold equity change: 0.4124
Train growth of $1: policy 12.6361  stock 7.0878
Test growth of $1:  policy 0.3085  stock 0.4124

Stock growth of $1: train 7.0878  test 0.4124
Alpha grid
 alpha   train   test
0.0000 23.0996 0.1443
0.0000 12.9842 0.1931
0.0001 12.6361 0.3085
0.0003  6.5007 0.2512
0.0010  6.3434 0.2436
```
