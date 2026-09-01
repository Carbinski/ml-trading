# ML experiment doc templates

Copy these structures. Fill from the run. Do not leave placeholder prose in the live docs.

## `docs/ml/index.md`

```markdown
# ML experiment log

Living record of learning algorithms tried in this repo: current status, latest metrics, and links to version history.

## Status board

| Method | Version | Status | Data | Target | Symbols trained | Key metrics | Page |
|--------|---------|--------|------|--------|-----------------|-------------|------|
| Parametric linear regression | v1 | current | clean-yfinance, 10y daily | next-day Adj Close return | SPY | MSE …; R² … | [parametric-linear-regression](parametric-linear-regression.md) |

## How to read this

One row per method, always the **current** version. Full history and vs-previous notes live on the method page.

Add or update rows only from a documented training run.
```

Only include families that exist. If you want groupings later, use short headers (`## Regression`) above subsets of the table. Do not add empty family sections.

## Method page: `docs/ml/<method-id>.md`

```markdown
# <Method name>

**Script:** `src/ml/...`
**Family:** regression | neighbors | trees | other
**Current:** vN (`current`)

One or two sentences on what this method is doing in this repo (target, model class). Not a textbook definition.

## Versions

### vN - <short label>

- **Date:** YYYY-MM-DD
- **Status:** current
- **Script:** `src/ml/...`
- **Git:** `<short-hash>` (dirty) | not recorded

**What this version is:**
One sentence.

**What changed vs vN-1:**
- … (omit on v1; write "First logged run.")

**Setup**

| Item | Value |
|------|--------|
| Data | `data/clean-yfinance`, period 10y |
| Calendar / split | dates < `2024-09-01` train, else test |
| Symbols loaded | … |
| Symbols trained | … |
| Features | … |
| Target | … |
| Model | `sklearn.linear_model.LinearRegression` |
| Other | scaling, regularization, incomplete TODOs |

**Metrics**

| Metric | Value |
|--------|--------|
| MSE | … |
| R² | … |
| Intercept | … |

Weights (if printed):

| Feature | Weight |
|---------|--------|
| Open | … |

**Read of the run:**
A few sentences: did it fit anything, leakage or setup issues, what is still unfinished. No strategy claims.

**vs previous:**
Only when this is not v1. Table plus one short paragraph (better / worse / not comparable, and why).

| | Previous (vN-1) | This (vN) |
|--|-----------------|-----------|
| Change | … | … |
| MSE | … | … (delta) |
| R² | … | … (delta) |

**Latest run notes:**
Optional. Re-runs of the same recipe go here.

**Terminal excerpt**

\`\`\`
Weights: ...
Intercept: ...
MSE: ...
R^2: ...
\`\`\`
```

Put the newest version at the **top** of `## Versions`. Keep older versions below, with status updated (`superseded` / `abandoned`).

Do not duplicate the vs-previous table on older versions.

## Short labels

Examples of good version labels:

- `v1 - raw OHLCV baseline, SPY only`
- `v2 - same model, returns features instead of prices`
- `v3 - all starting stocks, still next-day return`

The label should say what is different about the setup, not "improved" or "better data" unless the user used that phrasing and you also state the concrete change.
