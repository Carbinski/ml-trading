---
name: document
description: Records machine learning experiment runs in this repo: metrics, current status, and version-to-version changes. Use when the user invokes /document or asks to document model results, performance, or status after a training run.
disable-model-invocation: true
---

# Document ML runs

Log how a learning algorithm performed, its current status, and how it compares to earlier versions of the same method and to other methods in this repo.

This is a learning repo, not a live trading system. Record results honestly. Do not invent metrics. Do not describe a run as profitable, production-ready, or a trading strategy.

## Inputs

The user typically runs `/document` and provides:

1. **Terminal output** from the training run (paste)
2. **Code** (file path, attached file, or paste). Current first script: `src/ml/regression/parametric-regression.py`

Also use the repo: read the script, `src/ml/config.py`, and existing files under `docs/ml/`. Optional user notes about what changed (data quality, features, target, setup) are first-class.

If terminal output is missing and metrics cannot be recovered from the script, ask for the paste. Do not fabricate numbers.

## Where docs live

```
docs/ml/index.md           # status board + comparison across methods
docs/ml/<method-id>.md     # version history for one method
```

`<method-id>` is kebab-case, stable for the algorithm family, not the version. Examples: `parametric-linear-regression`, `knn`, `decision-tree`.

Create `docs/ml/index.md` if it is missing. Create the method page on first run of that method.

Page layouts are in [templates.md](templates.md). Read it before writing.

## Workflow

Copy this checklist and complete it in order:

```
- [ ] Read docs/ml/index.md and the method page if it exists
- [ ] Read the script and src/ml/config.py
- [ ] Parse the terminal paste (metrics, symbol, errors)
- [ ] Decide: new method vs new version vs re-run of an existing version
- [ ] Write or update the method page
- [ ] Update the index status board and latest-metrics row
- [ ] Reply with a short record summary
```

Do not edit `docs/data.md`, training scripts, or other docs unless the user asked.

## Versioning

A **method** is the algorithm family (parametric linear regression, kNN). Keep one page per method.

A **version** (`v1`, `v2`, ...) is a distinct setup of that method. Increment when any of these change:

- Data source or quality (`data/clean-yfinance` vs `data/ml4t`, different cleaning)
- Universe or which symbols were actually trained
- Target definition
- Feature set
- Split rule or cutoff
- Model class or material hyperparameters
- Training procedure (scaling, regularization, leakage fixes, etc.)

**Re-run of the same version** only when the recipe is unchanged (same data recipe, features, target, split, model). Update that version's "Latest run" date and metrics if they differ. Add a one-line re-run note. Do not increment.

If it is unclear, create a new version and say so in the reply.

Status values (one per version):

| Status | Meaning |
|--------|---------|
| `in progress` | Run did not produce a usable baseline (crash, no metrics) |
| `current` | Latest version you intend to iterate from |
| `superseded` | Replaced by a newer version of the same method |
| `abandoned` | Stopped; not replaced |

The newest intended version is `current`. When adding `vN`, mark the previous `current` as `superseded` unless the user abandoned the new run. Call the first working run a baseline in the version label (`v1 - raw OHLCV baseline`). A working train that still has TODOs (plot, one ticker) is still `current`.

## What to extract

Record what was **actually trained**, not only what the script can load.

From **code + config**:

- Script path
- Model class and library
- Target formula (write it explicitly, e.g. next-day `Adj Close` return)
- Feature columns
- Split: chronological only. Note cutoff (`CUTOFF` in `src/ml/config.py` is currently `2024-09-01`)
- Data dir and symbols. If the script loads `STARTING_STOCKS` but trains one ticker, say so
- Incomplete pieces (TODOs, unused plots)

From **terminal output**:

- Printed metrics (MSE, R², weights, intercept, accuracy, etc.)
- Errors or warnings
- Enough context to identify the run (symbol heads are optional; do not dump huge frames)

Optional: `git rev-parse --short HEAD` and whether the tree is dirty. Skip if noisy.

## Comparisons

On every new version, write **vs previous** on the method page:

- What changed (data vs features vs setup)
- Metric deltas (absolute and directional)
- Whether the change is a real improvement or an artifact (leakage, different test window, different target scale)

On the index, keep the status board showing **only the current version** of each method so methods can be compared at a glance. Historical rows stay on the method page.

## Honesty checks (this repo)

Flag these in the version notes when they apply. Do not lecture in the chat reply.

- Same-day prices (especially `Close` / `Adj Close`) used as features for a next-day return from those prices
- Lookahead: `bfill`, shuffle, or k-fold across time
- Target and features on incompatible scales
- Trained on one symbol after loading many
- Stale ML4T homework dump (ends 2012) vs local Yahoo cache
- Missing or unusable metrics

This repo's hard rules: chronological split, no lookahead fills, no brokerage or order code, do not call the work a live bot.

## Writing rules

- Match `docs/data.md`: specific, no fluff, no em dashes
- Do not over-explain ML fundamentals
- Prefer the user's words for what they were trying (baseline before a better setup, higher-quality data, etc.)
- Keep raw terminal excerpts short (metrics block, not the full session)
- Update existing sections in place. Do not rewrite unrelated versions

## Reply after writing

Short, in this order:

1. Method, version, status
2. Key metrics
3. vs previous (or "first version")
4. Files touched
