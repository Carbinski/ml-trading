"""
Grid search for the tabular Q-learning agent.

Alpha is concentrated below the previous grid (0.001, 0.01, 0.05, 0.1, 0.3),
with those older values kept as anchors. The other free learning parameters
are the exploration rate and the annual value retained by gamma.

Training uses the same updates, action tie-breaks, and random draws as
QLearning.train. Prices and market bins are read from arrays so a full
1000-epoch run is a few seconds. `python q-learning-grid.py --check` compares
the Q-table to QLearning.train before a search.

Run shards in parallel with `--shard i --shards N --seed S`. The figures next
to this script are from five seeds of this grid, then a finer pass on SPY at
annual retained 0.98: exploration 0.60 to 1.00 on alphas at or below 0.001.
At alpha 0.0003 and exploration 0.70, annual retained 0.90 and 1.00 matched
0.98.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

_SPEC = importlib.util.spec_from_file_location(
    "q_learning", Path(__file__).with_name("q-learning.py")
)
ql = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ql)

from src.ml.config import CLEAN_DIR, CUTOFF

# Lower than the previous cluster at 0.05 / 0.1 / 0.3, plus those anchors.
ALPHAS = (
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    0.01,
    0.03,
    0.05,
    0.1,
    0.3,
)
EXPLORATION = (0.02, 0.05, 0.1, 0.2, 0.4)
# 0.98 is the script default. 1.0 is undiscounted. Lower values discount faster.
ANNUAL_RETAINED = (0.90, 0.95, 0.98, 1.0)
SEED = ql.SEED
EPOCHS = ql.EPOCHS
SYMBOL = "SPY"


def annual_to_gamma(annual_retained: float) -> float:
    return float(annual_retained ** (1.0 / ql.TRADING_DAYS_PER_YEAR))


def configurations() -> list[dict]:
    rows = []
    for annual in ANNUAL_RETAINED:
        gamma = annual_to_gamma(annual)
        for alpha in ALPHAS:
            for exploration in EXPLORATION:
                rows.append(
                    {
                        "alpha": float(alpha),
                        "exploration_prob": float(exploration),
                        "annual_retained": float(annual),
                        "gamma": gamma,
                    }
                )
    return rows


def _prepare(agent: ql.QLearning):
    prices = agent.df["Adj Close"].to_numpy(dtype=float)
    ratio = agent.df["Price_to_SMA_Ratio"].to_numpy(dtype=float)
    bins = np.full(len(prices), -1, dtype=np.int16)
    bins[ql.WINDOW :] = np.digitize(ratio[ql.WINDOW :], ql.EDGES).astype(np.int16)
    dates = pd.to_datetime(agent.df["Date"]).to_numpy()
    last_train = int(np.flatnonzero(dates < np.datetime64(CUTOFF))[-1])
    return prices, bins, last_train


def train_fast(agent: ql.QLearning, seed: int = SEED) -> None:
    """Same Q updates as QLearning.train, without the per-bar pandas reads."""
    np.random.seed(seed)
    prices, bins, last_train = _prepare(agent)
    q = agent.q_table
    alpha = agent.alpha
    gamma = agent.gamma
    eps = agent.exploration_prob
    n_bins = ql.MARKET_BINS
    n = len(prices)
    legal_flat = [ql.BUY, ql.HOLD]
    legal_long = [ql.HOLD, ql.SELL]
    one_minus = 1.0 - alpha

    for _epoch in range(agent.epochs):
        position = ql.FLAT
        t = ql.WINDOW
        if t > last_train:
            break
        state_id = int(bins[t])
        while True:
            legal = legal_flat if position == ql.FLAT else legal_long
            if np.random.rand() < eps:
                action = int(np.random.choice(legal))
            else:
                v0 = q[state_id, legal[0]]
                action = legal[0] if v0 >= q[state_id, legal[1]] else legal[1]
            next_t = t + 1
            if next_t >= n or next_t > last_train:
                break
            if action == ql.HOLD:
                next_position = position
            elif position == ql.FLAT:
                next_position = ql.LONG
            else:
                next_position = ql.FLAT
            next_state_id = next_position * n_bins + int(bins[next_t])
            if next_position == ql.LONG:
                reward = prices[next_t] / prices[t] - 1.0
            else:
                reward = 0.0
            if next_t == last_train:
                target = reward
            else:
                if next_position == ql.FLAT:
                    future = max(q[next_state_id, ql.BUY], q[next_state_id, ql.HOLD])
                else:
                    future = max(q[next_state_id, ql.HOLD], q[next_state_id, ql.SELL])
                target = reward + gamma * future
            q[state_id, action] = one_minus * q[state_id, action] + alpha * target
            state_id = next_state_id
            position = next_position
            t = next_t
            if next_t == last_train:
                break


def load_frame(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(CLEAN_DIR / f"{symbol}.csv", parse_dates=["Date"])
    return df.sort_values("Date").dropna().reset_index(drop=True)


def run_one(df: pd.DataFrame, spec: dict, symbol: str, seed: int = SEED) -> dict:
    agent = ql.QLearning(
        df,
        spec["alpha"],
        exploration_prob=spec["exploration_prob"],
        gamma=spec["gamma"],
        epochs=EPOCHS,
    )
    started = time.perf_counter()
    train_fast(agent, seed=seed)
    seconds = time.perf_counter() - started
    curve = _quiet_curve(agent, symbol)
    return {
        "alpha": spec["alpha"],
        "exploration_prob": spec["exploration_prob"],
        "annual_retained": spec["annual_retained"],
        "gamma": spec["gamma"],
        "seed": seed,
        "epochs": EPOCHS,
        "train": curve["train_policy"],
        "test": curve["test_policy"],
        "train_stock": curve["train_stock"],
        "test_stock": curve["test_stock"],
        "seconds": seconds,
    }


def _quiet_curve(agent: ql.QLearning, symbol: str) -> dict:
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return agent.visualize(symbol)


def check_equivalence(df: pd.DataFrame) -> None:
    cases = [
        (0.01, 0.1, annual_to_gamma(0.98), 15),
        (1e-4, 0.4, annual_to_gamma(0.90), 8),
        (0.3, 0.02, 1.0, 5),
    ]
    for alpha, eps, gamma, epochs in cases:
        left = ql.QLearning(df.copy(), alpha, exploration_prob=eps, gamma=gamma, epochs=epochs)
        right = ql.QLearning(df.copy(), alpha, exploration_prob=eps, gamma=gamma, epochs=epochs)
        left.train(seed=SEED)
        train_fast(right, seed=SEED)
        gap = float(np.max(np.abs(left.q_table - right.q_table)))
        print(f"check alpha={alpha:g} eps={eps:g} gamma={gamma:.6f} epochs={epochs} maxabs={gap:.3e}")
        if gap > 1e-12:
            raise SystemExit(f"fast trainer diverged (maxabs={gap})")
    print("fast trainer matches QLearning.train")


def shard_rows(shard: int, shards: int) -> list[dict]:
    rows = configurations()
    return [row for index, row in enumerate(rows) if index % shards == shard]


def run_shard(
    df: pd.DataFrame,
    shard: int,
    shards: int,
    out_dir: Path,
    symbol: str,
    seed: int = SEED,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shard-{shard}-seed-{seed}.csv"
    rows = shard_rows(shard, shards)
    done = set()
    if path.exists():
        previous = pd.read_csv(path)
        for record in previous.itertuples(index=False):
            done.add((record.alpha, record.exploration_prob, record.annual_retained, int(record.seed)))
    pending = [
        row
        for row in rows
        if (row["alpha"], row["exploration_prob"], row["annual_retained"], seed) not in done
    ]
    print(f"shard {shard}/{shards}: {len(pending)} pending, {len(done)} already saved", flush=True)
    wrote_header = path.exists() and path.stat().st_size > 0
    for offset, spec in enumerate(pending, start=1):
        result = run_one(df, spec, symbol, seed=seed)
        frame = pd.DataFrame([result])
        frame.to_csv(path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        print(
            f"shard {shard} {offset}/{len(pending)} "
            f"alpha={result['alpha']:.5f} eps={result['exploration_prob']:.2f} "
            f"annual={result['annual_retained']:.2f} "
            f"train={result['train']:.4f} test={result['test']:.4f} "
            f"({result['seconds']:.1f}s)",
            flush=True,
        )
    return path


def load_results(out_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(out_dir.glob("shard-*.csv"))]
    if not frames:
        raise FileNotFoundError(f"no shard csv in {out_dir}")
    results = pd.concat(frames, ignore_index=True)
    results = results.drop_duplicates(
        ["alpha", "exploration_prob", "annual_retained", "seed"], keep="last"
    )
    return results.sort_values(
        ["test", "train"], ascending=False
    ).reset_index(drop=True)


def _agent_for(df: pd.DataFrame, row) -> ql.QLearning:
    agent = ql.QLearning(
        df,
        float(row.alpha),
        exploration_prob=float(row.exploration_prob),
        gamma=float(row.gamma),
        epochs=EPOCHS,
    )
    train_fast(agent, seed=int(row.seed) if hasattr(row, "seed") else SEED)
    return agent


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results.groupby(["alpha", "exploration_prob", "annual_retained", "gamma"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            train_mean=("train", "mean"),
            train_std=("train", "std"),
            test_mean=("test", "mean"),
            test_std=("test", "std"),
            test_median=("test", "median"),
            test_min=("test", "min"),
            test_max=("test", "max"),
            train_stock=("train_stock", "first"),
            test_stock=("test_stock", "first"),
        )
    )
    grouped["test_excess"] = grouped["test_mean"] - grouped["test_stock"]
    return grouped.sort_values(
        ["test_mean", "test_median", "train_mean"], ascending=False
    ).reset_index(drop=True)


def _long_fraction(agent: ql.QLearning) -> tuple[float, float]:
    """Share of steps the greedy policy is long, on train and on test."""
    last_train, first_test = agent._split_bounds()
    prices, bins, _ = _prepare(agent)
    n_bins = ql.MARKET_BINS

    def walk(start: int, last: int) -> float:
        position = ql.FLAT
        t = start
        long_steps = 0
        steps = 0
        while True:
            state_id = position * n_bins + int(bins[t])
            legal = [ql.BUY, ql.HOLD] if position == ql.FLAT else [ql.HOLD, ql.SELL]
            v0 = agent.q_table[state_id, legal[0]]
            action = legal[0] if v0 >= agent.q_table[state_id, legal[1]] else legal[1]
            next_t = t + 1
            if next_t > last or next_t >= len(prices):
                break
            if action == ql.HOLD:
                next_position = position
            elif position == ql.FLAT:
                next_position = ql.LONG
            else:
                next_position = ql.FLAT
            if next_position == ql.LONG:
                long_steps += 1
            steps += 1
            position = next_position
            t = next_t
        return long_steps / steps if steps else float("nan")

    return walk(ql.WINDOW, last_train), walk(first_test, len(prices) - 1)


def plot_results(df: pd.DataFrame, results: pd.DataFrame, fig_dir: Path, symbol: str) -> pd.DataFrame:
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(results)
    summary.to_csv(fig_dir / "q-learning-grid-summary.csv", index=False)
    best = summary.iloc[0]
    stock_test = float(best.test_stock)
    stock_train = float(best.train_stock)
    values = summary["test_mean"].to_numpy()
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    annuals = list(ANNUAL_RETAINED)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for ax, annual in zip(axes.ravel(), annuals):
        block = summary[np.isclose(summary["annual_retained"], annual)]
        pivot = block.pivot(index="exploration_prob", columns="alpha", values="test_mean")
        pivot = pivot.reindex(index=list(EXPLORATION), columns=list(ALPHAS))
        image = ax.imshow(
            pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
        )
        ax.set_xticks(range(len(ALPHAS)))
        ax.set_xticklabels([f"{alpha:g}" for alpha in ALPHAS], rotation=45, ha="right")
        ax.set_yticks(range(len(EXPLORATION)))
        ax.set_yticklabels([f"{eps:g}" for eps in EXPLORATION])
        ax.set_title(f"annual retained {annual:.2f}  (gamma={annual_to_gamma(annual):.6f})")
        ax.set_xlabel("alpha")
        ax.set_ylabel("exploration probability")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="mean test growth of $1")
    fig.suptitle(f"{symbol} Q-learning mean test growth across seeds")
    fig.tight_layout()
    heatmap_path = fig_dir / "q-learning-test-heatmap.png"
    fig.savefig(heatmap_path, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    slices = [(0.1, 0.98, {"marker": "o", "label": "eps=0.10, annual=0.98"})]
    if not (
        np.isclose(best.exploration_prob, 0.1) and np.isclose(best.annual_retained, 0.98)
    ):
        slices.append(
            (
                float(best.exploration_prob),
                float(best.annual_retained),
                {
                    "marker": "s",
                    "color": "black",
                    "label": (
                        f"best slice eps={float(best.exploration_prob):.2f}, "
                        f"annual={float(best.annual_retained):.2f}"
                    ),
                },
            )
        )
    for eps, annual, style in slices:
        block = summary[
            np.isclose(summary["annual_retained"], annual)
            & np.isclose(summary["exploration_prob"], eps)
        ].sort_values("alpha")
        ax.plot(block["alpha"], block["test_mean"], **style)
        ax.fill_between(
            block["alpha"],
            block["test_min"],
            block["test_max"],
            alpha=0.15,
            color=style.get("color"),
        )
    ax.axhline(stock_test, color="tab:blue", linestyle="--", label="buy and hold test")
    ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("test growth of $1")
    ax.set_title(f"{symbol} test growth across alpha (band is min to max seed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    slice_path = fig_dir / "q-learning-alpha-slice.png"
    fig.savefig(slice_path, dpi=140)
    plt.close(fig)

    # Equity of every seed at the best parameter combination.
    best_rows = results[
        np.isclose(results["alpha"], best.alpha)
        & np.isclose(results["exploration_prob"], best.exploration_prob)
        & np.isclose(results["annual_retained"], best.annual_retained)
    ].sort_values("seed")
    best_curves = []
    for row in best_rows.itertuples(index=False):
        best_curves.append(_quiet_curve(_agent_for(df, row), symbol))
    base = best_curves[0]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axvspan(base["first_test"], base["last"], color="0.85", label="test", zorder=0)
    ax.plot(base["dates"], base["stock"], label="stock", color="tab:blue", linewidth=2)
    for curve, row in zip(best_curves, best_rows.itertuples(index=False)):
        ax.plot(
            curve["dates"],
            curve["policy"],
            label=f"seed {int(row.seed)} test {row.test:.3f}",
            linewidth=1.2,
        )
    ax.axvline(pd.Timestamp(CUTOFF), color="0.4", linestyle="--", linewidth=1)
    ax.set_title(
        f"{symbol} Q-learning best mean test  "
        f"alpha={best.alpha:g}  eps={best.exploration_prob:g}  "
        f"annual={best.annual_retained:.2f}"
    )
    ax.set_ylabel("growth of $1")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    fig.tight_layout()
    equity_path = fig_dir / "q-learning-equity-best.png"
    fig.savefig(equity_path, dpi=140)
    plt.close(fig)

    # Original overlay, seed 0, at the script's exploration rate and discount.
    family_alphas = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.05, 0.1, 0.3]
    family = results[
        np.isclose(results["exploration_prob"], 0.1)
        & np.isclose(results["annual_retained"], 0.98)
        & np.isclose(results["seed"], SEED)
        & results["alpha"].isin(family_alphas)
    ].sort_values("alpha")
    family_curves = []
    for row in family.itertuples(index=False):
        family_curves.append(_quiet_curve(_agent_for(df, row), symbol))
    base = family_curves[0]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axvspan(base["first_test"], base["last"], color="0.85", label="test", zorder=0)
    ax.plot(base["dates"], base["stock"], label="stock", color="tab:blue")
    for curve in family_curves:
        ax.plot(curve["dates"], curve["policy"], label=f"alpha={curve['alpha']:g}")
    ax.axvline(pd.Timestamp(CUTOFF), color="0.4", linestyle="--", linewidth=1)
    ax.set_title(f"{symbol} Q-learning, seed 0, exploration=0.10, annual retained=0.98")
    ax.set_ylabel("growth of $1")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    family_path = fig_dir / "q-learning-equity-alpha.png"
    fig.savefig(family_path, dpi=140)
    plt.close(fig)

    probe = best_rows.iloc[0]
    agent = _agent_for(df, probe)
    train_long, test_long = _long_fraction(agent)
    print(f"train stock {stock_train:.4f}  test stock {stock_test:.4f}")
    print(
        f"best mean test {best.test_mean:.4f} ± {best.test_std:.4f} "
        f"(seeds={int(best.seeds)}, min={best.test_min:.4f}, max={best.test_max:.4f})"
    )
    print(
        f"alpha={best.alpha:g} exploration={best.exploration_prob:g} "
        f"annual_retained={best.annual_retained:.2f} gamma={best.gamma:.6f}"
    )
    print(
        f"seed {int(probe.seed)} time long: train {train_long:.1%}  test {test_long:.1%}"
    )
    print(summary.head(15).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"saved {heatmap_path}")
    print(f"saved {slice_path}")
    print(f"saved {equity_path}")
    print(f"saved {family_path}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Q-learning parameter grid")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("/tmp/q-learning-grid"))
    parser.add_argument("--fig", type=Path, default=Path("/tmp/q-learning-grid/figures"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_frame(args.symbol)
    if args.check:
        check_equivalence(df)
        return
    if args.plot or args.summarize:
        results = load_results(args.out)
        summary = summarize_results(results)
        print(f"{len(results)} runs, {summary['seeds'].max():.0f} seeds max")
        print(
            summary.head(20).to_string(
                index=False, float_format=lambda value: f"{value:.4f}"
            )
        )
        if args.plot:
            plot_results(df, results, args.fig, args.symbol)
        return
    run_shard(df, args.shard, args.shards, args.out, args.symbol, seed=args.seed)


if __name__ == "__main__":
    main()
