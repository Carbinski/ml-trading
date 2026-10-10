"""Score Q-learning across alpha, exploration probability, and annual discount.

`--shard i --shards N --seed S` runs one slice. `--check` compares the array
trainer with `QLearning.train`.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
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
    "q_learning_agent", Path(__file__).with_name("q-learning.py")
)
q_learning = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(q_learning)

from src.ml.config import CLEAN_DIR, CUTOFF

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
EXPLORATION_PROBABILITIES = (0.02, 0.05, 0.1, 0.2, 0.4)
ANNUAL_RETAINED_VALUES = (0.90, 0.95, 0.98, 1.0)
SEED = q_learning.SEED
EPOCHS = q_learning.EPOCHS
SYMBOL = "SPY"
RESULT_COLUMNS = (
    "alpha",
    "exploration_prob",
    "annual_retained",
    "seed",
)


def daily_gamma(annual_retained: float) -> float:
    return float(annual_retained ** (1.0 / q_learning.TRADING_DAYS_PER_YEAR))


def parameter_grid() -> list[dict]:
    grid = []
    for annual_retained in ANNUAL_RETAINED_VALUES:
        gamma = daily_gamma(annual_retained)
        for alpha in ALPHAS:
            for exploration_probability in EXPLORATION_PROBABILITIES:
                grid.append(
                    {
                        "alpha": float(alpha),
                        "exploration_prob": float(exploration_probability),
                        "annual_retained": float(annual_retained),
                        "gamma": gamma,
                    }
                )
    return grid


def prices_and_market_bins(agent: q_learning.QLearning):
    prices = agent.df["Adj Close"].to_numpy(dtype=float)
    ratio = agent.df["Price_to_SMA_Ratio"].to_numpy(dtype=float)
    market_bin = np.full(len(prices), -1, dtype=np.int16)
    market_bin[q_learning.WINDOW :] = np.digitize(
        ratio[q_learning.WINDOW :], q_learning.EDGES
    ).astype(np.int16)
    dates = pd.to_datetime(agent.df["Date"]).to_numpy()
    last_train_bar = int(np.flatnonzero(dates < np.datetime64(CUTOFF))[-1])
    return prices, market_bin, last_train_bar


def greedy_action(q_table: np.ndarray, state_id: int, legal_actions: list[int]) -> int:
    best_action = legal_actions[0]
    best_value = q_table[state_id, best_action]
    for action in legal_actions[1:]:
        value = q_table[state_id, action]
        if value > best_value:
            best_action = action
            best_value = value
    return best_action


def position_after(position: int, action: int) -> int:
    if action == q_learning.HOLD:
        return position
    if position == q_learning.FLAT and action == q_learning.BUY:
        return q_learning.LONG
    if position == q_learning.LONG and action == q_learning.SELL:
        return q_learning.FLAT
    raise ValueError(f"illegal action {action} from position {position}")


def reward_for_position(prices: np.ndarray, bar: int, next_bar: int, position: int) -> float:
    if position != q_learning.LONG:
        return 0.0
    return float(prices[next_bar] / prices[bar] - 1.0)


def train_from_arrays(agent: q_learning.QLearning, seed: int = SEED) -> None:
    """Same random draws and Q updates as QLearning.train."""
    np.random.seed(seed)
    prices, market_bin, last_train_bar = prices_and_market_bins(agent)
    q_table = agent.q_table
    alpha = agent.alpha
    gamma = agent.gamma
    exploration_probability = agent.exploration_prob
    weight_on_previous_q = 1.0 - alpha
    flat_actions = [q_learning.BUY, q_learning.HOLD]
    long_actions = [q_learning.HOLD, q_learning.SELL]
    bar_count = len(prices)

    for _epoch in range(agent.epochs):
        position = q_learning.FLAT
        bar = q_learning.WINDOW
        if bar > last_train_bar:
            break
        state_id = int(market_bin[bar])
        while True:
            legal_actions = flat_actions if position == q_learning.FLAT else long_actions
            if np.random.rand() < exploration_probability:
                action = int(np.random.choice(legal_actions))
            else:
                action = greedy_action(q_table, state_id, legal_actions)
            next_bar = bar + 1
            if next_bar >= bar_count or next_bar > last_train_bar:
                break
            next_position = position_after(position, action)
            next_state_id = next_position * q_learning.MARKET_BINS + int(market_bin[next_bar])
            reward = reward_for_position(prices, bar, next_bar, next_position)
            if next_bar == last_train_bar:
                target = reward
            else:
                next_legal_actions = (
                    flat_actions if next_position == q_learning.FLAT else long_actions
                )
                future_action = greedy_action(q_table, next_state_id, next_legal_actions)
                target = reward + gamma * q_table[next_state_id, future_action]
            q_table[state_id, action] = (
                weight_on_previous_q * q_table[state_id, action] + alpha * target
            )
            state_id = next_state_id
            position = next_position
            bar = next_bar
            if next_bar == last_train_bar:
                break


def load_prices(symbol: str) -> pd.DataFrame:
    prices = pd.read_csv(CLEAN_DIR / f"{symbol}.csv", parse_dates=["Date"])
    return prices.sort_values("Date").dropna().reset_index(drop=True)


def growth_curve(agent: q_learning.QLearning, symbol: str) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        return agent.visualize(symbol)


def score_parameters(prices: pd.DataFrame, parameters: dict, symbol: str, seed: int = SEED) -> dict:
    agent = q_learning.QLearning(
        prices,
        parameters["alpha"],
        exploration_prob=parameters["exploration_prob"],
        gamma=parameters["gamma"],
        epochs=EPOCHS,
    )
    started = time.perf_counter()
    train_from_arrays(agent, seed=seed)
    elapsed_seconds = time.perf_counter() - started
    growth = growth_curve(agent, symbol)
    return {
        "alpha": parameters["alpha"],
        "exploration_prob": parameters["exploration_prob"],
        "annual_retained": parameters["annual_retained"],
        "gamma": parameters["gamma"],
        "seed": seed,
        "epochs": EPOCHS,
        "train": growth["train_policy"],
        "test": growth["test_policy"],
        "train_stock": growth["train_stock"],
        "test_stock": growth["test_stock"],
        "seconds": elapsed_seconds,
    }


def assert_array_trainer_matches_q_learning(prices: pd.DataFrame) -> None:
    cases = [
        (0.01, 0.1, daily_gamma(0.98), 15),
        (1e-4, 0.4, daily_gamma(0.90), 8),
        (0.3, 0.02, 1.0, 5),
    ]
    for alpha, exploration_probability, gamma, epochs in cases:
        original = q_learning.QLearning(
            prices.copy(),
            alpha,
            exploration_prob=exploration_probability,
            gamma=gamma,
            epochs=epochs,
        )
        arrays = q_learning.QLearning(
            prices.copy(),
            alpha,
            exploration_prob=exploration_probability,
            gamma=gamma,
            epochs=epochs,
        )
        original.train(seed=SEED)
        train_from_arrays(arrays, seed=SEED)
        max_abs_difference = float(np.max(np.abs(original.q_table - arrays.q_table)))
        print(
            f"check alpha={alpha:g} exploration={exploration_probability:g} "
            f"gamma={gamma:.6f} epochs={epochs} maxabs={max_abs_difference:.3e}"
        )
        if max_abs_difference > 1e-12:
            raise SystemExit(f"array trainer diverged (maxabs={max_abs_difference})")
    print("array trainer matches QLearning.train")


def parameters_for_shard(shard: int, shard_count: int) -> list[dict]:
    return [
        parameters
        for index, parameters in enumerate(parameter_grid())
        if index % shard_count == shard
    ]


def parameter_identity(alpha, exploration_probability, annual_retained, seed) -> tuple:
    return (float(alpha), float(exploration_probability), float(annual_retained), int(seed))


def score_shard(
    prices: pd.DataFrame,
    shard: int,
    shard_count: int,
    out_dir: Path,
    symbol: str,
    seed: int = SEED,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"shard-{shard}-seed-{seed}.csv"
    already_scored = set()
    if csv_path.exists():
        previous = pd.read_csv(csv_path)
        for row in previous.itertuples(index=False):
            already_scored.add(
                parameter_identity(row.alpha, row.exploration_prob, row.annual_retained, row.seed)
            )
    remaining = [
        parameters
        for parameters in parameters_for_shard(shard, shard_count)
        if parameter_identity(
            parameters["alpha"],
            parameters["exploration_prob"],
            parameters["annual_retained"],
            seed,
        )
        not in already_scored
    ]
    print(
        f"shard {shard}/{shard_count}: {len(remaining)} remaining, "
        f"{len(already_scored)} already scored",
        flush=True,
    )
    csv_has_header = csv_path.exists() and csv_path.stat().st_size > 0
    for completed_count, parameters in enumerate(remaining, start=1):
        result = score_parameters(prices, parameters, symbol, seed=seed)
        pd.DataFrame([result]).to_csv(csv_path, mode="a", header=not csv_has_header, index=False)
        csv_has_header = True
        print(
            f"shard {shard} {completed_count}/{len(remaining)} "
            f"alpha={result['alpha']:.5f} exploration={result['exploration_prob']:.2f} "
            f"annual={result['annual_retained']:.2f} "
            f"train={result['train']:.4f} test={result['test']:.4f} "
            f"({result['seconds']:.1f}s)",
            flush=True,
        )
    return csv_path


def read_shard_results(out_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(out_dir.glob("shard-*.csv"))]
    if not frames:
        raise FileNotFoundError(f"no shard csv in {out_dir}")
    results = pd.concat(frames, ignore_index=True)
    results = results.drop_duplicates(list(RESULT_COLUMNS), keep="last")
    return results.sort_values(["test", "train"], ascending=False).reset_index(drop=True)


def trained_agent(prices: pd.DataFrame, row) -> q_learning.QLearning:
    agent = q_learning.QLearning(
        prices,
        float(row.alpha),
        exploration_prob=float(row.exploration_prob),
        gamma=float(row.gamma),
        epochs=EPOCHS,
    )
    seed = int(row.seed) if hasattr(row, "seed") else SEED
    train_from_arrays(agent, seed=seed)
    return agent


def mean_growth_by_parameters(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(
            ["alpha", "exploration_prob", "annual_retained", "gamma"], as_index=False
        )
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
    summary["test_excess"] = summary["test_mean"] - summary["test_stock"]
    return summary.sort_values(
        ["test_mean", "test_median", "train_mean"], ascending=False
    ).reset_index(drop=True)


def fraction_of_bars_held_long(agent: q_learning.QLearning) -> tuple[float, float]:
    last_train_bar, first_test_bar = agent._split_bounds()
    prices, market_bin, _last_train_bar = prices_and_market_bins(agent)

    def fraction_between(start_bar: int, end_bar: int) -> float:
        position = q_learning.FLAT
        bar = start_bar
        long_bars = 0
        bars = 0
        while True:
            state_id = position * q_learning.MARKET_BINS + int(market_bin[bar])
            legal_actions = (
                [q_learning.BUY, q_learning.HOLD]
                if position == q_learning.FLAT
                else [q_learning.HOLD, q_learning.SELL]
            )
            action = greedy_action(agent.q_table, state_id, legal_actions)
            next_bar = bar + 1
            if next_bar > end_bar or next_bar >= len(prices):
                break
            position = position_after(position, action)
            if position == q_learning.LONG:
                long_bars += 1
            bars += 1
            bar = next_bar
        return long_bars / bars if bars else float("nan")

    train_fraction = fraction_between(q_learning.WINDOW, last_train_bar)
    test_fraction = fraction_between(first_test_bar, len(prices) - 1)
    return train_fraction, test_fraction


def _style_heatmap(symbol: str, summary: pd.DataFrame, figure_dir: Path) -> Path:
    test_growth = summary["test_mean"].to_numpy()
    lowest = float(np.nanmin(test_growth))
    highest = float(np.nanmax(test_growth))
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for axis, annual_retained in zip(axes.ravel(), ANNUAL_RETAINED_VALUES):
        annual_rows = summary[np.isclose(summary["annual_retained"], annual_retained)]
        pivot = annual_rows.pivot(
            index="exploration_prob", columns="alpha", values="test_mean"
        )
        pivot = pivot.reindex(
            index=list(EXPLORATION_PROBABILITIES), columns=list(ALPHAS)
        )
        image = axis.imshow(
            pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=lowest, vmax=highest
        )
        axis.set_xticks(range(len(ALPHAS)))
        axis.set_xticklabels([f"{alpha:g}" for alpha in ALPHAS], rotation=45, ha="right")
        axis.set_yticks(range(len(EXPLORATION_PROBABILITIES)))
        axis.set_yticklabels([f"{probability:g}" for probability in EXPLORATION_PROBABILITIES])
        axis.set_title(
            f"annual retained {annual_retained:.2f}  "
            f"(gamma={daily_gamma(annual_retained):.6f})"
        )
        axis.set_xlabel("alpha")
        axis.set_ylabel("exploration probability")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="mean test growth of $1")
    figure.suptitle(f"{symbol} Q-learning mean test growth across seeds")
    figure.tight_layout()
    path = figure_dir / "q-learning-test-heatmap.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def _alpha_slice_styles(best) -> list[tuple]:
    styles = [(0.1, 0.98, {"marker": "o", "label": "exploration=0.10, annual=0.98"})]
    best_exploration = float(best.exploration_prob)
    best_annual = float(best.annual_retained)
    if not (np.isclose(best_exploration, 0.1) and np.isclose(best_annual, 0.98)):
        styles.append(
            (
                best_exploration,
                best_annual,
                {
                    "marker": "s",
                    "color": "black",
                    "label": (
                        f"best slice exploration={best_exploration:.2f}, "
                        f"annual={best_annual:.2f}"
                    ),
                },
            )
        )
    return styles


def _style_alpha_slice(symbol: str, summary: pd.DataFrame, figure_dir: Path, buy_and_hold_test: float) -> Path:
    best = summary.iloc[0]
    figure, axis = plt.subplots(figsize=(10, 5))
    for exploration_probability, annual_retained, style in _alpha_slice_styles(best):
        slice_rows = summary[
            np.isclose(summary["annual_retained"], annual_retained)
            & np.isclose(summary["exploration_prob"], exploration_probability)
        ].sort_values("alpha")
        axis.plot(slice_rows["alpha"], slice_rows["test_mean"], **style)
        axis.fill_between(
            slice_rows["alpha"],
            slice_rows["test_min"],
            slice_rows["test_max"],
            alpha=0.15,
            color=style.get("color"),
        )
    axis.axhline(buy_and_hold_test, color="tab:blue", linestyle="--", label="buy and hold test")
    axis.set_xscale("log")
    axis.set_xlabel("alpha")
    axis.set_ylabel("test growth of $1")
    axis.set_title(f"{symbol} test growth across alpha (band is lowest to highest seed)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = figure_dir / "q-learning-alpha-slice.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def _equity_curves_for_rows(prices: pd.DataFrame, rows: pd.DataFrame, symbol: str) -> list[dict]:
    return [growth_curve(trained_agent(prices, row), symbol) for row in rows.itertuples(index=False)]


def _style_best_equity(symbol: str, prices: pd.DataFrame, results: pd.DataFrame, best, figure_dir: Path) -> Path:
    matching_seeds = results[
        np.isclose(results["alpha"], best.alpha)
        & np.isclose(results["exploration_prob"], best.exploration_prob)
        & np.isclose(results["annual_retained"], best.annual_retained)
    ].sort_values("seed")
    curves = _equity_curves_for_rows(prices, matching_seeds, symbol)
    first_curve = curves[0]
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.axvspan(first_curve["first_test"], first_curve["last"], color="0.85", label="test", zorder=0)
    axis.plot(first_curve["dates"], first_curve["stock"], label="stock", color="tab:blue", linewidth=2)
    for curve, row in zip(curves, matching_seeds.itertuples(index=False)):
        axis.plot(
            curve["dates"],
            curve["policy"],
            label=f"seed {int(row.seed)} test {row.test:.3f}",
            linewidth=1.2,
        )
    axis.axvline(pd.Timestamp(CUTOFF), color="0.4", linestyle="--", linewidth=1)
    axis.set_title(
        f"{symbol} Q-learning best mean test  "
        f"alpha={best.alpha:g}  exploration={best.exploration_prob:g}  "
        f"annual={best.annual_retained:.2f}"
    )
    axis.set_ylabel("growth of $1")
    axis.set_xlabel("Date")
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = figure_dir / "q-learning-equity-best.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def _style_default_alpha_equity(symbol: str, prices: pd.DataFrame, results: pd.DataFrame, figure_dir: Path) -> Path:
    shown_alphas = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.05, 0.1, 0.3]
    seed_zero_defaults = results[
        np.isclose(results["exploration_prob"], 0.1)
        & np.isclose(results["annual_retained"], 0.98)
        & np.isclose(results["seed"], SEED)
        & results["alpha"].isin(shown_alphas)
    ].sort_values("alpha")
    curves = _equity_curves_for_rows(prices, seed_zero_defaults, symbol)
    first_curve = curves[0]
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.axvspan(first_curve["first_test"], first_curve["last"], color="0.85", label="test", zorder=0)
    axis.plot(first_curve["dates"], first_curve["stock"], label="stock", color="tab:blue")
    for curve in curves:
        axis.plot(curve["dates"], curve["policy"], label=f"alpha={curve['alpha']:g}")
    axis.axvline(pd.Timestamp(CUTOFF), color="0.4", linestyle="--", linewidth=1)
    axis.set_title(f"{symbol} Q-learning, seed 0, exploration=0.10, annual retained=0.98")
    axis.set_ylabel("growth of $1")
    axis.set_xlabel("Date")
    axis.legend()
    figure.tight_layout()
    path = figure_dir / "q-learning-equity-alpha.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def save_result_figures(
    prices: pd.DataFrame, results: pd.DataFrame, figure_dir: Path, symbol: str
) -> pd.DataFrame:
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary = mean_growth_by_parameters(results)
    summary.to_csv(figure_dir / "q-learning-grid-summary.csv", index=False)
    best = summary.iloc[0]
    buy_and_hold_test = float(best.test_stock)
    buy_and_hold_train = float(best.train_stock)
    heatmap_path = _style_heatmap(symbol, summary, figure_dir)
    alpha_slice_path = _style_alpha_slice(symbol, summary, figure_dir, buy_and_hold_test)
    best_equity_path = _style_best_equity(symbol, prices, results, best, figure_dir)
    alpha_equity_path = _style_default_alpha_equity(symbol, prices, results, figure_dir)

    first_seed = results[
        np.isclose(results["alpha"], best.alpha)
        & np.isclose(results["exploration_prob"], best.exploration_prob)
        & np.isclose(results["annual_retained"], best.annual_retained)
    ].sort_values("seed").iloc[0]
    agent = trained_agent(prices, first_seed)
    train_long, test_long = fraction_of_bars_held_long(agent)
    print(f"train stock {buy_and_hold_train:.4f}  test stock {buy_and_hold_test:.4f}")
    print(
        f"best mean test {best.test_mean:.4f} ± {best.test_std:.4f} "
        f"(seeds={int(best.seeds)}, min={best.test_min:.4f}, max={best.test_max:.4f})"
    )
    print(
        f"alpha={best.alpha:g} exploration={best.exploration_prob:g} "
        f"annual_retained={best.annual_retained:.2f} gamma={best.gamma:.6f}"
    )
    print(
        f"seed {int(first_seed.seed)} fraction long: "
        f"train {train_long:.1%}  test {test_long:.1%}"
    )
    print(summary.head(15).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    for path in (heatmap_path, alpha_slice_path, best_equity_path, alpha_equity_path):
        print(f"saved {path}")
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
    prices = load_prices(args.symbol)
    if args.check:
        assert_array_trainer_matches_q_learning(prices)
        return
    if args.plot or args.summarize:
        results = read_shard_results(args.out)
        summary = mean_growth_by_parameters(results)
        print(f"{len(results)} runs, {summary['seeds'].max():.0f} seeds max")
        print(summary.head(20).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        if args.plot:
            save_result_figures(prices, results, args.fig, args.symbol)
        return
    score_shard(prices, args.shard, args.shards, args.out, args.symbol, seed=args.seed)


if __name__ == "__main__":
    main()
