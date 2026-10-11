"""Train one Dyna-Q setup on every STARTING_STOCKS name except SPY.

Downloads and cleans only those symbols, then saves an equity chart per name,
a test-growth bar chart, and a CSV of compounded growth from visualize().
"""

from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

_SPEC = importlib.util.spec_from_file_location(
    "dyna_q_agent", Path(__file__).with_name("dyna-q.py")
)
dyna_q = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dyna_q)

from src.data.yfinance_clean import clean_data
from src.data.yfinance_dl_raw import download_data
from src.ml.config import (
    CLEAN_DIR,
    CUTOFF,
    DOWNLOAD_PAUSE_S,
    OHLCV_COLUMNS,
    RAW_DIR,
    STARTING_STOCKS,
)

ALPHA = 0.0001
PLANNING_STEPS = 5
SEED = 0
FIGURE_DIR = Path(__file__).resolve().parent / "figures"
CSV_PATH = FIGURE_DIR / "dyna-q-cross-section.csv"
CSV_COLUMNS = (
    "symbol",
    "alpha",
    "planning_steps",
    "train_policy",
    "test_policy",
    "train_stock",
    "test_stock",
    "test_excess",
)
EXPECTED_SYMBOLS = [
    "AAPL",
    "MSFT",
    "GOOG",
    "AMZN",
    "META",
    "NVDA",
    "IBM",
    "JPM",
    "XOM",
    "KO",
    "PG",
    "UNH",
    "JNJ",
    "CAT",
    "UNP",
    "V",
    "NEE",
    "LIN",
    "PLD",
]


def symbols_except_spy() -> list[str]:
    symbols = [symbol for symbol in STARTING_STOCKS if symbol != "SPY"]
    if symbols != EXPECTED_SYMBOLS:
        raise SystemExit(f"STARTING_STOCKS except SPY is {symbols}, expected {EXPECTED_SYMBOLS}")
    return symbols


def frame_with_date_column(raw: pd.DataFrame) -> pd.DataFrame:
    """Yahoo often returns Date as the index. clean_data selects a Date column."""
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(-1)
    if "Date" not in frame.columns:
        frame = frame.reset_index()
    if "Date" not in frame.columns and "index" in frame.columns:
        frame = frame.rename(columns={"index": "Date"})
    if "Date" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "Date"})
    return frame


def download_and_clean(symbol: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            raw = download_data(symbol)
            if raw is None or len(raw) == 0:
                raise RuntimeError(f"{symbol}: download returned no rows")
            cleaned = clean_data(frame_with_date_column(raw))
            missing = [column for column in OHLCV_COLUMNS if column not in cleaned.columns]
            if missing:
                raise RuntimeError(f"{symbol}: cleaned frame missing {missing}")
            if cleaned.empty:
                raise RuntimeError(f"{symbol}: cleaned frame is empty")
            CLEAN_DIR.mkdir(parents=True, exist_ok=True)
            cleaned.to_csv(CLEAN_DIR / f"{symbol}.csv", index=False)
            prices = pd.read_csv(CLEAN_DIR / f"{symbol}.csv", parse_dates=["Date"])
            if "Date" not in prices.columns:
                raise RuntimeError(f"{symbol}: cleaned csv has no Date column")
            return prices.sort_values("Date").dropna().reset_index(drop=True)
        except Exception as exc:
            last_error = exc
            print(f"{symbol} download attempt {attempt} failed: {exc}", flush=True)
            time.sleep(DOWNLOAD_PAUSE_S * attempt)
    assert last_error is not None
    raise last_error


def save_equity(symbol: str, curve: dict) -> Path:
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.axvspan(curve["first_test"], curve["last"], color="0.85", label="test", zorder=0)
    axis.plot(curve["dates"], curve["stock"], label="stock", color="tab:blue")
    axis.plot(curve["dates"], curve["policy"], label="policy", color="tab:orange")
    axis.axvline(pd.Timestamp(CUTOFF), color="0.4", linestyle="--", linewidth=1)
    axis.set_title(f"{symbol} Dyna-Q alpha=0.0001")
    axis.set_ylabel("growth of $1")
    axis.set_xlabel("Date")
    axis.legend()
    figure.tight_layout()
    path = FIGURE_DIR / f"dyna-q-equity-{symbol}.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def save_comparison(rows: list[dict]) -> Path:
    ordered = sorted(rows, key=lambda row: (row["test_excess"], row["symbol"]))
    positions = list(range(len(ordered)))
    width = 0.4
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.bar(
        [position - width / 2 for position in positions],
        [row["test_policy"] for row in ordered],
        width,
        label="test policy",
        color="tab:orange",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        [row["test_stock"] for row in ordered],
        width,
        label="test stock",
        color="tab:blue",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels([row["symbol"] for row in ordered], rotation=45, ha="right")
    axis.set_xlabel("symbol")
    axis.set_ylabel("growth of $1")
    axis.set_title("Dyna-Q test growth of $1 at alpha 0.0001")
    axis.legend()
    figure.tight_layout()
    path = FIGURE_DIR / "dyna-q-test-comparison.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def result_row(symbol: str, agent: dyna_q.DynaQ, curve: dict) -> dict:
    return {
        "symbol": symbol,
        "alpha": curve["alpha"],
        "planning_steps": agent.planning_steps,
        "train_policy": curve["train_policy"],
        "test_policy": curve["test_policy"],
        "train_stock": curve["train_stock"],
        "test_stock": curve["test_stock"],
        "test_excess": curve["test_policy"] - curve["test_stock"],
    }


def train_symbol(symbol: str, prices: pd.DataFrame) -> dict:
    agent = dyna_q.DynaQ(prices, alpha=ALPHA, planning_steps=PLANNING_STEPS)
    if agent.epochs != dyna_q.EPOCHS or agent.epochs != 1000:
        raise RuntimeError(f"{symbol}: epochs={agent.epochs}, expected 1000")
    if agent.alpha != ALPHA or agent.planning_steps != PLANNING_STEPS:
        raise RuntimeError(f"{symbol}: alpha/planning drifted")
    if agent.exploration_start != 0.1 or agent.exploration_end != 0.05:
        raise RuntimeError(
            f"{symbol}: exploration {agent.exploration_start} to {agent.exploration_end}"
        )
    print(
        f"\nTraining {symbol} for {agent.epochs} epochs, "
        f"alpha={agent.alpha:.4f}, planning_steps={agent.planning_steps}, "
        f"exploration {agent.exploration_start:g} to {agent.exploration_end:g}, "
        f"gamma={agent.gamma:.10f}, seed={SEED}, cutoff {CUTOFF}",
        flush=True,
    )
    started = time.perf_counter()
    agent.train(seed=SEED)
    agent.evaluate()
    curve = agent.visualize(symbol)
    path = save_equity(symbol, curve)
    elapsed = time.perf_counter() - started
    row = result_row(symbol, agent, curve)
    print(
        f"{symbol} test_policy={row['test_policy']:.4f} "
        f"test_stock={row['test_stock']:.4f} "
        f"test_excess={row['test_excess']:.4f} "
        f"({elapsed:.1f}s) saved {path}",
        flush=True,
    )
    return row


def write_csv(rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: (row["test_excess"], row["symbol"]))
    pd.DataFrame(ordered, columns=list(CSV_COLUMNS)).to_csv(CSV_PATH, index=False)


def main() -> None:
    started = time.perf_counter()
    symbols = symbols_except_spy()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failures: list[str] = []
    for index, symbol in enumerate(symbols):
        if index:
            time.sleep(DOWNLOAD_PAUSE_S)
        print(f"\n=== {symbol} ({index + 1}/{len(symbols)}) ===", flush=True)
        try:
            prices = download_and_clean(symbol)
        except Exception:
            message = traceback.format_exc()
            failures.append(f"{symbol} download failed\n{message}")
            print(message, flush=True)
            continue
        try:
            rows.append(train_symbol(symbol, prices))
            write_csv(rows)
        except Exception:
            message = traceback.format_exc()
            failures.append(f"{symbol} training failed\n{message}")
            print(message, flush=True)

    comparison_path = None
    if rows:
        comparison_path = save_comparison(rows)
        write_csv(rows)
        print(f"saved {comparison_path}", flush=True)
        print(f"saved {CSV_PATH}", flush=True)
        table = pd.read_csv(CSV_PATH)
        print(table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        ahead = table.loc[table["test_excess"] > 0, "symbol"].tolist()
        print(f"test_excess > 0: {len(ahead)}/{len(table)} {ahead}", flush=True)

    elapsed = time.perf_counter() - started
    print(f"wall_clock_seconds {elapsed:.1f}", flush=True)
    if failures:
        print(f"failures {len(failures)}", flush=True)
        for failure in failures:
            print(failure, flush=True)
    else:
        print("failures 0", flush=True)


if __name__ == "__main__":
    main()
