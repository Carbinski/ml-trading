from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
REGRESSION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REGRESSION_DIR))

import utility as ut
from src.ml.config import CLEAN_DIR as DATA_DIR, CUTOFF

LAG = 1
# Same fixed k set as the current return-target kNN closeout (v8).
N_NEIGHBORS_LIST = [15, 21, 51]

# Frozen from kNN v4/v5 before utility.py grew extra derived columns.
# Do not add Signed_Rel_Vol, Intraday, Vol_*, ATR_*, SMA_Slope, or
# Window_Close_Location — those were not in the original search pool.
FEATURE_SETS = {
    "v4": [
        "1_Day_Return",
        "Overnight",
        "Range",
        "Close Location",
        "Upper Wick",
        "Lower Wick",
    ],
    "v5": [
        "1_Day_Return",
        "Overnight",
        "Range",
        "Close Location",
        "Rel_Vol",
        "20_Day_Return",
        "Dist_From_SMA",
    ],
}

SPLIT_FEATURE_COLS = [
    "1_Day_Return",
    "Overnight",
    "Range",
    "Close Location",
    "Upper Wick",
    "Lower Wick",
    "Rel_Vol",
    "Shock_Vol",
    "5_Day_Return",
    "10_Day_Return",
    "20_Day_Return",
    "Dist_From_SMA",
]


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}Z] {message}", flush=True)


def load_stock_frames(data_dir: Path) -> dict[str, pd.DataFrame]:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No stock CSV files found in {data_dir}")
    frames: dict[str, pd.DataFrame] = {}
    for path in files:
        frame = pd.read_csv(path, parse_dates=["Date"])
        frames[path.stem] = frame.sort_values("Date").reset_index(drop=True)
    return frames


def train_model(X: pd.DataFrame, y: pd.Series, n_neighbors: int):
    model = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=n_neighbors),
    )
    model.fit(X, y)
    return model


def split_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    train_mean: float,
) -> dict[str, Any]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float).ravel()
    negative_rate = float((predicted < 0).mean())
    predicted = np.clip(predicted, a_min=0.0, a_max=None)
    mse = float(mean_squared_error(actual, predicted))
    mse_mean = float(
        mean_squared_error(actual, np.full_like(actual, train_mean, dtype=float))
    )
    pred_std = float(predicted.std())
    actual_std = float(actual.std())
    correlation = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if actual_std > 0 and pred_std > 0
        else None
    )
    return {
        "observations": int(len(actual)),
        "mse": mse,
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "mse_mean_baseline": mse_mean,
        "beat_mean_baseline": bool(mse < mse_mean),
        "prediction_std": pred_std,
        "actual_std": actual_std,
        "actual_mean": float(actual.mean()),
        "predicted_mean": float(predicted.mean()),
        "correlation": correlation,
        "negative_prediction_rate_before_clipping": negative_rate,
    }


def run_symbol(symbol: str, raw: pd.DataFrame) -> dict[str, Any]:
    df, _ = ut.process_OHLCV_all(raw.sort_values("Date").dropna())
    X_train, y_train, X_test, y_test = ut.split_data(
        df, SPLIT_FEATURE_COLS, CUTOFF, lag=LAG
    )
    y_train = y_train.abs()
    y_test = y_test.abs()
    train_mean = float(y_train.mean())
    recipes: dict[str, Any] = {}
    for recipe, features in FEATURE_SETS.items():
        cols = ut.lagged_cols(X_train, features)
        by_k: dict[str, Any] = {}
        for n_neighbors in N_NEIGHBORS_LIST:
            t0 = time.perf_counter()
            model = train_model(X_train[cols], y_train, n_neighbors)
            y_train_pred = model.predict(X_train[cols])
            y_test_pred = model.predict(X_test[cols])
            fit_s = time.perf_counter() - t0
            train = split_metrics(y_train, y_train_pred, train_mean)
            test = split_metrics(y_test, y_test_pred, train_mean)
            by_k[str(n_neighbors)] = {
                "n_neighbors": n_neighbors,
                "fit_seconds": fit_s,
                "train": train,
                "test": test,
            }
            log(
                f"    {symbol} {recipe} k={n_neighbors}: "
                f"train R2={train['r2']:+.4f}  test R2={test['r2']:+.4f}  "
                f"test corr={test['correlation'] if test['correlation'] is not None else float('nan'):+.3f}  "
                f"beat mean={test['beat_mean_baseline']}  "
                f"fit={fit_s:.2f}s"
            )
        recipes[recipe] = {"features": features, "by_k": by_k}
    return {
        "symbol": symbol,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_target_mean": train_mean,
        "test_target_mean": float(y_test.mean()),
        "recipes": recipes,
    }


def summarize_k(rows: list[dict[str, Any]], recipe: str, k: int) -> dict[str, Any]:
    per_symbol = []
    for row in rows:
        fit = row["recipes"][recipe]["by_k"][str(k)]
        train = fit["train"]
        test = fit["test"]
        per_symbol.append(
            {
                "symbol": row["symbol"],
                "n_neighbors": k,
                "n_train": row["n_train"],
                "n_test": row["n_test"],
                "train_mse": train["mse"],
                "train_mae": train["mae"],
                "train_r2": train["r2"],
                "test_mse": test["mse"],
                "test_mae": test["mae"],
                "test_r2": test["r2"],
                "test_mse_mean_baseline": test["mse_mean_baseline"],
                "test_beat_mean_baseline": test["beat_mean_baseline"],
                "test_correlation": test["correlation"],
                "test_actual_mean": test["actual_mean"],
                "test_predicted_mean": test["predicted_mean"],
                "test_negative_prediction_rate": test[
                    "negative_prediction_rate_before_clipping"
                ],
            }
        )
    frame = pd.DataFrame(per_symbol)
    correlations = frame["test_correlation"].dropna()
    return {
        "n_neighbors": k,
        "stock_count": int(len(frame)),
        "macro_train_mse": float(frame["train_mse"].mean()),
        "macro_train_mae": float(frame["train_mae"].mean()),
        "macro_train_r2": float(frame["train_r2"].mean()),
        "macro_test_mse": float(frame["test_mse"].mean()),
        "macro_test_mae": float(frame["test_mae"].mean()),
        "macro_test_r2": float(frame["test_r2"].mean()),
        "median_test_r2": float(frame["test_r2"].median()),
        "stocks_with_positive_test_r2": int((frame["test_r2"] > 0).sum()),
        "stocks_beating_mean_baseline": int(
            frame["test_beat_mean_baseline"].sum()
        ),
        "macro_test_correlation": (
            float(correlations.mean()) if len(correlations) else None
        ),
        "macro_test_actual_mean": float(frame["test_actual_mean"].mean()),
        "macro_test_predicted_mean": float(frame["test_predicted_mean"].mean()),
        "per_symbol": per_symbol,
    }


def compare_to_baseline(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    cand = {row["symbol"]: row for row in candidate["per_symbol"]}
    base = {row["symbol"]: row for row in baseline["per_symbol"]}
    better_test = []
    lower_train_better_test = []
    for symbol in sorted(cand):
        c, b = cand[symbol], base[symbol]
        if c["test_r2"] > b["test_r2"]:
            better_test.append(symbol)
            if c["train_r2"] < b["train_r2"]:
                lower_train_better_test.append(symbol)
    return {
        "baseline_k": baseline["n_neighbors"],
        "candidate_k": candidate["n_neighbors"],
        "stocks_better_test_r2": better_test,
        "n_better_test_r2": len(better_test),
        "stocks_lower_train_r2_and_better_test_r2": lower_train_better_test,
        "n_lower_train_r2_and_better_test_r2": len(lower_train_better_test),
        "macro_test_r2_delta": float(
            candidate["macro_test_r2"] - baseline["macro_test_r2"]
        ),
        "macro_train_r2_delta": float(
            candidate["macro_train_r2"] - baseline["macro_train_r2"]
        ),
    }


def print_recipe_report(recipe: str, payload: dict[str, Any]) -> None:
    log(f"=== {recipe} features: {payload['features']} ===")
    by_k = payload["by_k"]
    k15 = pd.DataFrame(by_k["15"]["per_symbol"]).set_index("symbol")
    k21 = pd.DataFrame(by_k["21"]["per_symbol"]).set_index("symbol")
    k51 = pd.DataFrame(by_k["51"]["per_symbol"]).set_index("symbol")
    table = pd.DataFrame(
        {
            "train_r2_15": k15["train_r2"],
            "test_r2_15": k15["test_r2"],
            "corr_15": k15["test_correlation"],
            "train_r2_21": k21["train_r2"],
            "test_r2_21": k21["test_r2"],
            "corr_21": k21["test_correlation"],
            "train_r2_51": k51["train_r2"],
            "test_r2_51": k51["test_r2"],
            "corr_51": k51["test_correlation"],
        }
    )
    table["best_test_k"] = (
        table[["test_r2_15", "test_r2_21", "test_r2_51"]]
        .idxmax(axis=1)
        .str.replace("test_r2_", "", regex=False)
    )
    print(table.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    for k in ("15", "21", "51"):
        s = by_k[k]
        log(
            f"{recipe} k={k}: "
            f"macro train R2={s['macro_train_r2']:.4f}, "
            f"macro test R2={s['macro_test_r2']:.4f}, "
            f"median test R2={s['median_test_r2']:.4f}, "
            f"positive test R2="
            f"{s['stocks_with_positive_test_r2']}/{s['stock_count']}, "
            f"beat train-mean baseline="
            f"{s['stocks_beating_mean_baseline']}/{s['stock_count']}, "
            f"macro test corr={s['macro_test_correlation']:.4f}."
        )
    for key in ("k15_vs_k51", "k21_vs_k51"):
        cmp_ = payload[key]
        log(
            f"{recipe} k={cmp_['candidate_k']} vs k={cmp_['baseline_k']}: "
            f"better test R2="
            f"{cmp_['n_better_test_r2']}/{by_k['51']['stock_count']} "
            f"{cmp_['stocks_better_test_r2']}; "
            f"lower train R2 and better test R2="
            f"{cmp_['n_lower_train_r2_and_better_test_r2']}/"
            f"{by_k['51']['stock_count']} "
            f"{cmp_['stocks_lower_train_r2_and_better_test_r2']}; "
            f"macro test R2 Δ={cmp_['macro_test_r2_delta']:+.4f}."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train per-ticker scaled kNN on every clean-yfinance CSV to "
            "predict next-day absolute Adj Close return (volatility), using "
            "frozen v4/v5 features and fixed k in {15, 21, 51}."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    frames = load_stock_frames(args.data_dir)
    log(
        f"Loaded {len(frames)} tickers from {args.data_dir}. "
        f"Cutoff={CUTOFF}. Lag={LAG}. "
        f"Target=|next-day Adj Close return|. "
        f"Fixed k={N_NEIGHBORS_LIST}. "
        "Feature sets frozen from kNN v4 and v5 (original candidate pool; "
        "new utility.py columns excluded)."
    )
    log(
        "Original features in play: "
        + ", ".join(SPLIT_FEATURE_COLS)
        + ". Excluded new columns: Signed_Rel_Vol, Intraday, Vol_*, ATR_*, "
        "SMA_Slope, Window_Close_Location."
    )

    symbols = sorted(frames)
    symbol_rows = []
    for i, symbol in enumerate(symbols, 1):
        ticker_t0 = time.perf_counter()
        log(f"[{i}/{len(symbols)}] start {symbol}")
        row = run_symbol(symbol, frames[symbol])
        symbol_rows.append(row)
        elapsed = time.perf_counter() - ticker_t0
        remaining = (time.perf_counter() - started) / i * (len(symbols) - i)
        v5_k51 = row["recipes"]["v5"]["by_k"]["51"]["test"]
        log(
            f"[{i}/{len(symbols)}] done {symbol} in {elapsed:.1f}s "
            f"(n_train={row['n_train']}, n_test={row['n_test']}, "
            f"v5 k=51 test R2={v5_k51['r2']:+.4f}). "
            f"ETA {remaining:.0f}s"
        )

    recipes: dict[str, Any] = {}
    for recipe, features in FEATURE_SETS.items():
        by_k = {
            str(k): summarize_k(symbol_rows, recipe, k) for k in N_NEIGHBORS_LIST
        }
        recipes[recipe] = {
            "features": features,
            "by_k": by_k,
            "k15_vs_k51": compare_to_baseline(by_k["15"], by_k["51"]),
            "k21_vs_k51": compare_to_baseline(by_k["21"], by_k["51"]),
        }

    results = {
        "experiment": (
            "per-ticker kNN on next-day absolute Adj Close return "
            "(volatility), frozen v4/v5 features, k=15/21/51"
        ),
        "script": "src/ml/regression/ai_scripts/ai-knn-volatility-regression.py",
        "target": "absolute next-day adjusted close-to-close return",
        "data_directory": str(args.data_dir),
        "stocks": symbols,
        "stock_count": len(frames),
        "cutoff": CUTOFF,
        "lag": LAG,
        "n_neighbors": N_NEIGHBORS_LIST,
        "excluded_new_utility_columns": [
            "Signed_Rel_Vol",
            "Intraday",
            "Vol_20",
            "ATR_20",
            "SMA_Slope",
            "Window_Close_Location",
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "recipes": recipes,
    }

    for recipe in FEATURE_SETS:
        print_recipe_report(recipe, results["recipes"][recipe])

    log(f"Finished in {results['elapsed_seconds']:.1f}s.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        log(f"Wrote full results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
