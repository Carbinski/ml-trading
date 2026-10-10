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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.ml import utility as ut
from src.ml.config import CLEAN_DIR as DATA_DIR, CUTOFF

LAG = 1
# Frozen from SPY SVM v4/v5 train-only CV (best C=0.01, gamma=0.01, epsilon=0.01).
SVR_C = 0.01
SVR_GAMMA = 0.01
SVR_EPSILON = 0.01
SVR_MAX_ITER = 16000

FEATURE_SETS = {
    "v4": [
        "1_Day_Return",
        "Overnight",
        "Range",
        "Signed_Rel_Vol",
        "Upper_Wick",
        "Lower_Wick",
    ],
    "v5": [
        "Adj Close",
        "1_Day_Return",
        "20_Day_Return",
        "Overnight",
        "Range",
        "Signed_Rel_Vol",
        "ATR_20",
        "Dist_From_SMA",
        "Upper_Wick",
        "Lower_Wick",
    ],
}

SPLIT_FEATURE_COLS = [
    "Adj Close",
    "1_Day_Return",
    "20_Day_Return",
    "Overnight",
    "Range",
    "Signed_Rel_Vol",
    "ATR_20",
    "Dist_From_SMA",
    "Upper_Wick",
    "Lower_Wick",
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


def train_model(X: pd.DataFrame, y: pd.Series):
    model = make_pipeline(
        StandardScaler(),
        SVR(
            kernel="rbf",
            C=SVR_C,
            gamma=SVR_GAMMA,
            epsilon=SVR_EPSILON,
            cache_size=500,
            max_iter=SVR_MAX_ITER,
            tol=1e-3,
        ),
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
        "r2": float(r2_score(actual, predicted)),
        "mse_mean_baseline": mse_mean,
        "beat_mean_baseline": bool(mse < mse_mean),
        "prediction_std": pred_std,
        "correlation": correlation,
        "directional_accuracy": float(
            np.mean(np.sign(predicted) == np.sign(actual))
        ),
    }


def run_symbol(symbol: str, raw: pd.DataFrame) -> dict[str, Any]:
    df, _ = ut.process_OHLCV_all(raw.sort_values("Date").dropna())
    X_train, y_train, X_test, y_test = ut.split_data(
        df, SPLIT_FEATURE_COLS, CUTOFF, lag=LAG
    )
    train_mean = float(y_train.mean())
    recipes: dict[str, Any] = {}
    for recipe, features in FEATURE_SETS.items():
        cols = ut.lagged_cols(X_train, features)
        t0 = time.perf_counter()
        model = train_model(X_train[cols], y_train)
        y_train_pred = model.predict(X_train[cols])
        y_test_pred = model.predict(X_test[cols])
        fit_s = time.perf_counter() - t0
        svr = model.named_steps["svr"]
        train = split_metrics(y_train, y_train_pred, train_mean)
        test = split_metrics(y_test, y_test_pred, train_mean)
        recipes[recipe] = {
            "features": features,
            "n_features": int(len(cols)),
            "n_support": int(svr.support_.size),
            "intercept": float(np.ravel(svr.intercept_)[0]),
            "fit_seconds": fit_s,
            "train": train,
            "test": test,
        }
        log(
            f"    {symbol} {recipe}: "
            f"train R2={train['r2']:+.4f}  test R2={test['r2']:+.4f}  "
            f"test corr="
            f"{test['correlation'] if test['correlation'] is not None else float('nan'):+.3f}  "
            f"beat mean={test['beat_mean_baseline']}  "
            f"n_support={svr.support_.size}  fit={fit_s:.2f}s"
        )
    return {
        "symbol": symbol,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "recipes": recipes,
    }


def summarize_recipe(rows: list[dict[str, Any]], recipe: str) -> dict[str, Any]:
    per_symbol = []
    for row in rows:
        fit = row["recipes"][recipe]
        train = fit["train"]
        test = fit["test"]
        per_symbol.append(
            {
                "symbol": row["symbol"],
                "n_train": row["n_train"],
                "n_test": row["n_test"],
                "n_support": fit["n_support"],
                "intercept": fit["intercept"],
                "train_mse": train["mse"],
                "train_r2": train["r2"],
                "test_mse": test["mse"],
                "test_r2": test["r2"],
                "test_mse_mean_baseline": test["mse_mean_baseline"],
                "test_beat_mean_baseline": test["beat_mean_baseline"],
                "test_correlation": test["correlation"],
                "test_directional_accuracy": test["directional_accuracy"],
                "test_prediction_std": test["prediction_std"],
            }
        )
    frame = pd.DataFrame(per_symbol)
    correlations = frame["test_correlation"].dropna()
    return {
        "features": FEATURE_SETS[recipe],
        "stock_count": int(len(frame)),
        "macro_train_mse": float(frame["train_mse"].mean()),
        "macro_train_r2": float(frame["train_r2"].mean()),
        "macro_test_mse": float(frame["test_mse"].mean()),
        "macro_test_r2": float(frame["test_r2"].mean()),
        "median_test_r2": float(frame["test_r2"].median()),
        "stocks_with_positive_test_r2": int((frame["test_r2"] > 0).sum()),
        "stocks_beating_mean_baseline": int(frame["test_beat_mean_baseline"].sum()),
        "macro_test_correlation": (
            float(correlations.mean()) if len(correlations) else None
        ),
        "macro_test_directional_accuracy": float(
            frame["test_directional_accuracy"].mean()
        ),
        "macro_n_support": float(frame["n_support"].mean()),
        "per_symbol": per_symbol,
    }


def compare_recipes(
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
        "baseline": "v4",
        "candidate": "v5",
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


def print_report(results: dict[str, Any]) -> None:
    v4 = pd.DataFrame(results["recipes"]["v4"]["per_symbol"]).set_index("symbol")
    v5 = pd.DataFrame(results["recipes"]["v5"]["per_symbol"]).set_index("symbol")
    table = pd.DataFrame(
        {
            "n_train": v4["n_train"],
            "n_support_v4": v4["n_support"],
            "train_r2_v4": v4["train_r2"],
            "test_r2_v4": v4["test_r2"],
            "corr_v4": v4["test_correlation"],
            "n_support_v5": v5["n_support"],
            "train_r2_v5": v5["train_r2"],
            "test_r2_v5": v5["test_r2"],
            "corr_v5": v5["test_correlation"],
        }
    )
    table["best_test"] = np.where(
        table["test_r2_v5"] > table["test_r2_v4"], "v5", "v4"
    )
    print(table.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    for recipe in FEATURE_SETS:
        s = results["recipes"][recipe]
        log(
            f"{recipe}: "
            f"macro train R2={s['macro_train_r2']:.4f}, "
            f"macro test R2={s['macro_test_r2']:.4f}, "
            f"median test R2={s['median_test_r2']:.4f}, "
            f"positive test R2="
            f"{s['stocks_with_positive_test_r2']}/{s['stock_count']}, "
            f"beat train-mean baseline="
            f"{s['stocks_beating_mean_baseline']}/{s['stock_count']}, "
            f"macro test corr={s['macro_test_correlation']:.4f}, "
            f"macro dir acc={s['macro_test_directional_accuracy']:.4f}."
        )
    cmp_ = results["v5_vs_v4"]
    log(
        f"v5 vs v4: better test R2="
        f"{cmp_['n_better_test_r2']}/{results['stock_count']} "
        f"{cmp_['stocks_better_test_r2']}; "
        f"lower train R2 and better test R2="
        f"{cmp_['n_lower_train_r2_and_better_test_r2']}/"
        f"{results['stock_count']} "
        f"{cmp_['stocks_lower_train_r2_and_better_test_r2']}; "
        f"macro test R2 Δ={cmp_['macro_test_r2_delta']:+.4f}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train per-ticker scaled RBF SVR on every clean-yfinance CSV using "
            "frozen SVM v4/v5 features and C=0.01, gamma=0.01, epsilon=0.01."
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
        f"Frozen SVR C={SVR_C}, gamma={SVR_GAMMA}, epsilon={SVR_EPSILON}. "
        "Feature sets frozen from SVM v4 (6 cols) and v5 (10 cols, includes Adj Close)."
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
        v5_test = row["recipes"]["v5"]["test"]
        log(
            f"[{i}/{len(symbols)}] done {symbol} in {elapsed:.1f}s "
            f"(n_train={row['n_train']}, n_test={row['n_test']}, "
            f"v5 test R2={v5_test['r2']:+.4f}). "
            f"ETA {remaining:.0f}s"
        )

    recipes = {
        recipe: summarize_recipe(symbol_rows, recipe) for recipe in FEATURE_SETS
    }
    results = {
        "experiment": (
            "per-ticker RBF SVR on frozen SVM v4/v5 features, "
            "C=0.01, gamma=0.01, epsilon=0.01"
        ),
        "script": "src/ml/regression/ai_scripts/ai-svm-regression.py",
        "data_directory": str(args.data_dir),
        "stocks": symbols,
        "stock_count": len(frames),
        "cutoff": CUTOFF,
        "lag": LAG,
        "svr": {
            "kernel": "rbf",
            "C": SVR_C,
            "gamma": SVR_GAMMA,
            "epsilon": SVR_EPSILON,
            "max_iter": SVR_MAX_ITER,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "recipes": recipes,
        "v5_vs_v4": compare_recipes(recipes["v5"], recipes["v4"]),
    }

    print_report(results)
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
