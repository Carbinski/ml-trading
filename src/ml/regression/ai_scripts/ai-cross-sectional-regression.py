from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ml.regression.parametric_experiment_common import (
    CLEAN_DIR,
    TEST_CUTOFF,
    VAL_CUTOFF,
    fit_ridge,
    load_return_panel,
    prediction_frame,
    split_panel,
    with_relative_return_target,
)


def summarize_cross_section(predictions: pd.DataFrame) -> dict[str, Any]:
    frame = predictions.copy()
    frame["Predicted"] -= frame.groupby("Date")["Predicted"].transform("mean")
    actual = frame["Actual"].to_numpy(dtype=float)
    predicted = frame["Predicted"].to_numpy(dtype=float)
    mse = float(mean_squared_error(actual, predicted))
    zero_mse = float(mean_squared_error(actual, np.zeros_like(actual)))

    daily_rows: list[dict[str, float]] = []
    for _, day in frame.groupby("Date", sort=True):
        selection_size = max(1, len(day) // 5)
        ordered = day.sort_values("Predicted")
        spread = float(
            ordered.tail(selection_size)["ForwardReturn"].mean()
            - ordered.head(selection_size)["ForwardReturn"].mean()
        )
        daily_rows.append(
            {
                "pearson_ic": float(day["Actual"].corr(day["Predicted"])),
                "rank_ic": float(
                    day["Actual"].corr(day["Predicted"], method="spearman")
                ),
                "top_bottom_spread": spread,
            }
        )

    daily = pd.DataFrame(daily_rows)
    spread_mean = float(daily["top_bottom_spread"].mean())
    spread_std = float(daily["top_bottom_spread"].std(ddof=1))
    spread_t_stat = (
        spread_mean / (spread_std / np.sqrt(len(daily)))
        if spread_std > 0
        else None
    )
    return {
        "observations": int(len(frame)),
        "dates": int(frame["Date"].nunique()),
        "stocks_per_date": int(frame.groupby("Date").size().median()),
        "mse": mse,
        "r2": float(r2_score(actual, predicted)),
        "zero_baseline_mse": zero_mse,
        "mse_improvement_vs_zero_pct": float(100 * (zero_mse - mse) / zero_mse),
        "cross_sectional_directional_accuracy": float(
            np.mean(np.sign(actual) == np.sign(predicted))
        ),
        "mean_daily_pearson_ic": float(daily["pearson_ic"].mean()),
        "mean_daily_rank_ic": float(daily["rank_ic"].mean()),
        "mean_daily_top_bottom_spread": spread_mean,
        "annualized_top_bottom_spread": spread_mean * 252,
        "top_bottom_positive_day_rate": float(
            (daily["top_bottom_spread"] > 0).mean()
        ),
        "naive_spread_t_stat": (
            float(spread_t_stat) if spread_t_stat is not None else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict next-day returns relative to the daily stock universe."
    )
    parser.add_argument("--data-dir", type=Path, default=CLEAN_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    panel, feature_columns = load_return_panel(args.data_dir, horizon=1)
    panel = with_relative_return_target(panel)
    train, validation, test = split_panel(panel)
    model, alpha = fit_ridge(
        train[feature_columns],
        train["Target"],
        train["Date"],
        n_jobs=args.n_jobs,
    )
    validation_predictions = prediction_frame(
        model, validation, feature_columns
    )
    test_predictions = prediction_frame(model, test, feature_columns)
    results = {
        "experiment": "cross-sectional relative-return target",
        "target": (
            "next-day adjusted return minus the equal-weight 20-stock "
            "cross-sectional mean on that target date"
        ),
        "model": "one pooled degree-1 Ridge model",
        "best_alpha": alpha,
        "validation_cutoff": VAL_CUTOFF,
        "test_cutoff": TEST_CUTOFF,
        "feature_count": len(feature_columns),
        "validation": summarize_cross_section(validation_predictions),
        "test": summarize_cross_section(test_predictions),
    }
    test_summary = results["test"]
    print(
        "Cross-sectional test: "
        f"R2={test_summary['r2']:.6f}, "
        f"rank IC={test_summary['mean_daily_rank_ic']:.4f}, "
        f"top-bottom spread/day="
        f"{test_summary['mean_daily_top_bottom_spread']:.6g}, "
        f"positive spread days="
        f"{test_summary['top_bottom_positive_day_rate']:.2%}."
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
