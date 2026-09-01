from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ml.regression.parametric_experiment_common import (
    CLEAN_DIR,
    TEST_CUTOFF,
    VAL_CUTOFF,
    evaluate_regression,
    fit_ridge,
    load_return_panel,
    prediction_frame,
    split_panel,
)

HORIZON = 5


def evaluate_long_horizon(
    predictions: pd.DataFrame, training_means: dict[str, float]
) -> dict[str, Any]:
    evaluation = evaluate_regression(predictions, training_means)
    actual = predictions["Actual"].to_numpy(dtype=float)
    predicted = predictions["Predicted"].to_numpy(dtype=float)
    up_rate = float(np.mean(actual > 0))
    evaluation["summary"].update(
        {
            "directional_accuracy": float(
                np.mean(np.sign(actual) == np.sign(predicted))
            ),
            "always_up_accuracy": up_rate,
            "majority_direction_accuracy": max(up_rate, 1 - up_rate),
        }
    )
    return evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the five-session forward adjusted return."
    )
    parser.add_argument("--data-dir", type=Path, default=CLEAN_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    panel, feature_columns = load_return_panel(
        args.data_dir, horizon=HORIZON
    )
    panel["Target"] = panel["ForwardReturn"]
    train, validation, test = split_panel(panel)
    model, alpha = fit_ridge(
        train[feature_columns],
        train["Target"],
        train["Date"],
        gap=HORIZON,
        n_jobs=args.n_jobs,
    )
    training_means = (
        train.groupby("Symbol")["Target"].mean().astype(float).to_dict()
    )
    validation_results = evaluate_long_horizon(
        prediction_frame(model, validation, feature_columns),
        training_means,
    )
    test_results = evaluate_long_horizon(
        prediction_frame(model, test, feature_columns),
        training_means,
    )
    results = {
        "experiment": "longer-horizon return target",
        "target": "five-session forward adjusted return",
        "model": "one pooled degree-1 Ridge model",
        "horizon_sessions": HORIZON,
        "overlapping_targets": True,
        "best_alpha": alpha,
        "validation_cutoff": VAL_CUTOFF,
        "test_cutoff": TEST_CUTOFF,
        "cross_validation_gap_sessions": HORIZON,
        "feature_count": len(feature_columns),
        "validation": validation_results,
        "test": test_results,
    }
    summary = test_results["summary"]
    print(
        "Five-session return test: "
        f"micro R2={summary['micro_r2']:.6f}, "
        f"macro R2={summary['macro_r2']:.6f}, "
        f"MSE improvement vs stock means="
        f"{summary['mse_improvement_vs_baseline_pct']:.3f}%, "
        f"directional accuracy={summary['directional_accuracy']:.2%} "
        f"(always up={summary['always_up_accuracy']:.2%})."
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
