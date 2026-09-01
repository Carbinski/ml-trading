from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
    with_volatility_target,
)


def evaluate_volatility(
    predictions: pd.DataFrame, training_means: dict[str, float]
) -> dict[str, Any]:
    result = predictions.copy()
    negative_prediction_rate = float((result["Predicted"] < 0).mean())
    result["Predicted"] = result["Predicted"].clip(lower=0)
    evaluation = evaluate_regression(result, training_means)
    evaluation["summary"].update(
        {
            "negative_prediction_rate_before_clipping": negative_prediction_rate,
            "mean_actual_absolute_return": float(result["Actual"].mean()),
            "mean_predicted_absolute_return": float(result["Predicted"].mean()),
        }
    )
    return evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict next-day absolute adjusted return as volatility."
    )
    parser.add_argument("--data-dir", type=Path, default=CLEAN_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    panel, feature_columns = load_return_panel(args.data_dir, horizon=1)
    panel = with_volatility_target(panel)
    train, validation, test = split_panel(panel)
    model, alpha = fit_ridge(
        train[feature_columns],
        train["Target"],
        train["Date"],
        n_jobs=args.n_jobs,
    )
    training_means = (
        train.groupby("Symbol")["Target"].mean().astype(float).to_dict()
    )
    validation_results = evaluate_volatility(
        prediction_frame(model, validation, feature_columns),
        training_means,
    )
    test_results = evaluate_volatility(
        prediction_frame(model, test, feature_columns),
        training_means,
    )
    results = {
        "experiment": "next-day volatility target",
        "target": "absolute next-day adjusted close-to-close return",
        "model": (
            "one pooled degree-1 Ridge model; negative volatility predictions "
            "are clipped to zero for evaluation"
        ),
        "best_alpha": alpha,
        "validation_cutoff": VAL_CUTOFF,
        "test_cutoff": TEST_CUTOFF,
        "feature_count": len(feature_columns),
        "validation": validation_results,
        "test": test_results,
    }
    summary = test_results["summary"]
    print(
        "Volatility test: "
        f"micro R2={summary['micro_r2']:.6f}, "
        f"macro R2={summary['macro_r2']:.6f}, "
        f"MSE improvement vs stock means="
        f"{summary['mse_improvement_vs_baseline_pct']:.3f}%, "
        f"correlation={summary['correlation']:.4f}."
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
