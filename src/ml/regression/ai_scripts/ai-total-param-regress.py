from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ml.config import CLEAN_DIR as DATA_DIR, CUTOFF

VAL_CUTOFF = CUTOFF
TEST_CUTOFF = "2025-09-01"
HORIZON = 1
COST = 1e-4
FEATURE_COLUMNS = (
    "Ret_1",
    "Overnight",
    "Intraday",
    "Range",
    "Close_Location",
    "Relative_Volume",
    "Vol_20",
    "5_Day_Return",
    "20_Day_Return",
    "Dist_From_SMA",
)
REQUIRED_COLUMNS = {
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Adj Close",
}
DOCUMENTED_SPY_V8 = {
    "source": "docs/ml/regression/parametric-regression/parametric-linear-regression.md",
    "holdout_start": VAL_CUTOFF,
    "holdout_end": None,
    "mse": 0.00010970328112697492,
    "r2": -0.008331362816047738,
    "note": (
        "Older SPY-only v8 used raw price-level features and one post-2024 "
        "holdout; it is context, not an apples-to-apples benchmark."
    ),
}


def lagged_feature_columns(lag: int) -> list[str]:
    if lag < 1:
        raise ValueError("lag must be at least 1")
    return [f"{feature}_{offset}" for offset in range(lag) for feature in FEATURE_COLUMNS]


def load_stock_frames(data_dir: Path) -> dict[str, pd.DataFrame]:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No stock CSV files found in {data_dir}")

    frames: dict[str, pd.DataFrame] = {}
    for path in files:
        frame = pd.read_csv(path, parse_dates=["Date"])
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        frames[path.stem] = frame.sort_values("Date").reset_index(drop=True)
    return frames


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy().sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Range"] = (df["High"] - df["Low"]) / df["Close"]
    adjusted_open = df["Open"] * (df["Adj Close"] / df["Close"])
    df["Overnight"] = adjusted_open / df["Adj Close"].shift(1) - 1
    df["Intraday"] = df["Close"] / df["Open"] - 1
    high_low = df["High"] - df["Low"]
    df["Close_Location"] = np.where(
        high_low > 0, (df["Close"] - df["Low"]) / high_low, 0.5
    )
    df["Relative_Volume"] = (
        df["Volume"] / df["Volume"].rolling(20).mean().shift(1)
    )
    df["Ret_1"] = df["Adj Close"] / df["Adj Close"].shift(1) - 1
    df["Vol_20"] = df["Ret_1"].rolling(20).std()
    df["5_Day_Return"] = df["Adj Close"] / df["Adj Close"].shift(5) - 1
    df["20_Day_Return"] = df["Adj Close"] / df["Adj Close"].shift(20) - 1
    lagged_close = df["Adj Close"].shift(1)
    rolling_mean = lagged_close.rolling(20).mean()
    rolling_std = lagged_close.rolling(20).std()
    df["Dist_From_SMA"] = (df["Adj Close"] - rolling_mean) / rolling_std
    return df


def build_supervised(
    engineered: pd.DataFrame, symbol: str, lag: int = 1
) -> pd.DataFrame:
    lagged = pd.concat(
        [
            engineered[list(FEATURE_COLUMNS)]
            .shift(offset)
            .add_suffix(f"_{offset}")
            for offset in range(lag)
        ],
        axis=1,
    )
    target = engineered["Adj Close"].shift(-HORIZON) / engineered["Adj Close"] - 1
    target_date = engineered["Date"].shift(-HORIZON)
    supervised = pd.concat(
        [
            pd.DataFrame(
                {
                    "Symbol": symbol,
                    "Date": engineered["Date"],
                    "TargetDate": target_date,
                }
            ),
            lagged,
            target.rename("Target"),
        ],
        axis=1,
    )
    valid = supervised.notna().all(axis=1)
    return supervised.loc[valid].reset_index(drop=True)


def split_supervised(
    panel: pd.DataFrame,
    val_cutoff: str = VAL_CUTOFF,
    test_cutoff: str = TEST_CUTOFF,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = panel["Date"]
    target_dates = panel["TargetDate"]
    train = panel.loc[(dates < val_cutoff) & (target_dates < val_cutoff)].copy()
    validation = panel.loc[
        (dates >= val_cutoff)
        & (dates < test_cutoff)
        & (target_dates < test_cutoff)
    ].copy()
    test = panel.loc[dates >= test_cutoff].copy()
    return train, validation, test


def date_group_splits(
    dates: pd.Series, n_splits: int = 5, gap: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    normalized = pd.to_datetime(dates).reset_index(drop=True)
    unique_dates = np.array(sorted(normalized.unique()))
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_date_idx, test_date_idx in splitter.split(unique_dates):
        train_dates = unique_dates[train_date_idx]
        test_dates = unique_dates[test_date_idx]
        train_rows = np.flatnonzero(normalized.isin(train_dates).to_numpy())
        test_rows = np.flatnonzero(normalized.isin(test_dates).to_numpy())
        splits.append((train_rows, test_rows))
    return splits


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    *,
    n_splits: int = 5,
    gap: int = 5,
    degree: int = 1,
    n_jobs: int = -1,
) -> tuple[Pipeline, float]:
    search = GridSearchCV(
        make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            StandardScaler(),
            Ridge(),
        ),
        param_grid={"ridge__alpha": np.logspace(-4, 4, 30)},
        cv=date_group_splits(dates, n_splits=n_splits, gap=gap),
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
    )
    search.fit(X, y)
    return search.best_estimator_, float(search.best_params_["ridge__alpha"])


def prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reference_mean: float,
) -> dict[str, Any]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(actual, predicted))
    mse_mean = float(
        mean_squared_error(actual, np.full_like(actual, reference_mean, dtype=float))
    )
    mse_zero = float(mean_squared_error(actual, np.zeros_like(actual, dtype=float)))
    actual_std = float(actual.std())
    predicted_std = float(predicted.std())
    correlation = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if actual_std > 0 and predicted_std > 0
        else None
    )
    positions = np.sign(predicted)
    turnover = float(np.mean(np.abs(np.diff(positions, prepend=0.0))))
    return {
        "observations": int(len(actual)),
        "mse": mse,
        "r2": float(r2_score(actual, predicted)),
        "mse_mean_baseline": mse_mean,
        "mse_zero_baseline": mse_zero,
        "mse_improvement_vs_mean_pct": float(100 * (mse_mean - mse) / mse_mean),
        "mse_improvement_vs_zero_pct": float(100 * (mse_zero - mse) / mse_zero),
        "beat_mean_baseline": bool(mse < mse_mean),
        "beat_zero_baseline": bool(mse < mse_zero),
        "prediction_std": predicted_std,
        "correlation": correlation,
        "directional_accuracy": float(np.mean(np.sign(predicted) == np.sign(actual))),
        "turnover": turnover,
        "estimated_cost_per_day": turnover * COST,
    }


def predictions_for_split(
    model: Pipeline, split: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    predicted = model.predict(split[feature_columns])
    return pd.DataFrame(
        {
            "Symbol": split["Symbol"].to_numpy(),
            "Date": split["Date"].to_numpy(),
            "Actual": split["Target"].to_numpy(),
            "Predicted": predicted,
        }
    )


def summarize_predictions(
    predictions: pd.DataFrame, training_means: dict[str, float]
) -> dict[str, Any]:
    per_symbol: list[dict[str, Any]] = []
    for symbol, rows in predictions.groupby("Symbol", sort=True):
        metrics = prediction_metrics(
            rows["Actual"].to_numpy(),
            rows["Predicted"].to_numpy(),
            reference_mean=training_means[symbol],
        )
        per_symbol.append({"symbol": symbol, **metrics})

    metrics_frame = pd.DataFrame(per_symbol)
    reference = predictions["Symbol"].map(training_means).to_numpy(dtype=float)
    actual = predictions["Actual"].to_numpy(dtype=float)
    predicted = predictions["Predicted"].to_numpy(dtype=float)
    micro_mse = float(mean_squared_error(actual, predicted))
    micro_mean_mse = float(mean_squared_error(actual, reference))
    correlation_values = metrics_frame["correlation"].dropna()
    summary = {
        "stock_count": int(len(per_symbol)),
        "observations": int(len(predictions)),
        "macro_mse": float(metrics_frame["mse"].mean()),
        "macro_r2": float(metrics_frame["r2"].mean()),
        "median_r2": float(metrics_frame["r2"].median()),
        "micro_mse": micro_mse,
        "micro_r2": float(r2_score(actual, predicted)),
        "macro_mse_mean_baseline": float(
            metrics_frame["mse_mean_baseline"].mean()
        ),
        "micro_mse_stock_mean_baseline": micro_mean_mse,
        "macro_mse_improvement_vs_mean_pct": float(
            100
            * (
                metrics_frame["mse_mean_baseline"].mean()
                - metrics_frame["mse"].mean()
            )
            / metrics_frame["mse_mean_baseline"].mean()
        ),
        "micro_mse_improvement_vs_stock_mean_pct": float(
            100 * (micro_mean_mse - micro_mse) / micro_mean_mse
        ),
        "stocks_beating_mean_baseline": int(
            metrics_frame["beat_mean_baseline"].sum()
        ),
        "stocks_beating_zero_baseline": int(
            metrics_frame["beat_zero_baseline"].sum()
        ),
        "stocks_with_positive_r2": int((metrics_frame["r2"] > 0).sum()),
        "macro_directional_accuracy": float(
            metrics_frame["directional_accuracy"].mean()
        ),
        "macro_correlation": (
            float(correlation_values.mean()) if len(correlation_values) else None
        ),
        "macro_prediction_std": float(metrics_frame["prediction_std"].mean()),
        "macro_turnover": float(metrics_frame["turnover"].mean()),
        "r2_10th_percentile": float(metrics_frame["r2"].quantile(0.10)),
        "r2_90th_percentile": float(metrics_frame["r2"].quantile(0.90)),
    }
    return {"summary": summary, "per_symbol": per_symbol}


def run_individual_models(
    panel: pd.DataFrame,
    feature_columns: list[str],
    *,
    n_splits: int,
    gap: int,
    degree: int,
    n_jobs: int,
) -> dict[str, Any]:
    train, validation, test = split_supervised(panel)
    training_means: dict[str, float] = {}
    alphas: dict[str, float] = {}
    validation_predictions: list[pd.DataFrame] = []
    test_predictions: list[pd.DataFrame] = []

    for symbol in sorted(panel["Symbol"].unique()):
        symbol_train = train.loc[train["Symbol"] == symbol].reset_index(drop=True)
        symbol_validation = validation.loc[
            validation["Symbol"] == symbol
        ].reset_index(drop=True)
        symbol_test = test.loc[test["Symbol"] == symbol].reset_index(drop=True)
        model, alpha = train_model(
            symbol_train[feature_columns],
            symbol_train["Target"],
            symbol_train["Date"],
            n_splits=n_splits,
            gap=gap,
            degree=degree,
            n_jobs=n_jobs,
        )
        training_means[symbol] = float(symbol_train["Target"].mean())
        alphas[symbol] = alpha
        validation_predictions.append(
            predictions_for_split(model, symbol_validation, feature_columns)
        )
        test_predictions.append(
            predictions_for_split(model, symbol_test, feature_columns)
        )

    validation_frame = pd.concat(validation_predictions, ignore_index=True)
    test_frame = pd.concat(test_predictions, ignore_index=True)
    return {
        "best_alphas": alphas,
        "validation": summarize_predictions(validation_frame, training_means),
        "test": summarize_predictions(test_frame, training_means),
        "_test_predictions": test_frame,
    }


def run_pooled_model(
    panel: pd.DataFrame,
    feature_columns: list[str],
    *,
    n_splits: int,
    gap: int,
    degree: int,
    n_jobs: int,
) -> dict[str, Any]:
    train, validation, test = split_supervised(panel)
    train = train.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    validation = validation.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    test = test.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    model, alpha = train_model(
        train[feature_columns],
        train["Target"],
        train["Date"],
        n_splits=n_splits,
        gap=gap,
        degree=degree,
        n_jobs=n_jobs,
    )
    training_means = (
        train.groupby("Symbol")["Target"].mean().astype(float).to_dict()
    )
    validation_predictions = predictions_for_split(
        model, validation, feature_columns
    )
    test_predictions = predictions_for_split(model, test, feature_columns)
    return {
        "best_alpha": alpha,
        "validation": summarize_predictions(
            validation_predictions, training_means
        ),
        "test": summarize_predictions(test_predictions, training_means),
        "_test_predictions": test_predictions,
    }


def compare_models(
    individual: dict[str, Any], pooled: dict[str, Any]
) -> dict[str, Any]:
    individual_summary = individual["test"]["summary"]
    pooled_summary = pooled["test"]["summary"]
    individual_by_symbol = {
        row["symbol"]: row for row in individual["test"]["per_symbol"]
    }
    pooled_by_symbol = {
        row["symbol"]: row for row in pooled["test"]["per_symbol"]
    }
    symbols = sorted(individual_by_symbol)
    return {
        "pooled_minus_individual_macro_r2": float(
            pooled_summary["macro_r2"] - individual_summary["macro_r2"]
        ),
        "pooled_minus_individual_micro_r2": float(
            pooled_summary["micro_r2"] - individual_summary["micro_r2"]
        ),
        "pooled_mse_change_vs_individual_pct": float(
            100
            * (
                pooled_summary["micro_mse"] - individual_summary["micro_mse"]
            )
            / individual_summary["micro_mse"]
        ),
        "pooled_symbols_with_lower_mse": int(
            sum(
                pooled_by_symbol[symbol]["mse"]
                < individual_by_symbol[symbol]["mse"]
                for symbol in symbols
            )
        ),
        "pooled_symbols_with_higher_r2": int(
            sum(
                pooled_by_symbol[symbol]["r2"]
                > individual_by_symbol[symbol]["r2"]
                for symbol in symbols
            )
        ),
    }


def run_experiment(
    data_dir: Path = DATA_DIR,
    *,
    lag: int = 1,
    degree: int = 1,
    n_splits: int = 5,
    gap: int = 5,
    n_jobs: int = -1,
) -> dict[str, Any]:
    frames = load_stock_frames(data_dir)
    supervised = [
        build_supervised(engineer_features(frame), symbol, lag=lag)
        for symbol, frame in frames.items()
    ]
    panel = pd.concat(supervised, ignore_index=True).sort_values(
        ["Date", "Symbol"]
    )
    feature_columns = lagged_feature_columns(lag)
    individual = run_individual_models(
        panel,
        feature_columns,
        n_splits=n_splits,
        gap=gap,
        degree=degree,
        n_jobs=n_jobs,
    )
    pooled = run_pooled_model(
        panel,
        feature_columns,
        n_splits=n_splits,
        gap=gap,
        degree=degree,
        n_jobs=n_jobs,
    )
    comparison = compare_models(individual, pooled)
    individual.pop("_test_predictions")
    pooled.pop("_test_predictions")
    return {
        "metadata": {
            "data_directory": str(data_dir),
            "stocks": sorted(frames),
            "stock_count": len(frames),
            "feature_count": len(feature_columns),
            "features": feature_columns,
            "degree": degree,
            "lag": lag,
            "validation_cutoff": VAL_CUTOFF,
            "test_cutoff": TEST_CUTOFF,
            "cross_validation": (
                f"{n_splits}-fold expanding date-grouped TimeSeriesSplit "
                f"with gap={gap}"
            ),
            "date_start": str(panel["Date"].min().date()),
            "date_end": str(panel["TargetDate"].max().date()),
        },
        "documented_spy_v8": DOCUMENTED_SPY_V8,
        "individual_models": individual,
        "pooled_model": pooled,
        "pooled_vs_individual": comparison,
    }


def print_report(results: dict[str, Any]) -> None:
    metadata = results["metadata"]
    individual = results["individual_models"]["test"]["summary"]
    pooled = results["pooled_model"]["test"]["summary"]
    individual_spy = next(
        row
        for row in results["individual_models"]["test"]["per_symbol"]
        if row["symbol"] == "SPY"
    )
    pooled_spy = next(
        row
        for row in results["pooled_model"]["test"]["per_symbol"]
        if row["symbol"] == "SPY"
    )
    documented = results["documented_spy_v8"]
    print(
        f"Loaded {metadata['stock_count']} stocks from "
        f"{metadata['date_start']} through {metadata['date_end']}."
    )
    print(
        "Individual models, test: "
        f"macro R2={individual['macro_r2']:.6f}, "
        f"micro R2={individual['micro_r2']:.6f}, "
        f"micro MSE={individual['micro_mse']:.8g}, "
        f"beat stock-mean baseline="
        f"{individual['stocks_beating_mean_baseline']}/{individual['stock_count']}."
    )
    print(
        "Pooled model, test: "
        f"macro R2={pooled['macro_r2']:.6f}, "
        f"micro R2={pooled['micro_r2']:.6f}, "
        f"micro MSE={pooled['micro_mse']:.8g}, "
        f"beat stock-mean baseline="
        f"{pooled['stocks_beating_mean_baseline']}/{pooled['stock_count']}."
    )
    print(
        "SPY on corrected 2025+ test: "
        f"individual R2={individual_spy['r2']:.6f}, "
        f"MSE={individual_spy['mse']:.8g}; "
        f"pooled R2={pooled_spy['r2']:.6f}, MSE={pooled_spy['mse']:.8g}."
    )
    print(
        "Documented SPY v8 on the older 2024+ holdout: "
        f"R2={documented['r2']:.6f}, MSE={documented['mse']:.8g}."
    )


class RegressionExperimentTests(unittest.TestCase):
    def make_prices(self, periods: int = 30) -> pd.DataFrame:
        close = np.arange(100.0, 100.0 + periods)
        return pd.DataFrame(
            {
                "Date": pd.bdate_range("2024-01-01", periods=periods),
                "Open": close - 0.5,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": np.full(periods, 1_000_000),
                "Adj Close": close,
            }
        )

    def test_engineer_features_builds_stationary_columns_without_mutating_input(self):
        function = globals().get("engineer_features")
        self.assertIsNotNone(function, "engineer_features must be implemented")
        raw = self.make_prices()
        original_columns = raw.columns.copy()

        result = function(raw)

        self.assertEqual(list(raw.columns), list(original_columns))
        self.assertTrue(set(FEATURE_COLUMNS).issubset(result.columns))
        self.assertAlmostEqual(result.loc[20, "Relative_Volume"], 1.0)
        self.assertAlmostEqual(result.loc[0, "Range"], 0.02)

    def test_build_supervised_aligns_features_with_next_day_target(self):
        function = globals().get("build_supervised")
        self.assertIsNotNone(function, "build_supervised must be implemented")
        engineered = engineer_features(self.make_prices())

        result = function(engineered, "TEST", lag=1)

        first = result.iloc[0]
        source_row = engineered.loc[engineered["Date"] == first["Date"]].iloc[0]
        target_row = engineered.loc[engineered["Date"] == first["TargetDate"]].iloc[0]
        expected = target_row["Adj Close"] / source_row["Adj Close"] - 1
        self.assertEqual(first["Symbol"], "TEST")
        self.assertAlmostEqual(first["Target"], expected)
        self.assertTrue(
            {f"{feature}_0" for feature in FEATURE_COLUMNS}.issubset(result.columns)
        )

    def test_split_supervised_purges_targets_crossing_boundaries(self):
        function = globals().get("split_supervised")
        self.assertIsNotNone(function, "split_supervised must be implemented")
        dates = pd.date_range("2024-01-01", periods=7)
        panel = pd.DataFrame(
            {
                "Date": dates[:-1],
                "TargetDate": dates[1:],
                "Symbol": "TEST",
                "Feature": np.arange(6),
                "Target": np.arange(6),
            }
        )

        train, validation, test = function(
            panel, val_cutoff="2024-01-04", test_cutoff="2024-01-06"
        )

        self.assertEqual(train["Date"].dt.day.tolist(), [1, 2])
        self.assertEqual(validation["Date"].dt.day.tolist(), [4])
        self.assertEqual(test["Date"].dt.day.tolist(), [6])

    def test_date_group_splits_keep_dates_together_and_respect_gap(self):
        function = globals().get("date_group_splits")
        self.assertIsNotNone(function, "date_group_splits must be implemented")
        dates = pd.Series(np.repeat(pd.bdate_range("2024-01-01", periods=12), 2))

        splits = function(dates, n_splits=3, gap=1)

        self.assertEqual(len(splits), 3)
        unique_dates = pd.Index(sorted(dates.unique()))
        positions = {date: i for i, date in enumerate(unique_dates)}
        for train_idx, test_idx in splits:
            train_dates = set(dates.iloc[train_idx])
            test_dates = set(dates.iloc[test_idx])
            self.assertTrue(train_dates.isdisjoint(test_dates))
            self.assertLess(
                max(positions[date] for date in train_dates),
                min(positions[date] for date in test_dates) - 1,
            )

    def test_prediction_metrics_reports_model_and_baselines(self):
        function = globals().get("prediction_metrics")
        self.assertIsNotNone(function, "prediction_metrics must be implemented")

        result = function(
            np.array([1.0, -1.0]),
            np.array([0.5, -0.5]),
            reference_mean=0.0,
        )

        self.assertAlmostEqual(result["mse"], 0.25)
        self.assertAlmostEqual(result["r2"], 0.75)
        self.assertAlmostEqual(result["directional_accuracy"], 1.0)
        self.assertAlmostEqual(result["correlation"], 1.0)
        self.assertAlmostEqual(result["mse_mean_baseline"], 1.0)
        self.assertAlmostEqual(result["mse_zero_baseline"], 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare stock-specific Ridge regressions with one pooled "
            "generalized regression across every clean-yfinance CSV."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--gap", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(
            RegressionExperimentTests
        )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    results = run_experiment(
        args.data_dir,
        lag=args.lag,
        degree=args.degree,
        n_splits=args.n_splits,
        gap=args.gap,
        n_jobs=args.n_jobs,
    )
    print_report(results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote full results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
