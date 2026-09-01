from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.ml.config import CLEAN_DIR, CUTOFF

VAL_CUTOFF = CUTOFF
TEST_CUTOFF = "2025-09-01"
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


def _engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy().sort_values("Date").reset_index(drop=True)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Range"] = (frame["High"] - frame["Low"]) / frame["Close"]
    adjusted_open = frame["Open"] * (frame["Adj Close"] / frame["Close"])
    frame["Overnight"] = adjusted_open / frame["Adj Close"].shift(1) - 1
    frame["Intraday"] = frame["Close"] / frame["Open"] - 1
    high_low = frame["High"] - frame["Low"]
    frame["Close_Location"] = np.where(
        high_low > 0, (frame["Close"] - frame["Low"]) / high_low, 0.5
    )
    frame["Relative_Volume"] = (
        frame["Volume"] / frame["Volume"].rolling(20).mean().shift(1)
    )
    frame["Ret_1"] = frame["Adj Close"] / frame["Adj Close"].shift(1) - 1
    frame["Vol_20"] = frame["Ret_1"].rolling(20).std()
    frame["5_Day_Return"] = frame["Adj Close"] / frame["Adj Close"].shift(5) - 1
    frame["20_Day_Return"] = (
        frame["Adj Close"] / frame["Adj Close"].shift(20) - 1
    )
    lagged_close = frame["Adj Close"].shift(1)
    rolling_mean = lagged_close.rolling(20).mean()
    rolling_std = lagged_close.rolling(20).std()
    frame["Dist_From_SMA"] = (
        frame["Adj Close"] - rolling_mean
    ) / rolling_std
    return frame


def _symbol_panel(
    raw: pd.DataFrame, symbol: str, horizon: int, lag: int
) -> tuple[pd.DataFrame, list[str]]:
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if lag < 1:
        raise ValueError("lag must be at least 1")

    engineered = _engineer_features(raw)
    lagged = pd.concat(
        [
            engineered[list(FEATURE_COLUMNS)]
            .shift(offset)
            .add_suffix(f"_{offset}")
            for offset in range(lag)
        ],
        axis=1,
    )
    feature_columns = list(lagged.columns)
    forward_return = (
        engineered["Adj Close"].shift(-horizon) / engineered["Adj Close"] - 1
    )
    panel = pd.concat(
        [
            pd.DataFrame(
                {
                    "Symbol": symbol,
                    "Date": engineered["Date"],
                    "TargetDate": engineered["Date"].shift(-horizon),
                }
            ),
            lagged,
            forward_return.rename("ForwardReturn"),
        ],
        axis=1,
    )
    valid = panel.notna().all(axis=1)
    return panel.loc[valid].reset_index(drop=True), feature_columns


def load_return_panel(
    data_dir: Path = CLEAN_DIR, *, horizon: int = 1, lag: int = 1
) -> tuple[pd.DataFrame, list[str]]:
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No stock CSV files found in {data_dir}")

    panels: list[pd.DataFrame] = []
    feature_columns: list[str] | None = None
    for path in files:
        raw = pd.read_csv(path, parse_dates=["Date"])
        missing = REQUIRED_COLUMNS.difference(raw.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        symbol_panel, columns = _symbol_panel(raw, path.stem, horizon, lag)
        panels.append(symbol_panel)
        feature_columns = columns

    assert feature_columns is not None
    panel = pd.concat(panels, ignore_index=True).sort_values(
        ["Date", "Symbol"]
    )
    return panel.reset_index(drop=True), feature_columns


def with_relative_return_target(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    market_return = result.groupby("Date")["ForwardReturn"].transform("mean")
    result["Target"] = result["ForwardReturn"] - market_return
    return result


def with_volatility_target(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    result["Target"] = result["ForwardReturn"].abs()
    return result


def split_panel(
    panel: pd.DataFrame,
    *,
    val_cutoff: str = VAL_CUTOFF,
    test_cutoff: str = TEST_CUTOFF,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = panel["Date"]
    target_dates = panel["TargetDate"]
    train = panel.loc[
        (dates < val_cutoff) & (target_dates < val_cutoff)
    ].copy()
    validation = panel.loc[
        (dates >= val_cutoff)
        & (dates < test_cutoff)
        & (target_dates < test_cutoff)
    ].copy()
    test = panel.loc[dates >= test_cutoff].copy()
    return train, validation, test


def date_group_splits(
    dates: pd.Series, *, n_splits: int = 5, gap: int = 5
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


def fit_ridge(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    *,
    n_splits: int = 5,
    gap: int = 5,
    degree: int = 1,
    alphas: np.ndarray | None = None,
    n_jobs: int = -1,
) -> tuple[Pipeline, float]:
    alpha_grid = np.logspace(-4, 4, 30) if alphas is None else alphas
    search = GridSearchCV(
        make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            StandardScaler(),
            Ridge(),
        ),
        param_grid={"ridge__alpha": alpha_grid},
        cv=date_group_splits(dates, n_splits=n_splits, gap=gap),
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
    )
    search.fit(X, y)
    return search.best_estimator_, float(search.best_params_["ridge__alpha"])


def prediction_frame(
    model: Pipeline, panel: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    forward_return = (
        panel["ForwardReturn"] if "ForwardReturn" in panel else panel["Target"]
    )
    return pd.DataFrame(
        {
            "Symbol": panel["Symbol"].to_numpy(),
            "Date": panel["Date"].to_numpy(),
            "ForwardReturn": forward_return.to_numpy(),
            "Actual": panel["Target"].to_numpy(),
            "Predicted": model.predict(panel[feature_columns]),
        }
    )


def _metrics(
    actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray
) -> dict[str, Any]:
    mse = float(mean_squared_error(actual, predicted))
    baseline_mse = float(mean_squared_error(actual, baseline))
    actual_std = float(np.std(actual))
    predicted_std = float(np.std(predicted))
    correlation = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if actual_std > 0 and predicted_std > 0
        else None
    )
    return {
        "observations": int(len(actual)),
        "mse": mse,
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "correlation": correlation,
        "prediction_std": predicted_std,
        "baseline_mse": baseline_mse,
        "mse_improvement_vs_baseline_pct": float(
            100 * (baseline_mse - mse) / baseline_mse
        ),
        "beat_baseline": bool(mse < baseline_mse),
    }


def evaluate_regression(
    predictions: pd.DataFrame, training_means: dict[str, float]
) -> dict[str, Any]:
    per_symbol: list[dict[str, Any]] = []
    for symbol, rows in predictions.groupby("Symbol", sort=True):
        actual = rows["Actual"].to_numpy(dtype=float)
        predicted = rows["Predicted"].to_numpy(dtype=float)
        baseline = np.full_like(actual, training_means[symbol])
        per_symbol.append(
            {
                "symbol": symbol,
                **_metrics(actual, predicted, baseline),
            }
        )

    actual = predictions["Actual"].to_numpy(dtype=float)
    predicted = predictions["Predicted"].to_numpy(dtype=float)
    baseline = predictions["Symbol"].map(training_means).to_numpy(dtype=float)
    micro = _metrics(actual, predicted, baseline)
    per_symbol_frame = pd.DataFrame(per_symbol)
    correlations = per_symbol_frame["correlation"].dropna()
    summary = {
        "stock_count": int(len(per_symbol)),
        **micro,
        "micro_r2": micro.pop("r2"),
        "micro_mse": micro.pop("mse"),
        "micro_mae": micro.pop("mae"),
        "macro_r2": float(per_symbol_frame["r2"].mean()),
        "median_r2": float(per_symbol_frame["r2"].median()),
        "macro_correlation": (
            float(correlations.mean()) if len(correlations) else None
        ),
        "stocks_beating_mean_baseline": int(
            per_symbol_frame["beat_baseline"].sum()
        ),
        "stocks_with_positive_r2": int((per_symbol_frame["r2"] > 0).sum()),
    }
    return {"summary": summary, "per_symbol": per_symbol}
