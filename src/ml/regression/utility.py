import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ml.config import CLEAN_DIR as DATA_DIR

def print_model_summary(model, y_test: np.array, y_pred: np.array):
    if hasattr(model, "best_params_"):
        print(f"Best params: {model.best_params_}")
        model = model.best_estimator_

    estimator = model[-1] if isinstance(model, Pipeline) else model

    names = getattr(model, "feature_names_in_", None)
    if isinstance(model, Pipeline):
        for _, step in model.steps[:-1]:
            if hasattr(step, "get_feature_names_out"):
                names = step.get_feature_names_out(names)

    if hasattr(estimator, "coef_"):
        coef = np.ravel(estimator.coef_)
        if names is not None and len(names) == len(coef):
            weights = ", ".join(f"{c:.2f} {n}" for c, n in zip(coef, names))
        else:
            weights = ", ".join(f"{c:.2f}" for c in coef)
        print(f"Weights: {weights}")
    if hasattr(estimator, "intercept_"):
        print(f"Intercept: {estimator.intercept_}")

    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"R^2: {r2_score(y_test, y_pred)}")


def load_data(stock_list: list[str]) -> dict[str, pd.DataFrame]:
    data_dict = {}
    for stock in stock_list:
        df = pd.read_csv(DATA_DIR / f"{stock}.csv", parse_dates=["Date"])
        df = df.sort_values("Date")
        df = df.dropna()
        data_dict[stock] = df
    return data_dict


def split_data(df: pd.DataFrame, feature_cols: list[str], cutoff: str, lag: int = 5,) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X = pd.concat(
        [df[feature_cols].shift(i).add_suffix(f"_{i}") for i in range(lag)],
        axis=1
    )
    y = df["Adj Close"].shift(-1) / df["Adj Close"] - 1

    valid = X.notna().all(axis=1) & y.notna()
    y = y.loc[valid]
    X = X.loc[valid]

    dates = df.loc[valid, "Date"]
    train = dates < cutoff
    test = dates >= cutoff
    return X.loc[train], y.loc[train], X.loc[test], y.loc[test]


def lagged_cols(X: pd.DataFrame, features: list[str]) -> list[str]:
    return [c for c in X.columns if any(c.startswith(f"{f}_") for f in features)]


# Derived OHLCV columns (process_OHLCV_*):
# Range                   (High - Low) / Close
# Close Location          (Close - Low) / (High - Low); 0.5 if High == Low
# Upper Wick              (High - body top) / (High - Low)
# Lower Wick              (body bottom - Low) / (High - Low)
# Rel_Vol                 Volume / prior 20-day mean Volume
# Shock_Vol               (Volume - prior 20-day mean) / prior 20-day std
# Signed_Rel_Vol          Rel_Vol * 1-day Adj Close return
# Overnight               split-adjusted Open / prior Adj Close - 1
# Intraday                Close / Open - 1
# {n}_Day_Return          Adj Close / Adj Close.shift(n) - 1  (n = 1, 5, 10, 20)
# Vol_{n}                 rolling std of 1-day Adj Close return (includes today)
# ATR_{n}                 rolling mean of true range / Close (includes today)
# Dist_From_SMA           (Adj Close - prior 20-day SMA) / prior 20-day std
# SMA_Slope               prior 20-day SMA / that SMA from 5 days earlier - 1
# Window_Close_Location   (Adj Close - prior 20-day min) / (prior max - min); 0.5 if flat


def process_OHLCV_bar_shape(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df["Range"] = (df["High"] - df["Low"]) / df["Close"]
    high_low = df["High"] - df["Low"]
    body_top = df[["Open", "Close"]].max(axis=1)
    body_bot = df[["Open", "Close"]].min(axis=1)
    df["Close Location"] = np.where(
        high_low > 0, (df["Close"] - df["Low"]) / high_low, 0.5
    )
    df["Upper Wick"] = np.where(high_low > 0, (df["High"] - body_top) / high_low, 0.0)
    df["Lower Wick"] = np.where(high_low > 0, (body_bot - df["Low"]) / high_low, 0.0)

    feature_cols = ["Range", "Close Location", "Upper Wick", "Lower Wick"]
    return df, feature_cols


def process_OHLCV_volume(df: pd.DataFrame, window: int = 20) -> tuple[pd.DataFrame, list[str]]:
    prior_mean = df["Volume"].rolling(window=window).mean().shift(1)
    prior_std = df["Volume"].rolling(window=window).std().shift(1)
    df["Rel_Vol"] = df["Volume"] / prior_mean.replace(0, np.nan)
    df["Shock_Vol"] = (df["Volume"] - prior_mean) / prior_std.replace(0, np.nan)
    ret_1 = df["Adj Close"] / df["Adj Close"].shift(1) - 1
    df["Signed_Rel_Vol"] = df["Rel_Vol"] * ret_1
    feature_cols = ["Rel_Vol", "Shock_Vol", "Signed_Rel_Vol"]
    return df, feature_cols


def process_OHLCV_standard(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    adjusted_open = df["Open"] * (df["Adj Close"] / df["Close"])
    df["Overnight"] = adjusted_open / df["Adj Close"].shift(1) - 1
    df["Intraday"] = df["Close"] / df["Open"] - 1
    feature_cols = ["Overnight", "Intraday"]
    return df, feature_cols


def process_OHLCV_returns(df: pd.DataFrame, window: int = 5) -> tuple[pd.DataFrame, list[str]]:
    df[f"{window}_Day_Return"] = (df["Adj Close"] / df["Adj Close"].shift(window)) - 1
    feature_cols = [f"{window}_Day_Return"]
    return df, feature_cols


def process_OHLCV_volatility(df: pd.DataFrame, window: int = 20) -> tuple[pd.DataFrame, list[str]]:
    ret_1 = df["Adj Close"] / df["Adj Close"].shift(1) - 1
    df[f"Vol_{window}"] = ret_1.rolling(window=window).std()
    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df[f"ATR_{window}"] = true_range.rolling(window=window).mean() / df["Close"].replace(
        0, np.nan
    )
    feature_cols = [f"Vol_{window}", f"ATR_{window}"]
    return df, feature_cols


def process_OHLCV_dist_from_sma(
    df: pd.DataFrame, window: int = 20, slope_span: int = 5
) -> tuple[pd.DataFrame, list[str]]:
    lagged = df["Adj Close"].shift(1)
    rolling_mean = lagged.rolling(window=window).mean()
    rolling_std = lagged.rolling(window=window).std()
    df["Dist_From_SMA"] = (df["Adj Close"] - rolling_mean) / rolling_std.replace(0, np.nan)
    df["SMA_Slope"] = rolling_mean / rolling_mean.shift(slope_span) - 1
    feature_cols = ["Dist_From_SMA", "SMA_Slope"]
    return df, feature_cols


def process_OHLCV_channel(df: pd.DataFrame, window: int = 20) -> tuple[pd.DataFrame, list[str]]:
    lagged = df["Adj Close"].shift(1)
    roll_min = lagged.rolling(window=window).min()
    roll_max = lagged.rolling(window=window).max()
    span = roll_max - roll_min
    df["Window_Close_Location"] = np.where(
        span > 0,
        (df["Adj Close"] - roll_min) / span,
        np.where(span.eq(0), 0.5, np.nan),
    )
    feature_cols = ["Window_Close_Location"]
    return df, feature_cols


def process_OHLCV_all(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df, bar_cols = process_OHLCV_bar_shape(df)
    df, vol_cols = process_OHLCV_volume(df)
    df, standard_cols = process_OHLCV_standard(df)
    df, return_1_cols = process_OHLCV_returns(df, window=1)
    df, return_5_cols = process_OHLCV_returns(df, window=5)
    df, return_10_cols = process_OHLCV_returns(df, window=10)
    df, return_20_cols = process_OHLCV_returns(df, window=20)
    df, vol_regime_cols = process_OHLCV_volatility(df)
    df, sma_cols = process_OHLCV_dist_from_sma(df)
    df, channel_cols = process_OHLCV_channel(df)
    feature_cols = (
        bar_cols
        + vol_cols
        + standard_cols
        + return_1_cols
        + return_5_cols
        + return_10_cols
        + return_20_cols
        + vol_regime_cols
        + sma_cols
        + channel_cols
    )
    return df, feature_cols