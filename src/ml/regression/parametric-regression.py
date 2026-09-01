from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from enum import Enum
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ml.config import CLEAN_DIR as DATA_DIR, CUTOFF, STARTING_STOCKS

def train_model(X: np.array, y: np.array, n_splits: int = 5, gap: int = 5, degree: int = 1) -> Pipeline:

    alphas = np.logspace(-4, 4, 30)

    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        gap=gap
    )

    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        # StandardScaler() - Causes data leakage because it scales the data based on the entire training set, meaning the timesplits have future data
        RidgeCV(
            alphas=alphas,
            cv=tscv,
            scoring="neg_mean_squared_error",
        )
    )

    model.fit(X, y)

    ridge = model.named_steps["ridgecv"]
    print(f"Best alpha: {ridge.alpha_}")
    
    return model


def load_data(stock_list: list[str]) -> dict[str, pd.DataFrame]:
    data_dict = {}
    for stock in stock_list:
        df = pd.read_csv(DATA_DIR / f"{stock}.csv", parse_dates=["Date"])
        df = df.sort_values("Date")
        df = df.dropna()
        data_dict[stock] = df
    return data_dict


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df["Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Overnight"] = (df["Open"] / df["Close"].shift(1)) - 1
    df["Relative_Volume"] = (
        df["Volume"] / df["Volume"].rolling(20).mean().shift(1)
    )
    df["5_Day_Return"] = (df["Adj Close"] / df["Adj Close"].shift(5)) - 1
    df["20_Day_Return"] = (df["Adj Close"] / df["Adj Close"].shift(20)) - 1
    df["Dist_From_SMA"] = (df["Adj Close"] - df["Adj Close"].shift(1).rolling(20).mean()) / df["Adj Close"].shift(1).rolling(20).std()
    
    return df


def print_model_summary(model: Pipeline, y_test: np.array, y_pred: np.array):
    regressor = model.named_steps["ridgecv"]
    poly = model.named_steps["polynomialfeatures"]
    feature_names = poly.get_feature_names_out(model.feature_names_in_)
    weights = ", ".join(
        f"{coeff:.2f} {name}" for coeff, name in zip(regressor.coef_, feature_names)
    )
    print(f"Weights: {weights}")
    print(f"Intercept: {regressor.intercept_}")
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"R^2: {r2_score(y_test, y_pred)}")


def split_data(df: pd.DataFrame, lag: int = 5) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    feature_cols = ["Open", "High", "Low", "Relative_Volume", "Range", "Overnight", "5_Day_Return", "20_Day_Return", "Dist_From_SMA", "Adj Close"]

    X = pd.concat(
        [df[feature_cols].shift(i).add_suffix(f"_{i}") for i in range(lag)],
        axis=1
    )
    y = df["Adj Close"].shift(-1) / df["Adj Close"] - 1

    valid = X.notna().all(axis=1) & y.notna()
    y = y.loc[valid]
    X = X.loc[valid]

    dates = df.loc[valid, "Date"]
    train = dates < CUTOFF
    test = dates >= CUTOFF
    return X.loc[train], y.loc[train], X.loc[test], y.loc[test]


def plot_data(title: str, X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.array):
    fig, (ax_time, ax_scatter) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)
    x = np.arange(len(y_test))
    ax_time.plot(x, y_test.values, label="Actual", color="tab:blue", linewidth=1)
    ax_time.plot(x, y_pred, label="Predicted", color="tab:orange", linewidth=1)
    ax_time.set_xlabel("Test sample (time order)")
    ax_time.set_ylabel("Next-day return")
    ax_time.set_title("Actual vs Predicted Returns")
    ax_time.legend()

    ax_scatter.scatter(y_test.values, y_pred, alpha=0.4, s=10, color="tab:blue")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax_scatter.plot(lims, lims, color="tab:red", linestyle="--", linewidth=1, label="y = x")
    ax_scatter.set_xlabel("Actual")
    ax_scatter.set_ylabel("Predicted")
    ax_scatter.set_title("Predicted vs Actual")
    ax_scatter.legend()

    fig.tight_layout()
    plt.show()


def main():
    n_splits = 5
    gap = 5
    lag = 1
    degree = 1
    display_plots = False

    print("Loading data...")
    data_dict = load_data(STARTING_STOCKS)

    print("Original SPY data: \n", data_dict["SPY"].head())
    df = data_dict["SPY"]

    print("Processing data...")
    df = process_data(df)

    print("Splitting data...")
    X_train, y_train, X_test, y_test = split_data(df, lag=lag)

    print("Training model...")
    model = train_model(X_train, y_train, n_splits=n_splits, gap=gap, degree=degree)
    y_train_pred = model.predict(X_train)
    print("Training model summary:")
    print_model_summary(model, y_train, y_train_pred)
    
    y_pred = model.predict(X_test)
    print("Test model summary:")
    print_model_summary(model, y_test, y_pred)

    if display_plots:
        plot_data("Training data", X_train, y_train, y_train_pred)
        plot_data("Test data", X_test, y_test, y_pred)

if __name__ == "__main__":
    main()