from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from time import perf_counter
from pathlib import Path

BASE_FEATURE_COLS = [
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

DISPLAY_PLOTS = True
LAG = 1

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ml.config import CUTOFF
from src.ml import utility as ut


def train_model(X: pd.DataFrame, y: pd.Series, n_splits: int = 3, gap: int = 5) -> GridSearchCV:
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    param_grid = {
        "svr__C": [0.01, 0.1, 1],
        "svr__gamma": [0.01, 0.1, "scale"],
    }
    n_candidates = len(param_grid["svr__C"]) * len(param_grid["svr__gamma"])
    n_fits = n_candidates * n_splits

    print(
        f"Grid search: SVR RBF, C={param_grid['svr__C']}, "
        f"gamma={param_grid['svr__gamma']}, {n_splits} folds, "
        f"{n_fits} fits, n={len(X)}, d={X.shape[1]}"
    )

    pipe = make_pipeline(
        StandardScaler(),
        SVR(
            kernel="rbf",
            epsilon=0.01,
            cache_size=500,
            max_iter=16000,
            tol=1e-3,
        ),
    )

    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=3,
        refit=True,
    )

    t0 = perf_counter()
    search.fit(X, y)
    print(f"Grid search finished in {perf_counter() - t0:.1f}s")
    print(f"Best params: {search.best_params_}  |  CV MSE: {-search.best_score_:.6g}")
    return search


def process_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.sort_values("Date")
    df = df.dropna()

    df, feature_cols = ut.process_OHLCV_all(df)

    return df, feature_cols


def plot_data(title: str, y: pd.Series, y_pred: np.ndarray) -> None:
    y_true = y.to_numpy()
    y_hat = np.asarray(y_pred).ravel()
    mse = mean_squared_error(y_true, y_hat)
    r2 = r2_score(y_true, y_hat)

    fig, (ax_time, ax_scatter) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"SVM | {title}  |  MSE={mse:.6g}  R²={r2:.4f}")

    t = np.arange(len(y_true))
    ax_time.plot(t, y_true, label="Actual", color="tab:blue", linewidth=0.8, alpha=0.85)
    ax_time.plot(t, y_hat, label="SVM predicted", color="tab:orange", linewidth=1.2)
    ax_time.set_xlabel("Sample (time order)")
    ax_time.set_ylabel("Next-day return")
    ax_time.set_title("Actual vs predicted")
    ax_time.legend()

    ax_scatter.scatter(y_true, y_hat, alpha=0.35, s=10, color="tab:blue")
    lims = [min(y_true.min(), y_hat.min()), max(y_true.max(), y_hat.max())]
    ax_scatter.plot(lims, lims, color="tab:red", linestyle="--", linewidth=1, label="y = x")
    slope, intercept = np.polyfit(y_true, y_hat, 1)
    fit_x = np.array(lims)
    ax_scatter.plot(
        fit_x,
        slope * fit_x + intercept,
        color="tab:green",
        linewidth=1.4,
        label=f"Fit: ŷ = {slope:.2f}y + {intercept:.4g}",
    )
    ax_scatter.set_xlabel("Actual")
    ax_scatter.set_ylabel("Predicted")
    ax_scatter.set_title("Predicted vs actual")
    ax_scatter.legend()

    fig.tight_layout()
    plt.show()


def main():

    print("Loading SPY...")
    data_dict = ut.load_data(["SPY"])

    print("Processing SPY data...")
    df = data_dict["SPY"]
    df, _ = process_data(df)

    print("Splitting data...")
    X_train, y_train, X_test, y_test = ut.split_data(df, BASE_FEATURE_COLS, CUTOFF, lag=LAG)
    print(f"Train {len(X_train)} rows  |  Test {len(X_test)} rows  |  {X_train.shape[1]} features")

    print("Training kernel SVR...")
    model = train_model(X_train, y_train, n_splits=3, gap=5)
    svr = model.best_estimator_.named_steps["svr"]
    print(f"n_support: {svr.support_.size}")

    print("Predicting train set...")
    y_train_pred = model.predict(X_train)
    print("Training model summary:")
    ut.print_model_summary(model, y_train, y_train_pred)

    print("Predicting test set...")
    y_pred = model.predict(X_test)
    print("Test model summary:")
    ut.print_model_summary(model, y_test, y_pred)

    if DISPLAY_PLOTS:
        plot_data("Training data", y_train, y_train_pred)
        plot_data("Test data", y_test, y_pred)


if __name__ == "__main__":
    main()
