from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ml.config import CUTOFF, STARTING_STOCKS
from src.ml import utility as ut

BASE_FEATURE_COLS = [
    "1_Day_Return",
    "Overnight",
    "Range",
    "Close_Location",
    "Rel_Vol",
]

DISPLAY_PLOTS = True
LAG = 1

# Standalone SPY settings; no joint GridSearchCV on the VotingRegressor.
RIDGE_ALPHAS = np.logspace(-4, 4, 30)
SVR_KERNEL = "rbf"
SVR_C = 1.0
SVR_EPSILON = 0.01
TREE_MAX_DEPTH = 2
KNN_NEIGHBORS = 51


def train_model(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, gap: int = 5) -> VotingRegressor:
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    ridge = make_pipeline(
        StandardScaler(),
        RidgeCV(
            alphas=RIDGE_ALPHAS,
            cv=tscv,
            scoring="neg_mean_squared_error",
        ),
    )
    svr = make_pipeline(
        StandardScaler(),
        SVR(kernel=SVR_KERNEL, C=SVR_C, epsilon=SVR_EPSILON, cache_size=2000, max_iter=16000),
    )
    tree = make_pipeline(
        DecisionTreeRegressor(max_depth=TREE_MAX_DEPTH, random_state=42)
    )
    knn = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=KNN_NEIGHBORS),
    )
    model = VotingRegressor(
        estimators=[("ridge", ridge), ("svr", svr), ("tree", tree), ("knn", knn)],
    )
    model.fit(X, y)
    ridge_cv = model.named_estimators_["ridge"].named_steps["ridgecv"]
    print(f"Ridge best alpha: {ridge_cv.alpha_}")
    return model

def process_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.sort_values("Date")
    df = df.dropna()

    df, feature_cols = ut.process_OHLCV_all(df)

    return df, feature_cols


def plot_data(
    title: str,
    y: pd.Series,
    y_pred: np.ndarray,
    model: VotingRegressor | None = None,
    X: pd.DataFrame | None = None,
) -> None:
    y_true = y.to_numpy()
    y_hat = np.asarray(y_pred).ravel()
    mse = mean_squared_error(y_true, y_hat)
    r2 = r2_score(y_true, y_hat)

    fig, (ax_time, ax_scatter) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Ensemble | {title}  |  MSE={mse:.6g}  R²={r2:.4f}")

    t = np.arange(len(y_true))
    ax_time.plot(t, y_true, label="Actual", color="tab:blue", linewidth=0.8, alpha=0.85)

    member_colors = ("tab:green", "tab:purple", "tab:brown", "tab:pink")
    if model is not None and X is not None and hasattr(model, "named_estimators_"):
        for (name, est), color in zip(model.named_estimators_.items(), member_colors):
            member_hat = np.asarray(est.predict(X)).ravel()
            ax_time.plot(t, member_hat, label=name, color=color, linewidth=0.8, alpha=0.55)
            ax_scatter.scatter(y_true, member_hat, alpha=0.2, s=8, color=color, label=name)

    ax_time.plot(t, y_hat, label="Ensemble predicted", color="tab:orange", linewidth=1.2)
    ax_time.set_xlabel("Sample (time order)")
    ax_time.set_ylabel("Next-day return")
    ax_time.set_title("Actual vs predicted")
    ax_time.legend()

    ax_scatter.scatter(y_true, y_hat, alpha=0.45, s=12, color="tab:orange", label="ensemble")
    lims = [min(y_true.min(), y_hat.min()), max(y_true.max(), y_hat.max())]
    ax_scatter.plot(lims, lims, color="tab:red", linestyle="--", linewidth=1, label="y = x")
    slope, intercept = np.polyfit(y_true, y_hat, 1)
    fit_x = np.array(lims)
    ax_scatter.plot(
        fit_x,
        slope * fit_x + intercept,
        color="tab:cyan",
        linewidth=1.4,
        label=f"Ensemble fit: ŷ = {slope:.2f}y + {intercept:.4g}",
    )
    ax_scatter.set_xlabel("Actual")
    ax_scatter.set_ylabel("Predicted")
    ax_scatter.set_title("Predicted vs actual")
    ax_scatter.legend()

    fig.tight_layout()
    plt.show()


def main():

    print("Loading data...")
    data_dict = ut.load_data(STARTING_STOCKS)

    print("Processing SPY data...")
    df = data_dict["SPY"]
    df, _ = process_data(df)

    print("Splitting data...")
    X_train, y_train, X_test, y_test = ut.split_data(df, BASE_FEATURE_COLS, CUTOFF, lag=LAG)

    ensemble = train_model(X_train, y_train, n_splits=5, gap=5)

    y_train_pred = ensemble.predict(X_train)
    print("Training model summary:")
    ut.print_model_summary(ensemble, y_train, y_train_pred)

    y_pred = ensemble.predict(X_test)
    print("Test model summary:")
    ut.print_model_summary(ensemble, y_test, y_pred)

    if DISPLAY_PLOTS:
        plot_data("Training data", y_train, y_train_pred, ensemble, X_train)
        plot_data("Test data", y_test, y_pred, ensemble, X_test)


if __name__ == "__main__":
    main()