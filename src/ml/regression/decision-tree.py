from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

BASE_FEATURE_COLS = [
    "1_Day_Return",
    "Overnight",
    "Range",
    "Upper_Wick",
    "Lower_Wick",
]

DISPLAY_PLOTS = True
LAG = 1

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ml.config import CLEAN_DIR as DATA_DIR, CUTOFF, STARTING_STOCKS
from src.ml import utility as ut

def train_model(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, gap: int = 5) -> GridSearchCV:
    
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    
    pipe = make_pipeline(
        DecisionTreeRegressor(random_state=42)
    )

    search = GridSearchCV(
        pipe,
        param_grid={"decisiontreeregressor__max_depth": [2, 4, 6, 8, 10, 15, 20, None]},
        cv=tscv,
        scoring="neg_mean_squared_error"
    )

    search.fit(X, y)
    return search


def process_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.sort_values("Date")
    df = df.dropna()

    df, feature_cols = ut.process_OHLCV_all(df)

    return df, feature_cols


def plot_data(
    title: str,
    tree: DecisionTreeRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    y_pred: np.ndarray,
) -> None:
    y_true = y.to_numpy()
    y_hat = np.asarray(y_pred).ravel()
    mse = mean_squared_error(y_true, y_hat)
    r2 = r2_score(y_true, y_hat)

    fig = plt.figure(figsize=(16, 11), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    ax_tree = fig.add_subplot(gs[0, :])
    ax_time = fig.add_subplot(gs[1, 0])
    ax_scatter = fig.add_subplot(gs[1, 1])
    fig.suptitle(f"Decision Tree | {title}  |  MSE={mse:.6g}  R²={r2:.4f}")

    plot_tree(
        tree,
        feature_names=list(X.columns),
        filled=True,
        rounded=True,
        impurity=False,
        fontsize=8,
        ax=ax_tree,
    )
    ax_tree.set_title("Fitted tree")

    t = np.arange(len(y_true))
    ax_time.plot(t, y_true, label="Actual", color="tab:blue", linewidth=0.8, alpha=0.85)
    ax_time.plot(t, y_hat, label="Tree predicted", color="tab:orange", linewidth=1.2)
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
    plt.show()


def main():

    print("Loading data...")
    data_dict = ut.load_data(STARTING_STOCKS)

    print("Processing SPY data...")
    df = data_dict["SPY"]
    df, _ = process_data(df)

    print("Splitting data...")
    X_train, y_train, X_test, y_test = ut.split_data(df, BASE_FEATURE_COLS, CUTOFF, lag=LAG)

    model = train_model(X_train, y_train, n_splits=5, gap=5)
    tree = model.best_estimator_.named_steps["decisiontreeregressor"]
    y_train_pred = model.predict(X_train)
    print("Training model summary:")
    ut.print_model_summary(model, y_train, y_train_pred)

    y_pred = model.predict(X_test)
    print("Test model summary:")
    ut.print_model_summary(model, y_test, y_pred)

    if DISPLAY_PLOTS:
        plot_data("Training data", tree, X_train, y_train, y_train_pred)
        plot_data("Test data", tree, X_test, y_test, y_pred)

if __name__ == "__main__":
    main()