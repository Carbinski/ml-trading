from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from itertools import combinations
import utility as ut

BASE_FEATURE_COLS = [
    "1_Day_Return",
    "Overnight",
    "Range"
]

CANDIDATE_GROUPS = {
    "bar_shape": ["Close Location", "Upper Wick", "Lower Wick"],
    "volume": ["Rel_Vol", "Shock_Vol"],
    "returns": ["5_Day_Return", "10_Day_Return", "20_Day_Return"],
    "sma": ["Dist_From_SMA"]
}

DISPLAY_PLOTS = True
LAG = 1

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ml.config import CLEAN_DIR as DATA_DIR, CUTOFF, STARTING_STOCKS

def train_model(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, gap: int = 5) -> GridSearchCV:
    
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    
    pipe = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor()
    )

    search = GridSearchCV(
        pipe,
        param_grid={"kneighborsregressor__n_neighbors": [3, 5, 11, 21, 51]},
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


def plot_data(title: str, X: pd.DataFrame, y: pd.Series, y_pred: np.ndarray) -> None:
    y_true = y.to_numpy()
    y_hat = np.asarray(y_pred).ravel()
    mse = mean_squared_error(y_true, y_hat)
    r2 = r2_score(y_true, y_hat)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"kNN | {title}  |  MSE={mse:.6g}  R²={r2:.4f}")

    t = np.arange(len(y_true))
    ax_time = axes[0, 0]
    ax_time.plot(t, y_true, label="Actual", color="tab:blue", linewidth=0.8, alpha=0.85)
    ax_time.plot(t, y_hat, label="kNN predicted", color="tab:orange", linewidth=1.2)
    ax_time.set_xlabel("Sample (time order)")
    ax_time.set_ylabel("Next-day return")
    ax_time.set_title("Actual vs predicted")
    ax_time.legend()

    ax_scatter = axes[0, 1]
    ax_scatter.scatter(y_true, y_hat, alpha=0.35, s=10, color="tab:blue")
    lims = [min(y_true.min(), y_hat.min()), max(y_true.max(), y_hat.max())]
    ax_scatter.plot(lims, lims, color="tab:red", linestyle="--", linewidth=1, label="y = x")
    ax_scatter.set_xlabel("Actual")
    ax_scatter.set_ylabel("Predicted")
    ax_scatter.set_title("Predicted vs actual")
    ax_scatter.legend()

    ax_hist = axes[1, 0]
    ax_hist.hist(y_true, bins=40, density=True, alpha=0.55, label="Actual", color="tab:blue")
    ax_hist.hist(y_hat, bins=40, density=True, alpha=0.55, label="Predicted", color="tab:orange")
    ax_hist.axvline(0.0, color="gray", linewidth=0.8)
    ax_hist.set_xlabel("Next-day return")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Return distribution (kNN shrinkage)")
    ax_hist.legend()

    ax_local = axes[1, 1]
    if {"Adj Close_0", "Adj Close_1"}.issubset(X.columns):
        recent_return = (X["Adj Close_0"] / X["Adj Close_1"]) - 1
        ax_local.scatter(recent_return, y_true, alpha=0.25, s=8, color="tab:blue", label="Actual")
        ax_local.scatter(recent_return, y_hat, alpha=0.45, s=10, color="tab:orange", label="kNN predicted")
        ax_local.axhline(0.0, color="gray", linewidth=0.8)
        ax_local.axvline(0.0, color="gray", linewidth=0.8)
        ax_local.set_xlabel("Same-window 1-day return (Adj Close_0 / Adj Close_1 - 1)")
        ax_local.set_ylabel("Next-day return")
        ax_local.set_title("Local structure in recent-return space")
        ax_local.legend()
    else:
        ax_local.plot(t, y_true - y_hat, color="tab:purple", linewidth=0.9)
        ax_local.axhline(0.0, color="gray", linewidth=0.8)
        ax_local.set_xlabel("Sample (time order)")
        ax_local.set_ylabel("Residual")
        ax_local.set_title("Residuals")

    fig.tight_layout()
    plt.show()


def evaluate_folds(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, gap: int = 5) -> tuple[float, float]:
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    mse = []
    r2 = []
    for train, test in tscv.split(X):
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        model = train_model(X_train, y_train, n_splits=n_splits, gap=gap)
        y_test_pred = model.predict(X_test)
        mse.append(mean_squared_error(y_test, y_test_pred))
        r2.append(r2_score(y_test, y_test_pred))
    return np.mean(mse), np.mean(r2)


def select_features(X: pd.DataFrame, y: pd.Series) -> list[str]:
    candidates = list(dict.fromkeys(
        c for group in CANDIDATE_GROUPS.values() for c in group
    ))
    n_trials = 2 ** len(candidates)
    best_trial = list(BASE_FEATURE_COLS)
    best_mse = None
    trial_i = 0
    for r in range(len(candidates) + 1):
        for subset in combinations(candidates, r):
            trial_i += 1
            trial = BASE_FEATURE_COLS + list(subset)
            cols = ut.lagged_cols(X, trial)
            fold_mse, fold_r2 = evaluate_folds(X[cols], y)
            label = ",".join(subset) if subset else "(base)"
            print(f"{trial_i}/{n_trials} | {label:60s} | MSE={fold_mse:.6g}  R²={fold_r2:.4f}")
            if best_mse is None or fold_mse < best_mse:
                best_mse = fold_mse
                best_trial = trial
    return best_trial

def main():

    print("Loading data...")
    data_dict = ut.load_data(STARTING_STOCKS)

    print("Processing SPY data...")
    df = data_dict["SPY"]
    df, _ = process_data(df)

    all_features = list(dict.fromkeys(
        BASE_FEATURE_COLS + [c for group in CANDIDATE_GROUPS.values() for c in group]
    ))

    print("Splitting data...")
    X_train, y_train, X_test, y_test = ut.split_data(df, all_features, CUTOFF, lag=LAG)

    print("Selecting features...")
    selected_features = select_features(X_train, y_train)
    print(f"Selected features: {selected_features}")
    cols = ut.lagged_cols(X_train, selected_features)
    X_train, X_test = X_train[cols], X_test[cols]

    model = train_model(X_train, y_train, n_splits=5, gap=5)
    y_train_pred = model.predict(X_train)
    print("Training model summary:")
    ut.print_model_summary(model, y_train, y_train_pred)

    y_pred = model.predict(X_test)
    print("Test model summary:")
    ut.print_model_summary(model, y_test, y_pred)

    if DISPLAY_PLOTS:
        plot_data("Training data", X_train, y_train, y_train_pred)
        plot_data("Test data", X_test, y_test, y_pred)

if __name__ == "__main__":
    main()