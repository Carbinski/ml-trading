from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ml.config import CLEAN_DIR as DATA_DIR, CUTOFF, STARTING_STOCKS

# IMPROVEMENT: post-2024 holdout is validation; later window is the untouched test set
VAL_CUTOFF = CUTOFF
TEST_CUTOFF = "2025-09-01"
HORIZON = 1
COST = 1e-4

# IMPROVEMENT: add one stationary feature group at a time
FEATURE_GROUPS = [
    ["Ret_1"],
    ["Overnight", "Intraday"],
    ["Range", "Close_Location"],
    ["Relative_Volume"],
    ["Vol_20", "5_Day_Return", "20_Day_Return", "Dist_From_SMA"],
]


def purge_idx(train_idx: np.ndarray, test_idx: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    # IMPROVEMENT: drop train rows whose next-day label uses the fold's test close
    return train_idx[train_idx <= test_idx.min() - horizon]


def group_cols(columns: pd.Index, features: list[str]) -> list[str]:
    return [c for c in columns if any(c.startswith(f"{name}_") for name in features)]


def train_model(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, gap: int = 5, degree: int = 1, verbose: bool = True) -> Pipeline:
    alphas = np.logspace(-4, 4, 30)
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    # IMPROVEMENT: GridSearchCV tunes scaler + Ridge on time-series folds (scaler never sees future-fold rows)
    search = GridSearchCV(
        make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            StandardScaler(),
            Ridge(),
        ),
        param_grid={"ridge__alpha": alphas},
        cv=tscv,
        scoring="neg_mean_squared_error",
    )
    search.fit(X, y)

    if verbose:
        print(f"Best alpha: {search.best_params_['ridge__alpha']}")
    return search.best_estimator_


def load_data(stock_list: list[str]) -> dict[str, pd.DataFrame]:
    data_dict = {}
    for stock in stock_list:
        df = pd.read_csv(DATA_DIR / f"{stock}.csv", parse_dates=["Date"])
        df = df.sort_values("Date")
        df = df.dropna()
        data_dict[stock] = df
    return data_dict


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    # IMPROVEMENT: no raw price levels; only stationary features
    df["Range"] = (df["High"] - df["Low"]) / df["Close"]
    adj_open = df["Open"] * (df["Adj Close"] / df["Close"])
    # IMPROVEMENT: overnight uses split/dividend-adjusted open vs prior adj close
    df["Overnight"] = adj_open / df["Adj Close"].shift(1) - 1
    df["Intraday"] = df["Close"] / df["Open"] - 1
    hl = df["High"] - df["Low"]
    df["Close_Location"] = np.where(hl > 0, (df["Close"] - df["Low"]) / hl, 0.5)
    df["Relative_Volume"] = (
        df["Volume"] / df["Volume"].rolling(20).mean().shift(1)
    )
    # IMPROVEMENT: 1-day adjusted return (missing autoregressive feature)
    df["Ret_1"] = df["Adj Close"] / df["Adj Close"].shift(1) - 1
    df["Vol_20"] = df["Ret_1"].rolling(20).std()
    df["5_Day_Return"] = (df["Adj Close"] / df["Adj Close"].shift(5)) - 1
    df["20_Day_Return"] = (df["Adj Close"] / df["Adj Close"].shift(20)) - 1
    df["Dist_From_SMA"] = (df["Adj Close"] - df["Adj Close"].shift(1).rolling(20).mean()) / df["Adj Close"].shift(1).rolling(20).std()
    return df


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric, n_boot: int = 500):
    rng = np.random.default_rng(0)
    stats = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric(y_true[idx], y_pred[idx]))
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


def print_model_summary(model: Pipeline, y_test: np.ndarray, y_pred: np.ndarray, y_ref_mean: float):
    regressor = model.named_steps["ridge"]
    poly = model.named_steps["polynomialfeatures"]
    feature_names = poly.get_feature_names_out(model.feature_names_in_)
    # IMPROVEMENT: print coefficients past two decimals so small weights are visible
    weights = ", ".join(
        f"{coeff:.6f} {name}" for coeff, name in zip(regressor.coef_, feature_names)
    )
    print(f"Weights: {weights}")
    print(f"Intercept: {regressor.intercept_}")

    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse}")
    print(f"R^2: {r2}")
    print(f"MSE mean-baseline: {mean_squared_error(y_test, np.full_like(y_test, y_ref_mean, dtype=float))}")
    print(f"MSE zero-baseline: {mean_squared_error(y_test, np.zeros_like(y_test, dtype=float))}")

    # IMPROVEMENT: dispersion, correlation, directional accuracy, turnover, costs, CIs
    pos = np.sign(y_pred)
    turnover = float(np.mean(np.abs(np.diff(pos, prepend=0.0))))
    mse_lo, mse_hi = bootstrap_ci(y_test, y_pred, mean_squared_error)
    r2_lo, r2_hi = bootstrap_ci(y_test, y_pred, r2_score)
    print(f"Pred std: {y_pred.std():.6g}  Corr: {np.corrcoef(y_test, y_pred)[0, 1]:.4f}  Dir acc: {np.mean(np.sign(y_pred) == np.sign(y_test)):.4f}")
    print(f"Turnover: {turnover:.4f}  Cost (1bp): {turnover * COST:.6g}")
    print(f"MSE 95% CI: [{mse_lo:.6g}, {mse_hi:.6g}]  R^2 95% CI: [{r2_lo:.4f}, {r2_hi:.4f}]")


def split_data(df: pd.DataFrame, lag: int = 5) -> tuple:
    # IMPROVEMENT: after-close execution — same-day High/Low/Close/Volume are known before the next-day return
    feature_cols = [
        "Ret_1", "Overnight", "Intraday", "Range", "Close_Location",
        "Relative_Volume", "Vol_20", "5_Day_Return", "20_Day_Return", "Dist_From_SMA",
    ]

    X = pd.concat(
        [df[feature_cols].shift(i).add_suffix(f"_{i}") for i in range(lag)],
        axis=1
    )
    y = df["Adj Close"].shift(-1) / df["Adj Close"] - 1

    valid = X.notna().all(axis=1) & y.notna()
    y = y.loc[valid]
    X = X.loc[valid]

    dates = df.loc[valid, "Date"]
    train = dates < VAL_CUTOFF
    val = (dates >= VAL_CUTOFF) & (dates < TEST_CUTOFF)
    test = dates >= TEST_CUTOFF
    # IMPROVEMENT: purge rows whose next-day target close sits in the following split
    next_dates = dates.shift(-1)
    train = train & (next_dates < VAL_CUTOFF)
    val = val & (next_dates < TEST_CUTOFF)
    return X.loc[train], y.loc[train], X.loc[val], y.loc[val], X.loc[test], y.loc[test]


def evaluate_expanding(X: pd.DataFrame, y: pd.Series, n_splits: int, gap: int, degree: int):
    # IMPROVEMENT: expanding-window folds, each compared to mean and zero-return baselines
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    print("Fold | MSE | R2 | MSE_mean | MSE_zero | beat_mean | beat_zero")
    for i, (tr, te) in enumerate(tscv.split(X), 1):
        tr = purge_idx(tr, te)
        model = train_model(X.iloc[tr], y.iloc[tr], n_splits=min(3, n_splits), gap=gap, degree=degree, verbose=False)
        pred = model.predict(X.iloc[te])
        y_te = y.iloc[te].to_numpy()
        y_mean = float(y.iloc[tr].mean())
        mse = mean_squared_error(y_te, pred)
        mse_mean = mean_squared_error(y_te, np.full_like(y_te, y_mean, dtype=float))
        mse_zero = mean_squared_error(y_te, np.zeros_like(y_te, dtype=float))
        print(
            f"{i} | {mse:.6g} | {r2_score(y_te, pred):.4f} | "
            f"{mse_mean:.6g} | {mse_zero:.6g} | {mse < mse_mean} | {mse < mse_zero}"
        )


def plot_data(title: str, X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray):
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
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df, lag=lag)
    print(f"Train < {VAL_CUTOFF} | Val [{VAL_CUTOFF}, {TEST_CUTOFF}) | Test >= {TEST_CUTOFF}")
    print(f"n train={len(X_train)}  n val={len(X_val)}  n test={len(X_test)}")

    used: list[str] = []
    for group in FEATURE_GROUPS:
        used.extend(group)
        cols = group_cols(X_train.columns, used)
        print(f"\nFeature groups so far: {used}")
        evaluate_expanding(X_train[cols], y_train, n_splits=n_splits, gap=gap, degree=degree)

    print("\nTraining model...")
    model = train_model(X_train, y_train, n_splits=n_splits, gap=gap, degree=degree)
    y_train_pred = model.predict(X_train)
    y_ref_mean = float(y_train.mean())
    print("Training model summary:")
    print_model_summary(model, y_train, y_train_pred, y_ref_mean)

    y_val_pred = model.predict(X_val)
    print("Validation model summary:")
    print_model_summary(model, y_val, y_val_pred, y_ref_mean)

    y_pred = model.predict(X_test)
    print("Test model summary:")
    print_model_summary(model, y_test, y_pred, y_ref_mean)

    if display_plots:
        plot_data("Training data", X_train, y_train, y_train_pred)
        plot_data("Validation data", X_val, y_val, y_val_pred)
        plot_data("Test data", X_test, y_test, y_pred)

if __name__ == "__main__":
    main()
