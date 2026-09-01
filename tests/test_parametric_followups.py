from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

try:
    common = importlib.import_module("src.ml.regression.parametric_experiment_common")
except ModuleNotFoundError:
    common = None


def require_common():
    assert common is not None, "parametric_experiment_common must be implemented"
    return common


def make_prices(periods: int = 35) -> pd.DataFrame:
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


def test_load_return_panel_builds_requested_forward_horizon(tmp_path):
    module = require_common()
    prices = make_prices()
    prices.to_csv(tmp_path / "TEST.csv", index=False)

    panel, feature_columns = module.load_return_panel(tmp_path, horizon=5)

    first = panel.iloc[0]
    start = prices.loc[prices["Date"] == first["Date"], "Adj Close"].iloc[0]
    end = prices.loc[prices["Date"] == first["TargetDate"], "Adj Close"].iloc[0]
    assert first["Symbol"] == "TEST"
    assert first["ForwardReturn"] == end / start - 1
    assert len(feature_columns) == 10


def test_relative_target_is_zero_mean_within_each_date():
    module = require_common()
    panel = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
            ),
            "ForwardReturn": [0.02, 0.00, -0.01, 0.03],
        }
    )

    result = module.with_relative_return_target(panel)

    assert np.allclose(result.groupby("Date")["Target"].mean(), 0.0)
    assert np.allclose(result["Target"], [0.01, -0.01, -0.02, 0.02])


def test_volatility_target_is_absolute_next_day_return():
    module = require_common()
    panel = pd.DataFrame({"ForwardReturn": [-0.03, 0.02]})

    result = module.with_volatility_target(panel)

    assert np.allclose(result["Target"], [0.03, 0.02])


def test_split_panel_purges_targets_crossing_boundaries():
    module = require_common()
    dates = pd.date_range("2024-01-01", periods=7)
    panel = pd.DataFrame(
        {
            "Date": dates[:-1],
            "TargetDate": dates[1:],
            "Symbol": "TEST",
            "Target": np.arange(6),
        }
    )

    train, validation, test = module.split_panel(
        panel, val_cutoff="2024-01-04", test_cutoff="2024-01-06"
    )

    assert train["Date"].dt.day.tolist() == [1, 2]
    assert validation["Date"].dt.day.tolist() == [4]
    assert test["Date"].dt.day.tolist() == [6]


def test_date_group_splits_keep_each_market_date_in_one_fold():
    module = require_common()
    dates = pd.Series(np.repeat(pd.bdate_range("2024-01-01", periods=12), 2))

    splits = module.date_group_splits(dates, n_splits=3, gap=1)

    assert len(splits) == 3
    for train_idx, test_idx in splits:
        train_dates = set(dates.iloc[train_idx])
        test_dates = set(dates.iloc[test_idx])
        assert train_dates.isdisjoint(test_dates)
        assert max(train_dates) < min(test_dates)


def test_fit_ridge_returns_a_model_and_selected_alpha():
    module = require_common()
    dates = pd.Series(np.repeat(pd.bdate_range("2024-01-01", periods=15), 2))
    X = pd.DataFrame({"signal": np.linspace(-1, 1, len(dates))})
    y = pd.Series(0.5 * X["signal"] + 0.01)

    model, alpha = module.fit_ridge(
        X,
        y,
        dates,
        n_splits=2,
        gap=1,
        alphas=np.array([0.1, 1.0]),
        n_jobs=1,
    )

    assert alpha in {0.1, 1.0}
    assert model.predict(X).shape == (len(X),)


def test_prediction_frame_preserves_symbol_date_and_actual_values():
    module = require_common()
    dates = pd.Series(pd.bdate_range("2024-01-01", periods=20))
    X = pd.DataFrame({"signal": np.linspace(-1, 1, len(dates))})
    y = pd.Series(0.5 * X["signal"] + 0.01)
    model, _ = module.fit_ridge(
        X,
        y,
        dates,
        n_splits=2,
        gap=1,
        alphas=np.array([0.1]),
        n_jobs=1,
    )
    panel = X.assign(Symbol="TEST", Date=dates, Target=y)

    result = module.prediction_frame(model, panel, ["signal"])

    assert result.columns.tolist() == [
        "Symbol",
        "Date",
        "ForwardReturn",
        "Actual",
        "Predicted",
    ]
    assert result["Symbol"].unique().tolist() == ["TEST"]
    assert np.allclose(result["Actual"], y)


def test_evaluate_regression_compares_against_per_symbol_training_means():
    module = require_common()
    predictions = pd.DataFrame(
        {
            "Symbol": ["A", "A", "B", "B"],
            "Actual": [1.0, -1.0, 2.0, -2.0],
            "Predicted": [0.5, -0.5, 1.0, -1.0],
        }
    )

    result = module.evaluate_regression(
        predictions, training_means={"A": 0.0, "B": 0.0}
    )

    assert result["summary"]["stock_count"] == 2
    assert result["summary"]["observations"] == 4
    assert result["summary"]["micro_r2"] == 0.75
    assert result["summary"]["stocks_beating_mean_baseline"] == 2
