"""Time split: earlier years/days in train, later in test, no leak."""

from __future__ import annotations

import pandas as pd
import pytest

from ml_trading.loader import load_prices
from ml_trading.split import assert_no_time_leak, time_split


def test_split_by_year_cutoff(fixture_dir):
    prices = load_prices(["SPY", "FAKE1", "FAKE2"], data_dir=fixture_dir)
    split = time_split(prices, cutoff="2011-01-01")
    train, test = split
    assert split.cutoff == pd.Timestamp("2011-01-01")
    assert train.index.max() == pd.Timestamp("2010-12-31")
    assert test.index.min() == pd.Timestamp("2011-01-03")
    assert train.index.max() < test.index.min()
    assert_no_time_leak(train, test)
    # No shuffle: both sides stay chronological.
    assert train.index.is_monotonic_increasing
    assert test.index.is_monotonic_increasing
    assert list(train.index) == sorted(train.index)
    assert pd.Timestamp("2011-01-03") not in train.index
    assert pd.Timestamp("2010-12-31") not in test.index


def test_cutoff_in_train(fixture_dir):
    prices = load_prices(["SPY"], data_dir=fixture_dir)
    split = time_split(prices, cutoff="2010-12-31", cutoff_in="train")
    assert split.train.index.max() == pd.Timestamp("2010-12-31")
    assert split.test.index.min() == pd.Timestamp("2011-01-03")
    assert_no_time_leak(split.train, split.test)


def test_cutoff_in_test_includes_boundary(fixture_dir):
    prices = load_prices(["SPY"], data_dir=fixture_dir)
    split = time_split(prices, cutoff="2010-12-31", cutoff_in="test")
    assert split.train.index.max() == pd.Timestamp("2010-12-30")
    assert split.test.index.min() == pd.Timestamp("2010-12-31")


def test_split_does_not_shuffle_or_kfold(fixture_dir):
    prices = load_prices(["FAKE1"], data_dir=fixture_dir)
    train, test = time_split(prices, cutoff="2011-01-01")
    # Reconstructing along time equals the original — no row mixing.
    rebuilt = pd.concat([train, test])
    pd.testing.assert_frame_equal(rebuilt, prices)


def test_assert_no_time_leak_catches_overlap():
    idx = pd.to_datetime(["2010-01-01", "2010-01-02", "2010-01-03"])
    frame = pd.DataFrame({"x": [1, 2, 3]}, index=idx)
    leaked_train = frame.iloc[:2]
    leaked_test = frame.iloc[1:]
    with pytest.raises(ValueError, match="Time leak"):
        assert_no_time_leak(leaked_train, leaked_test)
