"""Loader: SPY calendar, alignment, missing-bar policy, no future fill."""

from __future__ import annotations

import pandas as pd
import pytest

from ml_trading.loader import load_panel, load_prices, load_symbol, trading_calendar


def test_load_symbol_fixture(fixture_dir):
    spy = load_symbol("SPY", fixture_dir)
    fake1 = load_symbol("FAKE1", fixture_dir)
    fake2 = load_symbol("FAKE2", fixture_dir)
    assert len(spy) == 12
    assert len(fake1) == 12
    assert len(fake2) == 6
    assert spy.index.tz is None
    assert list(spy.index) == list(fake1.index)


def test_calendar_is_spy_dates_not_weekends(fixture_dir):
    calendar = trading_calendar(fixture_dir)
    assert calendar.tz is None
    assert all(day.weekday() < 5 for day in calendar)
    # Excerpt is not a contiguous year — still no Saturday/Sunday.
    assert pd.Timestamp("2010-06-12") not in calendar  # Saturday
    assert pd.Timestamp("2010-06-13") not in calendar  # Sunday
    assert pd.Timestamp("2011-01-01") not in calendar  # Saturday New Year
    assert pd.Timestamp("2010-06-14") in calendar
    assert pd.Timestamp("2011-01-03") in calendar


def test_alignment_introduces_nan_for_fake2_holes(fixture_dir):
    prices = load_prices(["SPY", "FAKE1", "FAKE2"], data_dir=fixture_dir, fill="none")
    assert list(prices.columns) == ["SPY", "FAKE1", "FAKE2"]
    assert list(prices.index) == list(trading_calendar(fixture_dir))
    assert prices["SPY"].isna().sum() == 0
    assert prices["FAKE1"].isna().sum() == 0
    missing = prices["FAKE2"].isna()
    assert bool(missing.loc["2010-06-15"])
    assert bool(missing.loc["2010-06-18"])
    assert bool(missing.loc["2010-12-31"])
    assert not bool(missing.loc["2010-06-14"])
    assert not bool(missing.loc["2011-01-03"])
    # Default policy: holes stay empty. Never invent a bar.
    assert prices.loc["2010-06-15", "FAKE2"] != prices.loc["2010-06-14", "FAKE2"]
    assert pd.isna(prices.loc["2010-06-15", "FAKE2"])


def test_ffill_uses_past_not_future(fixture_dir):
    raw = load_prices(["FAKE2"], data_dir=fixture_dir, fill="none")
    filled = load_prices(["FAKE2"], data_dir=fixture_dir, fill="ffill")
    past = raw.loc["2010-06-14", "FAKE2"]
    future = raw.loc["2011-01-03", "FAKE2"]
    assert past == pytest.approx(19.18)
    assert future == pytest.approx(26.13)
    assert filled.loc["2010-06-15", "FAKE2"] == pytest.approx(past)
    assert filled.loc["2010-12-31", "FAKE2"] == pytest.approx(past)
    # A future-looking fill would copy 26.13 backward onto June/December holes.
    assert filled.loc["2010-06-15", "FAKE2"] != pytest.approx(future)
    assert filled.loc["2011-01-03", "FAKE2"] == pytest.approx(future)


def test_bfill_is_rejected(fixture_dir):
    with pytest.raises(ValueError, match="future"):
        load_prices(["FAKE2"], data_dir=fixture_dir, fill="bfill")


def test_panel_multiindex(fixture_dir):
    panel = load_panel(["SPY", "FAKE2"], data_dir=fixture_dir)
    assert panel.columns.names == ["field", "symbol"]
    assert "adj_close" in panel.columns.get_level_values("field")
    assert panel["adj_close"]["SPY"].isna().sum() == 0
    assert pd.isna(panel["adj_close"].loc["2010-06-15", "FAKE2"])


def test_start_end_slice_does_not_look_ahead(fixture_dir):
    prices = load_prices(
        ["SPY", "FAKE2"],
        data_dir=fixture_dir,
        start="2010-06-10",
        end="2010-06-18",
        fill="none",
    )
    assert prices.index.max() == pd.Timestamp("2010-06-18")
    assert pd.Timestamp("2011-01-03") not in prices.index
    assert pd.isna(prices.loc["2010-06-15", "FAKE2"])
