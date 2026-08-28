"""Cleaning: dtypes, naive dates, flags, drop bad rows."""

from __future__ import annotations

import pandas as pd

from ml_trading.cleaning import clean_ohlcv, load_symbol_csv, read_ohlcv_csv


def test_read_maps_yahoo_columns(fixture_dir):
    raw = read_ohlcv_csv(fixture_dir / "SPY.csv")
    assert list(raw.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj_close",
    ]
    # Source files are newest-first; cleaning will reverse that.
    assert raw["date"].iloc[0] == "2011-01-05"


def test_clean_sorts_ascending_and_is_tz_naive(fixture_dir):
    result = load_symbol_csv(fixture_dir / "SPY.csv")
    frame = result.frame
    assert frame.index.is_monotonic_increasing
    assert frame.index.tz is None
    assert str(frame.index.dtype).startswith("datetime64")
    assert frame.index[0] == pd.Timestamp("2010-06-10")
    assert frame.index[-1] == pd.Timestamp("2011-01-05")
    assert frame["adj_close"].dtype.kind == "f"
    assert frame["volume"].dtype.kind in {"i", "u", "f"}


def test_volume_000_becomes_zero_and_bad_rows_drop(dirty_csv):
    raw = read_ohlcv_csv(dirty_csv)
    kept = clean_ohlcv(raw, drop_bad_rows=True)
    # Valid bars: 2010-06-16 (tz stripped) and 2010-06-17.
    # Dropped: duplicate 06-11 (second), high<low 06-10, invalid date,
    # nonpositive 06-14, missing close 06-15. First 06-11 is valid OHLC.
    assert pd.Timestamp("2010-06-11") in kept.frame.index
    assert pd.Timestamp("2010-06-10") not in kept.frame.index
    assert pd.Timestamp("2010-06-14") not in kept.frame.index
    assert pd.Timestamp("2010-06-15") not in kept.frame.index
    assert pd.Timestamp("2010-06-16") in kept.frame.index
    assert pd.Timestamp("2010-06-17") in kept.frame.index
    assert kept.frame.index.tz is None
    assert kept.dropped_rows >= 4

    flagged = clean_ohlcv(raw, drop_bad_rows=False)
    flags = flagged.flags
    assert flags["duplicate_date"].sum() == 1
    assert flags["high_lt_low"].sum() == 1
    assert flags["invalid_date"].sum() == 1
    assert flags["nonpositive_price"].sum() >= 1
    assert flags["missing_required"].sum() >= 1
    # Volume "000" parses as 0, not NaN.
    vol = flagged.frame.loc[flagged.frame.index == pd.Timestamp("2010-06-10"), "volume"]
    assert list(vol) == [0]


def test_duplicate_keeps_earlier_row(dirty_csv):
    raw = read_ohlcv_csv(dirty_csv)
    kept = clean_ohlcv(raw, drop_bad_rows=True).frame
    # File order for the duplicate date is close=10.00 then 10.50. Stable sort
    # keeps that order, and keep='first' keeps 10.00.
    assert kept.loc[pd.Timestamp("2010-06-11"), "close"] == 10.00


def test_clean_does_not_fill_gaps(fixture_dir):
    fake2 = load_symbol_csv(fixture_dir / "FAKE2.csv").frame
    assert pd.Timestamp("2010-06-15") not in fake2.index
    assert fake2["adj_close"].isna().sum() == 0
