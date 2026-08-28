"""Chronological train/test split. No shuffle, no k-fold across time."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    """Train is always strictly earlier than test."""

    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: pd.Timestamp
    cutoff_in: str

    def __iter__(self):
        yield self.train
        yield self.test


def _naive_cutoff(cutoff) -> pd.Timestamp:
    ts = pd.Timestamp(cutoff)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def assert_no_time_leak(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise if any train date is on or after the first test date."""
    if train.empty or test.empty:
        return
    train_last = pd.Timestamp(train.index.max())
    test_first = pd.Timestamp(test.index.min())
    if train_last >= test_first:
        raise ValueError(
            f"Time leak: train ends at {train_last.date()} but test starts at "
            f"{test_first.date()}. Train must be strictly earlier than test."
        )


def time_split(
    frame: pd.DataFrame,
    cutoff,
    *,
    cutoff_in: str = "test",
) -> TimeSplit:
    """Split a date-indexed frame into earlier train and later test.

    Parameters
    ----------
    frame:
        Must be indexed by dates (timezone-naive preferred).
    cutoff:
        Date boundary, e.g. ``"2011-01-01"``.
    cutoff_in:
        ``"test"`` (default) → train is ``index < cutoff``, test is
        ``index >= cutoff``. The cutoff date, if present, is the first test
        day. ``"train"`` puts the cutoff date in train instead.

    Rows keep their chronological order. This will not shuffle or k-fold.
    """
    if cutoff_in not in {"train", "test"}:
        raise ValueError("cutoff_in must be 'train' or 'test'")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("time_split() expects a DatetimeIndex (dates on the rows).")

    idx = frame.index
    if idx.tz is not None:
        frame = frame.copy()
        frame.index = idx.tz_convert("UTC").tz_localize(None)
        idx = frame.index

    if not idx.is_monotonic_increasing:
        frame = frame.sort_index()
        idx = frame.index

    edge = _naive_cutoff(cutoff)
    if cutoff_in == "test":
        train_mask = idx < edge
        test_mask = idx >= edge
    else:
        train_mask = idx <= edge
        test_mask = idx > edge

    train = frame.loc[train_mask]
    test = frame.loc[test_mask]
    assert_no_time_leak(train, test)
    return TimeSplit(train=train, test=test, cutoff=edge, cutoff_in=cutoff_in)
