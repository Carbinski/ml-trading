"""Load one or many symbols onto a shared trading calendar.

Missing-bar policy (intentionally strict):

1. The calendar is the set of dates in `calendar_symbol` (default **SPY**),
   not weekdays and not a union of sparse tickers. Weekends and exchange
   holidays are already absent from SPY in the Jackovich dump.
2. Other symbols are reindexed onto that calendar. Dates the CSV simply
   omitted become NaN.
3. Default fill is **none**. The only other option is `ffill` (copy the last
   known past bar). There is no bfill and no interpolation — those would
   leak the future into the past.
4. Rows are timezone-naive and sorted ascending before any fill.

This is the ML4T `get_data` idea (join onto dates, drop days SPY did not
trade) without the curriculum's backward-fill step.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from ml_trading.cleaning import load_symbol_csv
from ml_trading.constants import ALLOWED_FILL, FILL_FFILL, FILL_NONE, PRICE_COLUMNS


def symbol_csv_path(data_dir: str | Path, symbol: str) -> Path:
    return Path(data_dir) / f"{symbol}.csv"


def load_symbol(symbol: str, data_dir: str | Path, *, drop_bad_rows: bool = True) -> pd.DataFrame:
    """Clean OHLCV for one symbol, indexed by timezone-naive date."""
    path = symbol_csv_path(data_dir, symbol)
    if not path.is_file():
        raise FileNotFoundError(
            f"No CSV for {symbol!r} at {path}. "
            f"Fetch homework data with: python -m ml_trading.fetch --symbols {symbol}"
        )
    return load_symbol_csv(path, drop_bad_rows=drop_bad_rows).frame


def _as_naive_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def trading_calendar(
    data_dir: str | Path,
    *,
    calendar_symbol: str = "SPY",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """SPY (or another symbol) trading days, ascending, timezone-naive."""
    days = load_symbol(calendar_symbol, data_dir).index
    calendar = pd.DatetimeIndex(days)
    if calendar.tz is not None:
        calendar = calendar.tz_convert("UTC").tz_localize(None)
    calendar = calendar.sort_values()
    calendar = calendar[~calendar.duplicated()]
    if start is not None:
        calendar = calendar[calendar >= _as_naive_timestamp(start)]
    if end is not None:
        calendar = calendar[calendar <= _as_naive_timestamp(end)]
    calendar.name = "date"
    return calendar


def _apply_fill(frame: pd.DataFrame, fill: str | None) -> pd.DataFrame:
    policy = FILL_NONE if fill in (None, "") else str(fill).lower()
    if policy not in ALLOWED_FILL:
        raise ValueError(
            f"Unsupported fill={fill!r}. Allowed: {ALLOWED_FILL}. "
            "bfill / interpolation are rejected because they fill from the future."
        )
    if policy == FILL_FFILL:
        # Past -> present only. Leading NaNs stay NaN (no invented history).
        return frame.ffill()
    return frame


def _load_frames(
    symbols: Sequence[str],
    data_dir: str | Path,
    *,
    drop_bad_rows: bool,
) -> dict[str, pd.DataFrame]:
    if not symbols:
        raise ValueError("Pass at least one symbol.")
    return {symbol: load_symbol(symbol, data_dir, drop_bad_rows=drop_bad_rows) for symbol in symbols}


def _align_frame(frame: pd.DataFrame, calendar: pd.DatetimeIndex, fill: str | None) -> pd.DataFrame:
    aligned = frame.reindex(calendar)
    aligned.index.name = "date"
    return _apply_fill(aligned, fill)


def load_panel(
    symbols: Sequence[str],
    *,
    data_dir: str | Path,
    calendar_symbol: str | None = "SPY",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    fill: str | None = FILL_NONE,
    drop_bad_rows: bool = True,
    fields: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Date-aligned OHLCV panel.

    Columns are a MultiIndex `(field, symbol)` so `panel['adj_close']` is the
    ML4T-style prices frame. Index is the trading calendar.
    """
    wanted = list(dict.fromkeys(symbols))  # preserve order, drop dupes
    load_list = list(wanted)
    if calendar_symbol and calendar_symbol not in load_list:
        load_list.insert(0, calendar_symbol)

    frames = _load_frames(load_list, data_dir, drop_bad_rows=drop_bad_rows)

    if calendar_symbol:
        calendar = trading_calendar(
            data_dir, calendar_symbol=calendar_symbol, start=start, end=end
        )
    else:
        calendar = pd.DatetimeIndex([])
        for frame in frames.values():
            calendar = calendar.union(frame.index)
        calendar = pd.DatetimeIndex(calendar)
        if calendar.tz is not None:
            calendar = calendar.tz_convert("UTC").tz_localize(None)
        calendar = calendar.sort_values()
        if start is not None:
            calendar = calendar[calendar >= _as_naive_timestamp(start)]
        if end is not None:
            calendar = calendar[calendar <= _as_naive_timestamp(end)]
        calendar.name = "date"

    use_fields = list(fields) if fields is not None else list(PRICE_COLUMNS) + ["volume"]
    pieces = []
    for symbol in wanted:
        aligned = _align_frame(frames[symbol], calendar, fill)
        missing = [c for c in use_fields if c not in aligned.columns]
        if missing:
            raise KeyError(f"{symbol} missing fields {missing}")
        piece = aligned.loc[:, use_fields].copy()
        piece.columns = pd.MultiIndex.from_product(
            [piece.columns, [symbol]], names=["field", "symbol"]
        )
        pieces.append(piece)
    return pd.concat(pieces, axis=1)


def load_prices(
    symbols: Sequence[str],
    *,
    data_dir: str | Path,
    field: str = "adj_close",
    calendar_symbol: str | None = "SPY",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    fill: str | None = FILL_NONE,
    drop_bad_rows: bool = True,
) -> pd.DataFrame:
    """Date x symbol frame for one field (default adjusted close)."""
    panel = load_panel(
        symbols,
        data_dir=data_dir,
        calendar_symbol=calendar_symbol,
        start=start,
        end=end,
        fill=fill,
        drop_bad_rows=drop_bad_rows,
        fields=[field],
    )
    prices = panel[field].copy()
    prices.index.name = "date"
    return prices
