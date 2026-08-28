"""Read Yahoo-style OHLCV CSVs and clean them without looking ahead.

The Jackovich files are newest-first, timezone-naive `YYYY-MM-DD` dates, and
columns `Date, Open, High, Low, Close, Volume, Adj Close`. Dead tickers
sometimes store volume as `000`. This module:

* maps those columns to snake_case
* forces timezone-naive midnight timestamps
* coerces numeric dtypes
* flags (and optionally drops) unusable rows
* sorts **ascending** by date so later steps cannot accidentally read the future
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ml_trading.constants import (
    CANONICAL_COLUMNS,
    NUMERIC_COLUMNS,
    PRICE_COLUMNS,
    RAW_OHLCV_COLUMNS,
    RAW_TO_CANONICAL,
)


@dataclass
class CleaningResult:
    """Cleaned frame plus the flag table (one row per sorted input bar)."""

    frame: pd.DataFrame
    flags: pd.DataFrame
    dropped_rows: int = 0
    notes: list[str] = field(default_factory=list)


def read_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Read one per-symbol CSV as strings. Cleaning happens in `clean_ohlcv`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    # Read as strings so `000` volumes and mixed newlines do not surprise us.
    raw = pd.read_csv(path, dtype=str, skipinitialspace=True)
    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in RAW_OHLCV_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"{path.name} is not a Yahoo-style OHLCV CSV. "
            f"Missing columns {missing}; found {list(raw.columns)}."
        )
    return raw.loc[:, list(RAW_OHLCV_COLUMNS)].rename(columns=RAW_TO_CANONICAL)


def _parse_naive_dates(values: pd.Series) -> pd.Series:
    """Parse dates and drop any timezone. Date-only strings stay midnight naive."""
    parsed = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
    return parsed.dt.tz_localize(None).dt.normalize()


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.replace(",", "", regex=False)
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce")


def flag_ohlcv_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Boolean flags for rows that should not be trusted as a trading bar."""
    flags = pd.DataFrame(index=frame.index)
    flags["invalid_date"] = frame["date"].isna()
    flags["duplicate_date"] = frame["date"].duplicated(keep="first") & ~flags["invalid_date"]
    flags["high_lt_low"] = frame["high"] < frame["low"]
    price_nonpos = pd.DataFrame(
        {col: frame[col] <= 0 for col in PRICE_COLUMNS if col in frame.columns}
    )
    flags["nonpositive_price"] = price_nonpos.any(axis=1)
    flags["missing_required"] = frame.loc[:, list(NUMERIC_COLUMNS)].isna().any(axis=1)
    flags["any_bad"] = flags.any(axis=1)
    return flags


def clean_ohlcv(
    frame: pd.DataFrame,
    *,
    drop_bad_rows: bool = True,
    source: str | Path | None = None,
) -> CleaningResult:
    """Normalize dtypes, flag bad rows, sort ascending, never fill gaps.

    Missing bars are **not** invented here. Alignment onto a trading calendar
    (and optional past-only ffill) happens in `ml_trading.loader`.
    """
    if frame.empty:
        empty = pd.DataFrame(columns=list(CANONICAL_COLUMNS[1:]))
        empty.index = pd.DatetimeIndex([], name="date")
        flags = pd.DataFrame(
            columns=[
                "invalid_date",
                "duplicate_date",
                "high_lt_low",
                "nonpositive_price",
                "missing_required",
                "any_bad",
            ]
        )
        return CleaningResult(frame=empty, flags=flags, notes=["empty input"])

    work = frame.copy()
    if "date" not in work.columns:
        raise ValueError("Expected a 'date' column after read_ohlcv_csv().")

    work["date"] = _parse_naive_dates(work["date"])
    for col in NUMERIC_COLUMNS:
        work[col] = _to_numeric(work[col])

    # Ascending first so "keep first duplicate" means the earlier bar, not the
    # first row of the newest-first Jackovich file.
    work = work.sort_values("date", kind="mergesort", na_position="last").reset_index(drop=True)
    flags = flag_ohlcv_rows(work)

    notes: list[str] = []
    if source is not None:
        notes.append(f"source={source}")
    dropped = 0
    if drop_bad_rows:
        dropped = int(flags["any_bad"].sum())
        work = work.loc[~flags["any_bad"]].copy()
        notes.append(f"dropped_bad_rows={dropped}")

    work = work.set_index("date")
    work.index.name = "date"
    if work.index.tz is not None:
        work.index = work.index.tz_localize(None)

    # Keep a clean, predictable column order.
    work = work.loc[:, [c for c in CANONICAL_COLUMNS if c != "date"]]
    return CleaningResult(frame=work, flags=flags, dropped_rows=dropped, notes=notes)


def load_symbol_csv(path: str | Path, *, drop_bad_rows: bool = True) -> CleaningResult:
    """Read + clean one `{SYMBOL}.csv` file."""
    path = Path(path)
    raw = read_ohlcv_csv(path)
    return clean_ohlcv(raw, drop_bad_rows=drop_bad_rows, source=path)
