"""Shared names and the pinned Jackovich homework-data source."""

from __future__ import annotations

# Yahoo-style columns as they appear in the Jackovich CSVs (confirmed).
RAW_OHLCV_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume", "Adj Close")

# Canonical names used inside this repo after cleaning.
CANONICAL_COLUMNS = ("date", "open", "high", "low", "close", "volume", "adj_close")

RAW_TO_CANONICAL = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Adj Close": "adj_close",
}

PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close")
NUMERIC_COLUMNS = PRICE_COLUMNS + ("volume",)

# Pinned commit so fetches are reproducible even if master moves.
JACKOVICH_OWNER = "JeffreyJackovich"
JACKOVICH_REPO = "machine-learning-for-trading"
JACKOVICH_REF = "08763f80649604158040dd1559925b3b90c97d32"
JACKOVICH_DATA_DIR = "Part 1-manipulating_financial_data_in_python/data"

# Small default pull. Use --all for the ~1005-file, ~170MB dump.
STARTER_SYMBOLS = ("SPY", "FAKE1", "FAKE2", "AAPL", "IBM", "GOOG")

FILL_NONE = "none"
FILL_FFILL = "ffill"
ALLOWED_FILL = (FILL_NONE, FILL_FFILL)
