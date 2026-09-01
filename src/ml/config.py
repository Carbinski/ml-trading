from __future__ import annotations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = REPO_ROOT / "data" / "raw-yfinance"
CLEAN_DIR = REPO_ROOT / "data" / "clean-yfinance"

# 80-20 train/test split on the current 10y daily window
CUTOFF = "2024-09-01"

# yfinance download lookback
PERIOD = "10y"

DOWNLOAD_PAUSE_S = 0.5

STARTING_STOCKS = [
    "SPY",
    "AAPL",
    "MSFT",
    "GOOG",
    "AMZN",
    "META",
    "NVDA",
    "IBM",
    "JPM",
    "XOM",
    "KO",
    "PG",
    "UNH",
    "JNJ",
    "CAT",
    "UNP",
    "V",
    "NEE",
    "LIN",
    "PLD",
]

OHLCV_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Adj Close",
]
