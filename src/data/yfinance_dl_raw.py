import yfinance as yf
import pandas as pd
import os
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ml.config import DOWNLOAD_PAUSE_S, PERIOD, RAW_DIR, STARTING_STOCKS

"""
YFINANCE DOWNLOAD

The goal of this script is to download data from Yahoo Finance and save it to a CSV file.

Our starting set of stocks will be:
- SPY, AAPL, MSFT, GOOG, AMZN, META, NVDA, IBM, JPM, XOM, KO, PG, K

We will be downloading 10 years of daily data for each stock.
"""

def download_data(ticker: str, period: str = PERIOD) -> pd.DataFrame:
    """Download data for a given ticker and period."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    data = yf.download(ticker, period=period, actions=True, auto_adjust=False, repair=True, multi_level_index=False, progress=True)
    data.to_csv(RAW_DIR / f"{ticker}.csv")
    return data



def main():
    # Download data for each stock
    for stock in STARTING_STOCKS:
        path = RAW_DIR / f"{stock}.csv"
        if (os.path.exists(path)):
            print(f"Data for {stock} already exists")
            continue
        else:
            download_data(stock)
            print(f"Downloaded data for {stock}")
            time.sleep(DOWNLOAD_PAUSE_S)
    print("All data downloaded successfully")

    # Process the data and save to a new CSV file


if __name__ == "__main__":
    main()