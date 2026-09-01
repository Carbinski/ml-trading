import yfinance as yf
import pandas as pd
import os
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ml.config import CLEAN_DIR, OHLCV_COLUMNS, RAW_DIR

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Want the following data: Date,Open,High,Low,Close,Volume,Adj Close
    df = df[OHLCV_COLUMNS]
    if df.isna().sum().any():
        print("Missing values: \n", df.isna().sum())
        df = df.dropna()

    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(by="Date", inplace=True)

    return df
    

def main():
    for path in RAW_DIR.iterdir():
        if path.suffix == ".csv":
            df = pd.read_csv(path)
            df = clean_data(df)
            df.to_csv(CLEAN_DIR / path.name, index=False)

if __name__ == "__main__":
    main()