"""
data_ingestion.py
-----------------
Fetches and maintains a rolling 10-year window of HDFC Bank daily price data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TICKER         = "HDFCBANK.NS"   # NSE ticker via yfinance
TRADING_DAYS   = 2520            # ~10 years
CACHE_FILE     = Path("data/hdfc_prices.parquet")


def fetch_raw(ticker: str = TICKER, years: int = 10) -> pd.DataFrame:
    """Download raw OHLCV + Adj Close from yfinance."""
    end   = datetime.today()
    start = end - timedelta(days=years * 365 + 60)   # +60 buffer for weekends/holidays
    logger.info(f"Downloading {ticker} from {start.date()} to {end.date()} …")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker or internet connection.")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    logger.info(f"  → {len(df)} raw rows fetched.")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing values, drop rows where Close is still NaN."""
    # Flatten multi-level columns (new yfinance versions return ticker as extra level)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].copy()
    # Ensure Close is a flat numeric Series, not nested
    if isinstance(df["Close"].iloc[0], pd.Series):
        df["Close"] = df["Close"].apply(lambda x: x.iloc[0])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df


def rolling_window(df: pd.DataFrame, window: int = TRADING_DAYS) -> pd.DataFrame:
    """Keep only the most recent `window` trading days."""
    if len(df) > window:
        df = df.iloc[-window:]
    return df


def load_or_fetch(ticker: str = TICKER) -> pd.DataFrame:
    """
    Load cached data if available; otherwise fetch fresh.
    Appends today's data and trims to rolling window.
    """
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists():
        logger.info("Loading cached data …")
        df = pd.read_parquet(CACHE_FILE)
        last_date = df.index[-1].date()
        today     = datetime.today().date()

        if last_date < today:
            logger.info(f"Cache is stale (last: {last_date}). Fetching incremental update …")
            new_df = fetch_raw(ticker, years=1)   # only last year for speed
            new_df = clean(new_df)
            df = pd.concat([df, new_df])
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)
        else:
            logger.info("Cache is up-to-date.")
    else:
        logger.info("No cache found. Performing full 10-year fetch …")
        df = fetch_raw(ticker, years=10)
        df = clean(df)

    df = rolling_window(df)
    df.to_parquet(CACHE_FILE)
    logger.info(f"Data ready: {len(df)} rows  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


def compute_returns(df: pd.DataFrame) -> pd.Series:
    """Compute daily log returns from Close prices."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    return log_ret


if __name__ == "__main__":
    df  = load_or_fetch()
    ret = compute_returns(df)
    print(df.tail())
    print(f"\nMean daily log return : {ret.mean():.6f}")
    print(f"Daily volatility (σ)  : {ret.std():.6f}")
