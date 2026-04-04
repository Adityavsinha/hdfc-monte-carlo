"""
scripts/data_ingestion.py
-------------------------
Fetches live Nifty 50 composition from NSE + historical OHLCV data.
Auto-updates whenever Nifty 50 composition changes.
"""

import requests
import yfinance as yf
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from config import NIFTY50_SYMBOLS, SECTOR_MAP

logger = logging.getLogger(__name__)


# ── Live Nifty 50 Composition ──────────────────
def fetch_live_nifty50() -> list[dict]:
    """
    Fetches current Nifty 50 stocks from NSE India API.
    Falls back to hardcoded list if API fails.
    Returns list of {symbol, name, sector, isin}.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com"
    }

    try:
        session = requests.Session()
        # First hit the main page to get cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
        r   = session.get(url, headers=headers, timeout=15)
        data = r.json()

        stocks = []
        for item in data.get("data", [])[1:]:  # skip index row
            sym = item.get("symbol", "").strip()
            if sym:
                stocks.append({
                    "symbol": sym,
                    "name"  : item.get("meta", {}).get("companyName", sym),
                    "sector": item.get("meta", {}).get("industry", SECTOR_MAP.get(sym, "Other")),
                    "isin"  : item.get("meta", {}).get("isin", ""),
                    "source": "live_nse"
                })

        if len(stocks) >= 45:
            logger.info(f"✅ Live Nifty 50 fetched: {len(stocks)} stocks")
            return stocks

    except Exception as e:
        logger.warning(f"NSE API failed: {e} — using fallback list")

    # Fallback to hardcoded list
    return [{
        "symbol": s,
        "name"  : s,
        "sector": SECTOR_MAP.get(s, "Other"),
        "isin"  : "",
        "source": "fallback"
    } for s in NIFTY50_SYMBOLS]


# ── Historical Data Fetch ──────────────────────
def fetch_stock_data(symbol: str, years: int = 10) -> pd.DataFrame | None:
    """
    Fetches 10 years of daily OHLCV + Adj Close from yfinance.
    Handles NSE symbols (adds .NS suffix).
    Returns cleaned DataFrame or None on failure.
    """
    ticker = symbol if "." in symbol else f"{symbol}.NS"

    try:
        end   = datetime.today()
        start = end - timedelta(days=years * 365 + 90)

        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)

        if df.empty or len(df) < 200:
            # Try BSE suffix as fallback
            ticker_bse = f"{symbol}.BO"
            df = yf.download(ticker_bse, start=start, end=end,
                             auto_adjust=True, progress=False)

        if df.empty or len(df) < 200:
            logger.warning(f"  {symbol}: insufficient data ({len(df)} rows)")
            return None

        # Flatten multi-level columns (new yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure numeric
        for col in ["Open","High","Low","Close","Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].squeeze(), errors="coerce")

        df.ffill(inplace=True)
        df.dropna(subset=["Close"], inplace=True)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        logger.info(f"  {symbol}: {len(df)} rows | "
                    f"{df.index[0].date()} → {df.index[-1].date()} | "
                    f"₹{float(df['Close'].iloc[-1]):,.2f}")
        return df

    except Exception as e:
        logger.error(f"  {symbol}: fetch failed — {e}")
        return None


# ── Financial Info ─────────────────────────────
def fetch_financial_info(symbol: str) -> dict:
    """
    Fetches key financial metrics from yfinance.
    P/E, Market Cap, EPS, Revenue, Dividend Yield, ROE, etc.
    """
    ticker = symbol if "." in symbol else f"{symbol}.NS"
    result = {}

    try:
        info = yf.Ticker(ticker).info
        result = {
            "market_cap"    : info.get("marketCap"),
            "pe_ratio"      : info.get("trailingPE"),
            "pb_ratio"      : info.get("priceToBook"),
            "eps"           : info.get("trailingEps"),
            "revenue"       : info.get("totalRevenue"),
            "net_income"    : info.get("netIncomeToCommon"),
            "roe"           : info.get("returnOnEquity"),
            "roa"           : info.get("returnOnAssets"),
            "debt_equity"   : info.get("debtToEquity"),
            "current_ratio" : info.get("currentRatio"),
            "dividend_yield": info.get("dividendYield"),
            "beta"          : info.get("beta"),
            "book_value"    : info.get("bookValue"),
            "52w_high"      : info.get("fiftyTwoWeekHigh"),
            "52w_low"       : info.get("fiftyTwoWeekLow"),
            "avg_volume"    : info.get("averageVolume"),
            "shares_out"    : info.get("sharesOutstanding"),
            "sector"        : info.get("sector", SECTOR_MAP.get(symbol, "Other")),
            "industry"      : info.get("industry", ""),
            "description"   : info.get("longBusinessSummary", "")[:300] if info.get("longBusinessSummary") else "",
            "employees"     : info.get("fullTimeEmployees"),
            "founded"       : info.get("founded"),
            "website"       : info.get("website", ""),
        }
    except Exception as e:
        logger.warning(f"  {symbol}: financial info failed — {e}")

    return result


# ── Feature Engineering ────────────────────────
def compute_features(df: pd.DataFrame) -> dict:
    """
    Computes all derived features from OHLCV data.
    Returns dict of Series/values.
    """
    close = df["Close"].squeeze()

    log_ret = np.log(close / close.shift(1)).dropna()

    # Rolling statistics
    roll21  = log_ret.rolling(21)
    roll63  = log_ret.rolling(63)
    roll252 = log_ret.rolling(252)

    # Momentum indicators
    mom_1m  = (close.iloc[-1] / close.iloc[-21]  - 1) if len(close) > 21  else 0
    mom_3m  = (close.iloc[-1] / close.iloc[-63]  - 1) if len(close) > 63  else 0
    mom_6m  = (close.iloc[-1] / close.iloc[-126] - 1) if len(close) > 126 else 0
    mom_1y  = (close.iloc[-1] / close.iloc[-252] - 1) if len(close) > 252 else 0

    # EWMA volatility (more responsive)
    ewma_vol = log_ret.ewm(span=63).std().iloc[-1]

    # Rolling max drawdown
    rolling_max   = close.cummax()
    drawdown      = (close - rolling_max) / rolling_max
    max_drawdown  = float(drawdown.min())

    # Average True Range (ATR proxy)
    if "High" in df.columns and "Low" in df.columns:
        atr = (df["High"].squeeze() - df["Low"].squeeze()).rolling(14).mean().iloc[-1]
    else:
        atr = None

    return {
        "log_returns"   : log_ret,
        "close"         : close,
        "current_price" : float(close.iloc[-1]),
        "mu_daily"      : float(log_ret.mean()),
        "sigma_daily"   : float(ewma_vol),
        "sigma_annual"  : float(ewma_vol * np.sqrt(252)),
        "vol_21"        : float(roll21.std().iloc[-1]) if len(log_ret) > 21  else float(ewma_vol),
        "vol_63"        : float(roll63.std().iloc[-1]) if len(log_ret) > 63  else float(ewma_vol),
        "vol_252"       : float(roll252.std().iloc[-1]) if len(log_ret) > 252 else float(ewma_vol),
        "mom_1m"        : float(mom_1m),
        "mom_3m"        : float(mom_3m),
        "mom_6m"        : float(mom_6m),
        "mom_1y"        : float(mom_1y),
        "max_drawdown"  : max_drawdown,
        "atr"           : float(atr) if atr is not None else None,
        "week52_high"   : float(close.iloc[-252:].max()) if len(close) >= 252 else float(close.max()),
        "week52_low"    : float(close.iloc[-252:].min()) if len(close) >= 252 else float(close.min()),
        "avg_volume_30" : float(df["Volume"].squeeze().iloc[-30:].mean()) if "Volume" in df.columns else None,
    }
