"""
scripts/data_ingestion.py
-------------------------
Fetches live Nifty 50 composition from NSE + historical OHLCV data.
Phase 1 fix: Yahoo Finance ticker alias map for symbols that differ from NSE codes.
"""

import requests
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from config import NIFTY50_SYMBOLS, SECTOR_MAP

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
#  YAHOO FINANCE TICKER ALIAS MAP
#  Some NSE symbols differ from Yahoo Finance tickers.
#  Add any symbol that fails with 404 here.
# ══════════════════════════════════════════════════════════════
YF_ALIAS = {
    # NSE symbol  → Yahoo Finance .NS ticker
    "INFOSYS"     : "INFY",
    "TATAMOTORS"  : "TATAMTRDVR",   # fallback; primary tried first
    "M&M"         : "M%26M",
    "BAJAJ-AUTO"  : "BAJAJ-AUTO",
    "BRITANNIA"   : "BRITANNIA",
    "HINDUNILVR"  : "HINDUNILVR",
    "ASIANPAINT"  : "ASIANPAINT",
    "NESTLEIND"   : "NESTLEIND",
    "ULTRACEMCO"  : "ULTRACEMCO",
    "APOLLOHOSP"  : "APOLLOHOSP",
    "TATACONSUM"  : "TATACONSUM",
    "EICHERMOT"   : "EICHERMOT",
    "HEROMOTOCO"  : "HEROMOTOCO",
    "DIVISLAB"    : "DIVISLAB",
    "SHRIRAMFIN"  : "SHRIRAMFIN",
    "INDUSINDBK"  : "INDUSINDBK",
    "ADANIPORTS"  : "ADANIPORTS",
    "ADANIENT"    : "ADANIENT",
}

# Stocks that ONLY work with a specific ticker (skip .NS attempt)
YF_DIRECT = {
    "INFOSYS": "INFY.NS",
}

# ══════════════════════════════════════════════════════════════
#  LIVE NIFTY 50 COMPOSITION
# ══════════════════════════════════════════════════════════════
def fetch_live_nifty50() -> list[dict]:
    """
    Fetches current Nifty 50 stocks from NSE India API.
    Falls back to hardcoded list if API fails.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        url  = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
        r    = session.get(url, headers=headers, timeout=15)
        data = r.json()

        stocks = []
        for item in data.get("data", [])[1:]:
            sym = item.get("symbol", "").strip()
            if sym:
                stocks.append({
                    "symbol": sym,
                    "name"  : item.get("meta", {}).get("companyName", sym),
                    "sector": item.get("meta", {}).get("industry", SECTOR_MAP.get(sym, "Other")),
                    "isin"  : item.get("meta", {}).get("isin", ""),
                    "source": "live_nse",
                })

        if len(stocks) >= 45:
            logger.info(f"✅ Live Nifty 50 fetched: {len(stocks)} stocks")
            return stocks

    except Exception as e:
        logger.warning(f"NSE API failed: {e} — using fallback list")

    return [{
        "symbol": s, "name": s,
        "sector": SECTOR_MAP.get(s, "Other"),
        "isin": "", "source": "fallback",
    } for s in NIFTY50_SYMBOLS]


# ══════════════════════════════════════════════════════════════
#  HISTORICAL DATA FETCH — with alias + fallback chain
# ══════════════════════════════════════════════════════════════
def _download(ticker: str, start, end) -> pd.DataFrame:
    """Single yfinance download attempt, returns empty DF on failure."""
    try:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False, timeout=15)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def fetch_stock_data(symbol: str, years: int = 10) -> pd.DataFrame | None:
    """
    Fetches historical OHLCV from yfinance with a 4-step fallback chain:
      1. Direct alias if defined (e.g. INFY.NS for INFOSYS)
      2. symbol.NS  (standard NSE)
      3. Alias.NS   (from YF_ALIAS map)
      4. symbol.BO  (BSE fallback)
    """
    end   = datetime.today()
    start = end - timedelta(days=years * 365 + 90)

    # Build candidate ticker list
    candidates = []
    if symbol in YF_DIRECT:
        candidates.append(YF_DIRECT[symbol])          # e.g. INFY.NS

    candidates.append(f"{symbol}.NS")                 # standard NSE

    alias = YF_ALIAS.get(symbol)
    if alias and f"{alias}.NS" not in candidates:
        candidates.append(f"{alias}.NS")

    candidates.append(f"{symbol}.BO")                 # BSE fallback

    df = pd.DataFrame()
    used_ticker = None

    for ticker in candidates:
        df = _download(ticker, start, end)
        if not df.empty and len(df) >= 100:
            used_ticker = ticker
            break

    if df.empty or len(df) < 100:
        logger.warning(f"  {symbol}: insufficient data ({len(df)} rows) — tried {candidates}")
        return None

    # Flatten multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].squeeze(), errors="coerce")

    df.ffill(inplace=True)
    df.dropna(subset=["Close"], inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    logger.info(
        f"  {symbol}: {len(df)} rows | "
        f"{df.index[0].date()} → {df.index[-1].date()} | "
        f"₹{float(df['Close'].iloc[-1]):,.2f}"
        + (f" [{used_ticker}]" if used_ticker != f"{symbol}.NS" else "")
    )
    return df


# ══════════════════════════════════════════════════════════════
#  FINANCIAL INFO
# ══════════════════════════════════════════════════════════════
def fetch_financial_info(symbol: str) -> dict:
    """Fetches key financial metrics. Uses alias if needed."""
    # Use the direct alias if available, else standard .NS
    ticker_str = YF_DIRECT.get(symbol, f"{YF_ALIAS.get(symbol, symbol)}.NS")
    result = {}

    try:
        info = yf.Ticker(ticker_str).info
        if not info or info.get("quoteType") == "NONE":
            # Try standard .NS as fallback
            info = yf.Ticker(f"{symbol}.NS").info

        def g(k):
            v = info.get(k)
            if v is None or (isinstance(v, float) and (v != v)):  # NaN check
                return None
            return v

        result = {
            "name"          : info.get("longName") or info.get("shortName") or symbol,
            "market_cap"    : g("marketCap"),
            "pe_ratio"      : g("trailingPE"),
            "pb_ratio"      : g("priceToBook"),
            "eps"           : g("trailingEps"),
            "revenue"       : g("totalRevenue"),
            "net_income"    : g("netIncomeToCommon"),
            "roe"           : g("returnOnEquity"),
            "roa"           : g("returnOnAssets"),
            "debt_equity"   : g("debtToEquity"),
            "current_ratio" : g("currentRatio"),
            "dividend_yield": g("dividendYield"),
            "beta"          : g("beta"),
            "book_value"    : g("bookValue"),
            "sector"        : info.get("sector", SECTOR_MAP.get(symbol, "Other")),
            "industry"      : info.get("industry", ""),
            "description"   : (info.get("longBusinessSummary") or "")[:300],
            "employees"     : info.get("fullTimeEmployees"),
            "website"       : info.get("website", ""),
        }
    except Exception as e:
        logger.debug(f"  {symbol}: financial info failed — {e}")

    return result


# ══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
def compute_features(df: pd.DataFrame) -> dict:
    """Computes all derived features from OHLCV data."""
    close   = df["Close"].squeeze()
    log_ret = np.log(close / close.shift(1)).dropna()

    roll21  = log_ret.rolling(21)
    roll63  = log_ret.rolling(63)
    roll252 = log_ret.rolling(252)

    mom_1m = float(close.iloc[-1] / close.iloc[-21]  - 1) if len(close) > 21  else 0.0
    mom_3m = float(close.iloc[-1] / close.iloc[-63]  - 1) if len(close) > 63  else 0.0
    mom_6m = float(close.iloc[-1] / close.iloc[-126] - 1) if len(close) > 126 else 0.0
    mom_1y = float(close.iloc[-1] / close.iloc[-252] - 1) if len(close) > 252 else 0.0

    ewma_vol     = float(log_ret.ewm(span=63).std().iloc[-1])
    rolling_max  = close.cummax()
    drawdown     = (close - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    atr = None
    if "High" in df.columns and "Low" in df.columns:
        atr = float((df["High"].squeeze() - df["Low"].squeeze()).rolling(14).mean().iloc[-1])

    return {
        "log_returns"   : log_ret,
        "close"         : close,
        "current_price" : float(close.iloc[-1]),
        "mu_daily"      : float(log_ret.mean()),
        "sigma_daily"   : ewma_vol,
        "sigma_annual"  : ewma_vol * (252 ** 0.5),
        "vol_21"        : float(roll21.std().iloc[-1]) if len(log_ret) > 21  else ewma_vol,
        "vol_63"        : float(roll63.std().iloc[-1]) if len(log_ret) > 63  else ewma_vol,
        "vol_252"       : float(roll252.std().iloc[-1]) if len(log_ret) > 252 else ewma_vol,
        "mom_1m"        : mom_1m,
        "mom_3m"        : mom_3m,
        "mom_6m"        : mom_6m,
        "mom_1y"        : mom_1y,
        "max_drawdown"  : max_drawdown,
        "atr"           : atr,
        "week52_high"   : float(close.iloc[-252:].max()) if len(close) >= 252 else float(close.max()),
        "week52_low"    : float(close.iloc[-252:].min()) if len(close) >= 252 else float(close.min()),
        "avg_volume_30" : float(df["Volume"].squeeze().iloc[-30:].mean()) if "Volume" in df.columns else None,
    }
