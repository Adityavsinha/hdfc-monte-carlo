"""
scripts/fetch_prices.py
-----------------------
Fetches EOD prices + fundamentals for Nifty 500 stocks via yfinance.
Runs daily at 6:30 PM IST via GitHub Actions (same job as pipeline).
Saves to docs/screener_data.json — read directly by frontend, zero CORS.

Data per stock:
  price, chgPct, chgAbs, open, high, low, volume
  week52High, week52Low, marketCap, pe, pb, eps
  roe, dividendYield, debtEquity, sector, industry, name
"""
import json, logging, sys, time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT  = SCRIPT_DIR.parent
DOCS_DIR   = REPO_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# ── Nifty 500 symbol list (NSE) ───────────────────────────────────────────────
NIFTY500 = [
    # Nifty 50
    "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK","INFOSYS","SBIN",
    "HINDUNILVR","ITC","LT","KOTAKBANK","AXISBANK","ASIANPAINT","MARUTI",
    "SUNPHARMA","TITAN","ULTRACEMCO","BAJFINANCE","WIPRO","ONGC","NTPC",
    "POWERGRID","HCLTECH","M&M","TATAMOTORS","TATASTEEL","ADANIENT",
    "ADANIPORTS","COALINDIA","BAJAJFINSV","NESTLEIND","CIPLA","DRREDDY",
    "HEROMOTOCO","BRITANNIA","TECHM","GRASIM","BPCL","EICHERMOT","DIVISLAB",
    "APOLLOHOSP","TATACONSUM","BAJAJ-AUTO","HINDALCO","JSWSTEEL","SHRIRAMFIN",
    "INDUSINDBK","SBILIFE","HDFCLIFE","BEL",
    # Nifty Next 50
    "PIDILITIND","HAVELLS","SIEMENS","TORNTPHARM","MUTHOOTFIN","LUPIN",
    "AMBUJACEM","TRENT","IRCTC","CONCOR","DMART","ZYDUSLIFE","AUROPHARMA",
    "ALKEM","TATAPOWER","NHPC","SAIL","NATIONALUM","VEDL","JINDALSTEL",
    "GODREJCP","DABUR","EMAMILTD","BANKBARODA","CANBK","FEDERALBNK",
    "IDFCFIRSTB","BANDHANBNK","PNB","UNIONBANK","MAXHEALTH","FORTIS",
    "LALPATHLAB","BIOCON","LAURUSLABS","INDIGO","IRFC","RECLTD","PFC",
    "CHOLAFIN","JIOFIN","ETERNAL","PAYTM","NYKAA","LTTS","COFORGE",
    "MPHASIS","PERSISTENT","KPITTECH","TATAELXSI",
    # Midcap / Others
    "RBLBANK","YESBANK","AUBANK","UJJIVANSFB","EQUITASBNK","ASTRAL",
    "SUPREMEIND","RELAXO","BATA","PAGEIND","VOLTAS","BLUESTAR","CROMPTON",
    "POLYCAB","KEI","HINDPETRO","IOC","GAIL","PETRONET","IGL","MGL",
    "GUJARATGAS","ADANIGREEN","TORNTPOWER","CESC","JSPL","NMDC","MOIL",
    "APLAPOLLO","RATNAMANI","RAMCOCEM","JKCEMENT","DALBHARAT","ACC",
    "KEC","RVNL","NBCC","DLF","GODREJPROP","OBEROIRLTY","PRESTIGE",
    "BRIGADE","LODHA","PHOENIXLTD","MOTHERSON","BOSCHLTD","BHARATFORG",
    "EXIDEIND","BALKRISIND","CEATLTD","MRF","APOLLOTYRE","MINDA",
    "SUNDRMFAST","SCHAEFFLER","ABB","CUMMINSIND","THERMAX","BHEL",
    "BDL","HAL","COCHINSHIP","GRSE","VARUNBEV","UBL","RADICO",
    "BERGERPAINTS","KANSAINER","PFIZER","ABBOTINDIA","AJANTPHARM",
    "NATCO","JBCHEPHARM","MANKIND","CDSL","BSE","MCX","ANGELONE",
    "IIFL","INDIAMART","NAUKRI","JSWENERGY","NAVINFLUOR","TATACHEM",
    "PIIND","DEEPAKNTR","MFSL","PVRINOX","DELTACORP","CARTRADE",
    "JUSTDIAL","DELHIVERY","HAPPSTMNDS","METROPOLIS","FORTIS",
    "ASTER","KIMS","RAINBOW","MEDANTA","SYNGENE","GRANULES","IPCALAB",
    "SOLARA","LAURUS","OPTOCIRCUI","AJANTPHARM","TORNTPHARM",
    "CHOLAFIN","SUNDARMFIN","M&MFIN","BAJAJHLDNG","LICHSGFIN","HUDCO",
    "RECLTD","IRFC","RVNL","RAILTEL","NBCC","IRCON","RITES",
    "TTML","HFCL","STLTECH","TEJASNET","VINDHYATEL","GTPL",
    "TATACHEM","DEEPAKNTR","ATUL","PIIND","NAVINFLUOR","ALKYLAMINE",
    "FINEORG","GALAXYSURF","MAPMYINDIA","RATEGAIN","POLICYBZR",
    "PAYTM","NYKAA","ZOMATO","ETERNAL","DELHIVERY",
    "SOLARINDS","MAXESTATES","SIGNATURE","LODHADEV",
    "OBEROIRLTY","PRESTIGE","BRIGADE","SOBHA","PHOENIXLTD",
    "NESCO","ESAB","GRINDWELL","TIINDIA","CRAFTSMAN",
    "JYOTICNC","ELGIEQUIP","KIRLOSENG","KALPATPOWR","KEC",
    "KNRCON","ASHOKA","IRB","GMRINFRA","DILIPBUILDCON",
    "TITAGARH","TEXRAIL","IRCON","HGINFRA","PNCINFRA",
    "JINDALSAW","RATNAMANI","WELCORP","APLAPOLLO","APL",
    "JKLAKSHMI","HEIDELBERG","NCLTEX","KESORAMIND",
    "CENTURYPLY","GREENPANEL","ARCHIDPLY","KITEX","DCMSHRIRAM",
    "BALRAMCHIN","DHAMPUR","TRIVENI","BAJAJCON",
    "HATSUN","PARAGMILK","HERITAGE","DODLA","PRABHAGAS",
    "SUPPETRO","GUJGAS","IGL","MGL","MAHANGAS",
    "SAREGAMA","NETWORK18","TV18BRDCST","ZEEL","SUNTV",
    "PVRINOX","DELTACORP","WONDERLA","MANYAVAR","ABFRL",
    "SHOPERSTOP","TRENT","VMART","DMART",
]
# Deduplicate while preserving order
seen = set()
NIFTY500 = [x for x in NIFTY500 if not (x in seen or seen.add(x))]

SECTOR_MAP = {
    "RELIANCE":"Energy","TCS":"IT","HDFCBANK":"Banking","BHARTIARTL":"Telecom",
    "ICICIBANK":"Banking","INFOSYS":"IT","SBIN":"Banking","HINDUNILVR":"FMCG",
    "ITC":"FMCG","LT":"Infra","KOTAKBANK":"Banking","AXISBANK":"Banking",
    "ASIANPAINT":"Consumer","MARUTI":"Auto","SUNPHARMA":"Pharma","TITAN":"Consumer",
    "ULTRACEMCO":"Cement","BAJFINANCE":"Finance","WIPRO":"IT","ONGC":"Energy",
    "NTPC":"Power","POWERGRID":"Power","HCLTECH":"IT","M&M":"Auto",
    "TATAMOTORS":"Auto","TATASTEEL":"Metal","ADANIENT":"Energy","ADANIPORTS":"Infra",
    "COALINDIA":"Mining","BAJAJFINSV":"Finance","NESTLEIND":"FMCG","CIPLA":"Pharma",
    "DRREDDY":"Pharma","HEROMOTOCO":"Auto","BRITANNIA":"FMCG","TECHM":"IT",
    "GRASIM":"Cement","BPCL":"Energy","EICHERMOT":"Auto","DIVISLAB":"Pharma",
    "APOLLOHOSP":"Healthcare","TATACONSUM":"FMCG","BAJAJ-AUTO":"Auto",
    "HINDALCO":"Metal","JSWSTEEL":"Metal","SHRIRAMFIN":"Finance",
    "INDUSINDBK":"Banking","SBILIFE":"Insurance","HDFCLIFE":"Insurance",
    "BEL":"Defence","HAL":"Defence","BDL":"Defence","COCHINSHIP":"Defence",
    "GRSE":"Defence","IRFC":"Finance","RECLTD":"Finance","PFC":"Finance",
    "RVNL":"Infra","NBCC":"Infra","IRCON":"Infra","KEC":"Infra",
    "TATAPOWER":"Power","NHPC":"Power","TORNTPOWER":"Power","CESC":"Power",
    "JSWENERGY":"Power","ADANIGREEN":"Energy","GAIL":"Energy","IOC":"Energy",
    "HINDPETRO":"Energy","BPCL":"Energy","PETRONET":"Energy","IGL":"Energy",
    "MGL":"Energy","GUJARATGAS":"Energy","NMDC":"Mining","MOIL":"Mining",
    "SAIL":"Metal","NATIONALUM":"Metal","VEDL":"Metal","JINDALSTEL":"Metal",
    "JSPL":"Metal","APLAPOLLO":"Metal","RATNAMANI":"Metal","WELCORP":"Metal",
    "DLF":"Real Estate","GODREJPROP":"Real Estate","PRESTIGE":"Real Estate",
    "OBEROIRLTY":"Real Estate","LODHA":"Real Estate","BRIGADE":"Real Estate",
    "PHOENIXLTD":"Real Estate","SOBHA":"Real Estate",
    "ETERNAL":"Tech","PAYTM":"Fintech","NYKAA":"Retail","ZOMATO":"Tech",
    "DELHIVERY":"Logistics","INDIAMART":"Tech","NAUKRI":"Tech",
    "CDSL":"Finance","BSE":"Finance","MCX":"Finance","ANGELONE":"Finance",
    "TATACHEM":"Chemicals","PIIND":"Chemicals","DEEPAKNTR":"Chemicals",
    "NAVINFLUOR":"Chemicals","ATUL":"Chemicals",
}


def safe_float(v, default=None):
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except:
        return default


def fetch_batch_prices(symbols: list[str]) -> dict:
    """Fetch OHLCV for a batch using yfinance download (fast, bulk)."""
    tickers = [s + ".NS" for s in symbols]
    result  = {}
    try:
        df = yf.download(tickers, period="5d", interval="1d",
                         auto_adjust=True, progress=False, threads=True)
        if df.empty:
            return result

        if isinstance(df.columns, pd.MultiIndex):
            close  = df["Close"]
            open_  = df["Open"]  if "Open"   in df else None
            high   = df["High"]  if "High"   in df else None
            low    = df["Low"]   if "Low"    in df else None
            volume = df["Volume"] if "Volume" in df else None
        else:
            # Single ticker — wrap
            close  = df[["Close"]]
            open_  = df[["Open"]]  if "Open"   in df.columns else None
            high   = df[["High"]]  if "High"   in df.columns else None
            low    = df["Low"]   if "Low"    in df.columns else None
            volume = df[["Volume"]] if "Volume" in df.columns else None

        for sym_ns in tickers:
            sym = sym_ns.replace(".NS", "")
            try:
                col = close.get(sym_ns, close.get(sym))
                if col is None:
                    continue
                col = col.dropna()
                if len(col) < 2:
                    continue

                price = safe_float(col.iloc[-1])
                prev  = safe_float(col.iloc[-2])
                if price is None or prev is None:
                    continue

                chg_abs = round(price - prev, 2)
                chg_pct = round((price - prev) / prev * 100, 2) if prev else 0

                def g(df_, col_name):
                    if df_ is None: return None
                    c = df_.get(col_name, df_.get(col_name.replace(".NS","")))
                    if c is None: return None
                    v = c.dropna()
                    return safe_float(v.iloc[-1]) if len(v) else None

                result[sym] = {
                    "price"   : price,
                    "chgPct"  : chg_pct,
                    "chgAbs"  : chg_abs,
                    "prev"    : prev,
                    "open"    : g(open_,  sym_ns),
                    "high"    : g(high,   sym_ns),
                    "low"     : g(low,    sym_ns),
                    "volume"  : int(g(volume, sym_ns) or 0) or None,
                }
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"  Batch download failed: {e}")
    return result


def fetch_fundamentals(sym: str) -> dict:
    """Fetch fundamentals for a single stock via yfinance Ticker.info."""
    try:
        t    = yf.Ticker(sym + ".NS")
        info = t.info or {}

        def g(k): return safe_float(info.get(k))

        return {
            "name"         : info.get("longName") or info.get("shortName") or sym,
            "sector"       : info.get("sector") or SECTOR_MAP.get(sym, "Other"),
            "industry"     : info.get("industry", ""),
            "marketCap"    : g("marketCap"),
            "pe"           : g("trailingPE"),
            "pb"           : g("priceToBook"),
            "eps"          : g("trailingEps"),
            "roe"          : g("returnOnEquity"),
            "roa"          : g("returnOnAssets"),
            "debtEquity"   : g("debtToEquity"),
            "currentRatio" : g("currentRatio"),
            "dividendYield": g("dividendYield"),
            "week52High"   : g("fiftyTwoWeekHigh"),
            "week52Low"    : g("fiftyTwoWeekLow"),
            "bookValue"    : g("bookValue"),
            "beta"         : g("beta"),
            "employees"    : info.get("fullTimeEmployees"),
            "website"      : info.get("website", ""),
            "description"  : (info.get("longBusinessSummary") or "")[:400],
        }
    except Exception:
        return {
            "name"  : sym,
            "sector": SECTOR_MAP.get(sym, "Other"),
        }


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info(f"QuantEdge — Fetching Nifty 500 EOD data ({len(NIFTY500)} stocks)")
    logger.info(f"Started: {datetime.now().strftime('%d %b %Y %H:%M IST')}")
    logger.info("=" * 60)

    # ── Step 1: Bulk price download in batches of 50 ─────────────────
    logger.info("\n[1/3] Downloading EOD prices (bulk)...")
    all_prices = {}
    BATCH = 50
    for i in range(0, len(NIFTY500), BATCH):
        batch = NIFTY500[i:i+BATCH]
        logger.info(f"  Batch {i//BATCH+1}/{(len(NIFTY500)+BATCH-1)//BATCH} ({len(batch)} stocks)...")
        prices = fetch_batch_prices(batch)
        all_prices.update(prices)
        time.sleep(0.3)

    logger.info(f"  ✅ Prices fetched: {len(all_prices)}/{len(NIFTY500)}")

    # ── Step 2: Fundamentals (per-stock, with rate limiting) ──────────
    logger.info("\n[2/3] Fetching fundamentals...")
    all_funds = {}
    for i, sym in enumerate(NIFTY500):
        try:
            fund = fetch_fundamentals(sym)
            all_funds[sym] = fund
            if i % 25 == 0:
                logger.info(f"  Fundamentals: {i+1}/{len(NIFTY500)}")
        except Exception as e:
            all_funds[sym] = {"name": sym, "sector": SECTOR_MAP.get(sym, "Other")}
        time.sleep(0.15)   # gentle rate limit

    logger.info(f"  ✅ Fundamentals fetched: {len(all_funds)}")

    # ── Step 3: Merge and save ────────────────────────────────────────
    logger.info("\n[3/3] Merging and saving...")
    stocks = {}
    for sym in NIFTY500:
        px   = all_prices.get(sym, {})
        fund = all_funds.get(sym, {})
        if not px.get("price"):
            continue
        stocks[sym] = {
            # Price data
            "symbol"  : sym,
            "name"    : fund.get("name", sym),
            "sector"  : fund.get("sector", SECTOR_MAP.get(sym, "Other")),
            "industry": fund.get("industry", ""),
            "price"   : px["price"],
            "chgPct"  : px.get("chgPct", 0),
            "chgAbs"  : px.get("chgAbs", 0),
            "open"    : px.get("open"),
            "high"    : px.get("high"),
            "low"     : px.get("low"),
            "volume"  : px.get("volume"),
            # 52W from fundamentals (more accurate)
            "week52High": fund.get("week52High") or px.get("h52"),
            "week52Low" : fund.get("week52Low")  or px.get("l52"),
            # Fundamentals
            "marketCap"   : fund.get("marketCap"),
            "pe"          : fund.get("pe"),
            "pb"          : fund.get("pb"),
            "eps"         : fund.get("eps"),
            "roe"         : fund.get("roe"),
            "roa"         : fund.get("roa"),
            "debtEquity"  : fund.get("debtEquity"),
            "currentRatio": fund.get("currentRatio"),
            "dividendYield": fund.get("dividendYield"),
            "beta"        : fund.get("beta"),
            "bookValue"   : fund.get("bookValue"),
            "employees"   : fund.get("employees"),
            "website"     : fund.get("website", ""),
            "description" : fund.get("description", ""),
        }

    output = {
        "last_updated": datetime.now().strftime("%d %b %Y %H:%M IST"),
        "timestamp"   : int(time.time() * 1000),
        "total"       : len(stocks),
        "stocks"      : stocks,
    }

    out_path = DOCS_DIR / "screener_data.json"
    with open(out_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_kb = out_path.stat().st_size / 1024
    elapsed = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"  ✅ screener_data.json saved: {size_kb:.0f} KB")
    logger.info(f"  Total stocks: {len(stocks)}")
    logger.info(f"  Runtime: {elapsed/60:.1f} min")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
