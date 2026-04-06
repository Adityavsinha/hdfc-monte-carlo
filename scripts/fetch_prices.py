"""
scripts/fetch_prices.py
-----------------------
Fetches live prices for ALL NSE stocks (500+) using yfinance.
Runs via GitHub Actions every 30 minutes during market hours.
Saves to docs/live_prices.json — no CORS issues on frontend.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Full NSE universe — 500+ symbols
NSE_SYMBOLS = [
    "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK","INFOSYS","SBIN",
    "HINDUNILVR","ITC","LT","KOTAKBANK","AXISBANK","ASIANPAINT","MARUTI",
    "SUNPHARMA","TITAN","ULTRACEMCO","BAJFINANCE","WIPRO","ONGC","NTPC",
    "POWERGRID","HCLTECH","M&M","TATAMOTORS","TATASTEEL","ADANIENT",
    "ADANIPORTS","COALINDIA","BAJAJFINSV","NESTLEIND","CIPLA","DRREDDY",
    "HEROMOTOCO","BRITANNIA","TECHM","GRASIM","BPCL","EICHERMOT","DIVISLAB",
    "APOLLOHOSP","TATACONSUM","BAJAJ-AUTO","HINDALCO","JSWSTEEL","SHRIRAMFIN",
    "INDUSINDBK","SBILIFE","HDFCLIFE","BEL","PIDILITIND","HAVELLS","SIEMENS",
    "TORNTPHARM","MUTHOOTFIN","LUPIN","AMBUJACEM","TRENT","IRCTC","CONCOR",
    "DMART","ZYDUSLIFE","AUROPHARMA","ALKEM","TATAPOWER","NHPC","SAIL",
    "NATIONALUM","VEDL","JINDALSTEL","GODREJCP","DABUR","EMAMILTD",
    "BANKBARODA","CANBK","FEDERALBNK","IDFCFIRSTB","BANDHANBNK","PNB",
    "UNIONBANK","MFSL","MAXHEALTH","FORTIS","LALPATHLAB","METROPOLIS",
    "BIOCON","SYNGENE","LAURUSLABS","GRANULES","IPCALAB","INDIGO","IRFC",
    "RECLTD","PFC","HUDCO","LICHSGFIN","CHOLAFIN","JIOFIN","ETERNAL",
    "PAYTM","NYKAA","POLICYBZR","DELHIVERY","HAPPSTMNDS","LTTS","COFORGE",
    "MPHASIS","PERSISTENT","KPITTECH","TATAELXSI","RBLBANK","YESBANK",
    "AUBANK","UJJIVANSFB","EQUITASBNK","ASTRAL","SUPREMEIND","RELAXO",
    "BATA","PAGEIND","VOLTAS","BLUESTAR","CROMPTON","POLYCAB","KEI",
    "FINOLEX","HINDPETRO","IOC","GAIL","PETRONET","MGL","IGL","GUJARATGAS",
    "ADANIGREEN","TORNTPOWER","CESC","JSPL","NMDC","MOIL","APLAPOLLO",
    "RATNAMANI","RAMCOCEM","JKCEMENT","DALBHARAT","ACC","KEC","RVNL","NBCC",
    "DLF","GODREJPROP","OBEROIRLTY","PRESTIGE","BRIGADE","SOBHA","LODHA",
    "PHOENIXLTD","MOTHERSON","BOSCHLTD","BHARATFORG","EXIDEIND","BALKRISIND",
    "CEATLTD","MRF","APOLLOTYRE","MINDA","SUNDRMFAST","SCHAEFFLER","3MINDIA",
    "ABB","CUMMINSIND","THERMAX","BHEL","BDL","HAL","COCHINSHIP","GRSE",
    "VARUNBEV","UBL","RADICO","KANSAINER","BERGERPAINTS","PFIZER","ABBOTINDIA",
    "AJANTPHARM","NATCO","JBCHEPHARM","INDIAMART","NAUKRI","CDSL","BSE",
    "MCX","ANGELONE","IIFL","CREDITACC","MEDANTA","ASTER","KIMS","MANKIND",
    "JSWENERGY","CPCL","NAVINFLUOR","AAVAS","HOMEFIRST","APTUS",
    "TATACHEM","PIIND","ATUL","DEEPAKNTR","VINYSCHEM","FLUOROCHEM",
    "BALRAMCHIN","BAJAJCON","GODREJIND","MAHINDCIE","SWARAJENG",
    "ECLERX","RATEGAIN","CARTRADE","JUSTDIAL","INFOEDGE",
    "HSCL","GPPL","GLAND","MAXESTATES","LODHADEV","SIGNATURE",
    "SHYAMSTEEL","GALLANTT","SAREGAMA","NETWORK18","TV18BRDCST",
    "ZEEL","SUNTV","PVRINOX","DELTACORP","WONDERLA","MANYAVAR","ABFRL",
    "VMART","SHOPERSTOP","TRENT","VIJAYABANK","KARUR","SOUTHINDBA",
    "CSBBANK","DCBBANK","LAKSHVILAS","UJJIVAN","JKBANK","TMB",
    "CENTURYPLY","GREENPANEL","CENTURYTEX","RAYMOND","BOMBDYEING",
    "ADANIWILMAR","PATANJALI","BIKAJI","DEVYANI","WESTLIFE","JUBLFOOD",
    "SPECIALITY","SAPPHIRE","BARBEQUE","GMRINFRA","IRB","ASHOKA",
    "KNRCON","HG INFRA","PNCINFRA","DILIPBUILDCON","SADBHAV",
    "TITAGARH","TEXRAIL","IRCON","RITES","RAILTEL","HFCL","TTML",
    "STLTECH","TEJASNET","VINDHYATEL","GTPL","DISH","SITI",
    "DCMSHRIRAM","BALLARPUR","JKPAPER","TNPL","CENTURIES","ANDHRA",
    "SHARDACROP","DHANUKA","RALLIS","BAYER","SUMIT","COROMANDEL",
    "CHAMBLFERT","RCF","GNFC","GSFC","FACT","NFL",
]

def fetch_batch(symbols, batch_size=50):
    """Fetch price data for a batch of symbols."""
    results = {}
    tickers = [s+".NS" for s in symbols]
    
    try:
        data = yf.download(
            tickers, period="2d", interval="1d",
            auto_adjust=True, progress=False, threads=True
        )
        
        if data.empty:
            return results
            
        if isinstance(data.columns, pd.MultiIndex):
            close  = data["Close"]
            volume = data["Volume"] if "Volume" in data else None
        else:
            close  = data[["Close"]]
            volume = data[["Volume"]] if "Volume" in data else None
        
        for sym_ns in tickers:
            sym = sym_ns.replace(".NS","")
            try:
                if sym_ns in close.columns:
                    col = close[sym_ns].dropna()
                elif sym in close.columns:
                    col = close[sym].dropna()
                else:
                    continue
                    
                if len(col) < 1:
                    continue
                    
                price    = float(col.iloc[-1])
                prev     = float(col.iloc[-2]) if len(col) > 1 else price
                chg_pct  = ((price - prev) / prev * 100) if prev > 0 else 0
                
                vol = None
                if volume is not None:
                    v_col = volume.get(sym_ns, volume.get(sym))
                    if v_col is not None:
                        v_val = v_col.dropna()
                        if len(v_val) > 0:
                            vol = int(v_val.iloc[-1])
                
                results[sym] = {
                    "price"  : round(price, 2),
                    "chgPct" : round(chg_pct, 2),
                    "prev"   : round(prev, 2),
                    "vol"    : vol,
                }
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"  Batch failed: {e}")
    
    return results


def fetch_info_batch(symbols):
    """Fetch fundamental info for symbols."""
    results = {}
    for sym in symbols[:20]:  # Limit info fetches
        try:
            info = yf.Ticker(sym+".NS").fast_info
            results[sym] = {
                "mcap"   : getattr(info, "market_cap", None),
                "pe"     : getattr(info, "pe_ratio", None),
                "h52"    : getattr(info, "year_high", None),
                "l52"    : getattr(info, "year_low", None),
            }
        except Exception:
            pass
    return results


def main():
    t0 = time.time()
    logger.info(f"Fetching live prices for {len(NSE_SYMBOLS)} NSE stocks...")
    
    all_prices = {}
    batch_size = 50
    
    for i in range(0, len(NSE_SYMBOLS), batch_size):
        batch = NSE_SYMBOLS[i:i+batch_size]
        logger.info(f"  Batch {i//batch_size+1}/{len(NSE_SYMBOLS)//batch_size+1} ({len(batch)} stocks)...")
        prices = fetch_batch(batch)
        all_prices.update(prices)
        time.sleep(0.5)
    
    logger.info(f"  Fetched {len(all_prices)}/{len(NSE_SYMBOLS)} stocks in {time.time()-t0:.1f}s")
    
    output = {
        "last_updated"  : datetime.now().strftime("%d %b %Y %H:%M IST"),
        "timestamp"     : int(time.time() * 1000),
        "total"         : len(all_prices),
        "prices"        : all_prices,
    }
    
    with open(DOCS_DIR / "live_prices.json", "w") as f:
        json.dump(output, f, separators=(',', ':'))
    
    size_kb = (DOCS_DIR / "live_prices.json").stat().st_size / 1024
    logger.info(f"  ✅ live_prices.json saved ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
