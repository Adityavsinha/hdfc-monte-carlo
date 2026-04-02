"""
run_nifty50.py
--------------
Runs Monte Carlo simulation for all Nifty 50 stocks.
Saves results to docs/nifty50_data.json for the website to read.
Runs automatically via GitHub Actions every weekday.
"""

import json
import logging
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/nifty50_{datetime.now().strftime('%Y%m%d')}.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── All Nifty 50 stocks ──────────────────────
NIFTY50_STOCKS = [
    {"symbol": "RELIANCE.NS",   "name": "Reliance Industries",   "sector": "Energy"},
    {"symbol": "TCS.NS",        "name": "Tata Consultancy Svcs", "sector": "IT"},
    {"symbol": "HDFCBANK.NS",   "name": "HDFC Bank",             "sector": "Banking"},
    {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel",         "sector": "Telecom"},
    {"symbol": "ICICIBANK.NS",  "name": "ICICI Bank",            "sector": "Banking"},
    {"symbol": "INFOSYS.NS",    "name": "Infosys",               "sector": "IT"},
    {"symbol": "SBIN.NS",       "name": "State Bank of India",   "sector": "Banking"},
    {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever",    "sector": "FMCG"},
    {"symbol": "ITC.NS",        "name": "ITC",                   "sector": "FMCG"},
    {"symbol": "LT.NS",         "name": "Larsen & Toubro",       "sector": "Infra"},
    {"symbol": "KOTAKBANK.NS",  "name": "Kotak Mahindra Bank",   "sector": "Banking"},
    {"symbol": "AXISBANK.NS",   "name": "Axis Bank",             "sector": "Banking"},
    {"symbol": "ASIANPAINT.NS", "name": "Asian Paints",          "sector": "Consumer"},
    {"symbol": "MARUTI.NS",     "name": "Maruti Suzuki",         "sector": "Auto"},
    {"symbol": "SUNPHARMA.NS",  "name": "Sun Pharmaceutical",    "sector": "Pharma"},
    {"symbol": "TITAN.NS",      "name": "Titan Company",         "sector": "Consumer"},
    {"symbol": "ULTRACEMCO.NS", "name": "UltraTech Cement",      "sector": "Cement"},
    {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance",         "sector": "Finance"},
    {"symbol": "WIPRO.NS",      "name": "Wipro",                 "sector": "IT"},
    {"symbol": "ONGC.NS",       "name": "ONGC",                  "sector": "Energy"},
    {"symbol": "NTPC.NS",       "name": "NTPC",                  "sector": "Power"},
    {"symbol": "POWERGRID.NS",  "name": "Power Grid Corp",       "sector": "Power"},
    {"symbol": "HCLTECH.NS",    "name": "HCL Technologies",      "sector": "IT"},
    {"symbol": "M&M.NS",        "name": "Mahindra & Mahindra",   "sector": "Auto"},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors",           "sector": "Auto"},
    {"symbol": "TATASTEEL.NS",  "name": "Tata Steel",            "sector": "Metal"},
    {"symbol": "ADANIENT.NS",   "name": "Adani Enterprises",     "sector": "Conglomerate"},
    {"symbol": "ADANIPORTS.NS", "name": "Adani Ports",           "sector": "Infra"},
    {"symbol": "COALINDIA.NS",  "name": "Coal India",            "sector": "Mining"},
    {"symbol": "BAJAJFINSV.NS", "name": "Bajaj Finserv",         "sector": "Finance"},
    {"symbol": "NESTLEIND.NS",  "name": "Nestle India",          "sector": "FMCG"},
    {"symbol": "CIPLA.NS",      "name": "Cipla",                 "sector": "Pharma"},
    {"symbol": "DRREDDY.NS",    "name": "Dr Reddy's Labs",       "sector": "Pharma"},
    {"symbol": "HEROMOTOCO.NS", "name": "Hero MotoCorp",         "sector": "Auto"},
    {"symbol": "BRITANNIA.NS",  "name": "Britannia Industries",  "sector": "FMCG"},
    {"symbol": "TECHM.NS",      "name": "Tech Mahindra",         "sector": "IT"},
    {"symbol": "GRASIM.NS",     "name": "Grasim Industries",     "sector": "Cement"},
    {"symbol": "BPCL.NS",       "name": "BPCL",                  "sector": "Energy"},
    {"symbol": "EICHERMOT.NS",  "name": "Eicher Motors",         "sector": "Auto"},
    {"symbol": "DIVISLAB.NS",   "name": "Divi's Laboratories",   "sector": "Pharma"},
    {"symbol": "APOLLOHOSP.NS", "name": "Apollo Hospitals",      "sector": "Healthcare"},
    {"symbol": "TATACONSUM.NS", "name": "Tata Consumer Products","sector": "FMCG"},
    {"symbol": "BAJAJ-AUTO.NS", "name": "Bajaj Auto",            "sector": "Auto"},
    {"symbol": "HINDALCO.NS",   "name": "Hindalco Industries",   "sector": "Metal"},
    {"symbol": "JSWSTEEL.NS",   "name": "JSW Steel",             "sector": "Metal"},
    {"symbol": "SHRIRAMFIN.NS", "name": "Shriram Finance",       "sector": "Finance"},
    {"symbol": "INDUSINDBK.NS", "name": "IndusInd Bank",         "sector": "Banking"},
    {"symbol": "SBILIFE.NS",    "name": "SBI Life Insurance",    "sector": "Insurance"},
    {"symbol": "HDFCLIFE.NS",   "name": "HDFC Life Insurance",   "sector": "Insurance"},
    {"symbol": "BEL.NS",        "name": "Bharat Electronics",    "sector": "Defence"},
]


def fetch_and_simulate(stock: dict, n_sim: int = 5000, horizon: int = 252) -> dict | None:
    """Fetch data and run Monte Carlo for one stock. Returns result dict or None on failure."""
    import yfinance as yf
    from datetime import timedelta

    symbol = stock["symbol"]
    try:
        end   = datetime.today()
        start = end - timedelta(days=365 * 10 + 60)
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)

        if df.empty or len(df) < 100:
            logger.warning(f"  {symbol}: insufficient data ({len(df)} rows)")
            return None

        # Flatten multi-level columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors="coerce").dropna()

        if len(close) < 100:
            return None

        current_price = float(close.iloc[-1])
        log_returns   = np.log(close / close.shift(1)).dropna()

        # Parameters
        rolling_vol  = log_returns.ewm(span=63).std().iloc[-1]
        mu_daily     = log_returns.mean()
        sigma_daily  = float(rolling_vol)

        # GBM simulation with Student-t
        rng = np.random.default_rng(42)
        from scipy.stats import t as student_t
        Z     = student_t.rvs(df=5, size=(horizon, n_sim), random_state=42)
        Z     = Z * np.sqrt(3 / 5)
        drift = (mu_daily - 0.5 * sigma_daily**2)
        paths = current_price * np.exp(
            np.cumsum(drift + sigma_daily * Z, axis=0)
        )

        final = paths[-1, :]

        # Scenarios
        def scenario_median(mu_mult, sig_mult):
            Z2   = student_t.rvs(df=5, size=(horizon, 500), random_state=99)
            Z2   = Z2 * np.sqrt(3 / 5)
            p    = current_price * np.exp(
                np.cumsum((mu_daily * mu_mult - 0.5 * (sigma_daily * sig_mult)**2)
                          + sigma_daily * sig_mult * Z2, axis=0)
            )
            return float(np.median(p[-1, :]))

        bull_med = scenario_median(1.30, 0.80)
        base_med = scenario_median(1.00, 1.00)
        bear_med = scenario_median(0.70, 1.40)

        # Percentile paths (sampled every 5 days for size)
        p5_path  = np.percentile(paths, 5,  axis=1)[::5].tolist()
        p50_path = np.percentile(paths, 50, axis=1)[::5].tolist()
        p95_path = np.percentile(paths, 95, axis=1)[::5].tolist()

        # Histogram
        counts, edges = np.histogram(final, bins=40)
        hist = {
            "labels": [round((edges[i]+edges[i+1])/2, 2) for i in range(len(counts))],
            "counts": counts.tolist()
        }

        pnl    = (final - current_price) / current_price
        var_95 = float(-np.percentile(pnl, 5)  * current_price)
        var_99 = float(-np.percentile(pnl, 1)  * current_price)
        cvar   = float(-pnl[pnl <= np.percentile(pnl, 5)].mean() * current_price)

        # 52-week high/low
        year_data = close.iloc[-252:] if len(close) >= 252 else close
        week52_high = float(year_data.max())
        week52_low  = float(year_data.min())

        return {
            "symbol"       : symbol.replace(".NS", ""),
            "name"         : stock["name"],
            "sector"       : stock["sector"],
            "price"        : round(current_price, 2),
            "week52_high"  : round(week52_high, 2),
            "week52_low"   : round(week52_low, 2),
            "mean_price"   : round(float(np.mean(final)), 2),
            "median_price" : round(float(np.median(final)), 2),
            "ci_5"         : round(float(np.percentile(final, 5)), 2),
            "ci_25"        : round(float(np.percentile(final, 25)), 2),
            "ci_75"        : round(float(np.percentile(final, 75)), 2),
            "ci_95"        : round(float(np.percentile(final, 95)), 2),
            "var_95"       : round(var_95, 2),
            "var_99"       : round(var_99, 2),
            "cvar_95"      : round(cvar, 2),
            "prob_up"      : round(float(np.mean(final > current_price)), 4),
            "prob_10up"    : round(float(np.mean(final > current_price * 1.10)), 4),
            "prob_10down"  : round(float(np.mean(final < current_price * 0.90)), 4),
            "mu_annual"    : round(float(mu_daily * 252), 4),
            "sigma_annual" : round(float(sigma_daily * np.sqrt(252)), 4),
            "expected_return_pct": round((float(np.mean(final)) / current_price - 1) * 100, 2),
            "bull_median"  : round(bull_med, 2),
            "base_median"  : round(base_med, 2),
            "bear_median"  : round(bear_med, 2),
            "p5_path"      : [round(v, 2) for v in p5_path],
            "p50_path"     : [round(v, 2) for v in p50_path],
            "p95_path"     : [round(v, 2) for v in p95_path],
            "histogram"    : hist,
            "horizon_days" : horizon,
            "n_simulations": n_sim,
        }

    except Exception as e:
        logger.error(f"  {symbol}: failed — {e}")
        return None


def main():
    logger.info("=" * 60)
    logger.info("NIFTY 50 — MONTE CARLO BATCH SIMULATION")
    logger.info(f"Started: {datetime.now().strftime('%d %b %Y %H:%M IST')}")
    logger.info("=" * 60)

    Path("docs").mkdir(exist_ok=True)
    results = []
    failed  = []

    for i, stock in enumerate(NIFTY50_STOCKS, 1):
        logger.info(f"[{i:02d}/50] {stock['symbol']} — {stock['name']}")
        result = fetch_and_simulate(stock)
        if result:
            results.append(result)
            logger.info(f"         ₹{result['price']:,.2f} → E[₹{result['mean_price']:,.2f}]  "
                        f"P(↑)={result['prob_up']:.1%}  σ={result['sigma_annual']:.1%}")
        else:
            failed.append(stock["symbol"])

    # Sort by expected return descending
    results.sort(key=lambda x: x["expected_return_pct"], reverse=True)

    # Add rank
    for i, r in enumerate(results, 1):
        r["rank"] = i

    output = {
        "last_updated" : datetime.now().strftime("%d %b %Y %H:%M IST"),
        "total_stocks" : len(results),
        "failed"       : failed,
        "stocks"       : results,
    }

    out_path = Path("docs/nifty50_data.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n✅ Done! {len(results)}/50 stocks simulated.")
    if failed:
        logger.warning(f"   Failed: {', '.join(failed)}")
    logger.info(f"   Data saved → {out_path}")
    logger.info(f"   File size : {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
