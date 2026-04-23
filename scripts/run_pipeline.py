"""
scripts/run_pipeline.py — QuantEdge Analytics Daily Pipeline
Fetches Nifty 50 data first, computes stock betas vs Nifty,
runs CAPM+GARCH+MC on top 75 stocks, saves JSON to docs/.
"""
import json, logging, sys, time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT  = SCRIPT_DIR.parent
DOCS_DIR   = REPO_ROOT / "docs"
LOGS_DIR   = REPO_ROOT / "logs"

DOCS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"{LOGS_DIR}/pipeline_{datetime.now().strftime('%Y%m%d')}.log"),
    ]
)
logger = logging.getLogger(__name__)

from config import TOP_N_STOCKS, NIFTY50_SYMBOLS, SECTOR_MAP
from data_ingestion import (fetch_live_nifty50, fetch_stock_data,
                             fetch_financial_info, compute_features)
from quant_engine import run_full_pipeline, compute_technical_indicators, compute_fundamental_score


def fetch_nifty50_returns():
    """Fetch Nifty 50 index (^NSEI) returns for beta computation."""
    logger.info("  Fetching Nifty 50 index data (^NSEI)...")
    import yfinance as yf
    from datetime import timedelta
    end   = datetime.today()
    start = end - timedelta(days=365 * 3 + 60)
    try:
        df = yf.download("^NSEI", start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            logger.warning("  Nifty index data empty — beta will default to 1.0")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = pd.to_numeric(df["Close"].squeeze(), errors="coerce").dropna()
        returns = np.log(close / close.shift(1)).dropna()
        logger.info(f"  Nifty 50 returns: {len(returns)} days")
        return returns
    except Exception as e:
        logger.warning(f"  Nifty fetch failed: {e}")
        return None


def score_stock(features):
    s  = features.get("mom_3m", 0) * 2.0
    s += features.get("mom_1m", 0) * 1.0
    s += features.get("mom_1y", 0) * 0.5
    v  = features.get("sigma_annual", 0.3)
    if v > 0.6: s -= 0.15
    return s


def main():
    t0 = time.time()
    logger.info("="*65)
    logger.info("QUANTEDGE ANALYTICS — DAILY QUANT PIPELINE")
    logger.info(f"Started: {datetime.now().strftime('%d %b %Y %H:%M IST')}")
    logger.info("="*65)

    # 1. Nifty 50 composition
    logger.info("\n[1/6] Fetching live Nifty 50 composition...")
    nifty50 = fetch_live_nifty50()
    logger.info(f"  → {len(nifty50)} stocks")
    with open(DOCS_DIR / "nifty50_composition.json","w") as f:
        json.dump({"last_updated": datetime.now().strftime("%d %b %Y %H:%M IST"),
                   "count": len(nifty50), "stocks": nifty50}, f)

    # 2. Nifty 50 index returns for beta
    logger.info("\n[2/6] Fetching Nifty 50 index for beta computation...")
    nifty_returns = fetch_nifty50_returns()

    # 3. Quick feature scan
    logger.info(f"\n[3/6] Quick feature scan for stock selection...")
    all_symbols = [s["symbol"] for s in nifty50]
    quick_feat  = {}
    for sym in all_symbols:
        try:
            df = fetch_stock_data(sym, years=1)
            if df is not None and len(df) > 60:
                quick_feat[sym] = compute_features(df)
        except Exception as e:
            logger.warning(f"  {sym}: quick scan failed — {e}")

    # 4. Select top stocks
    logger.info(f"\n[4/6] Selecting top {TOP_N_STOCKS} stocks by momentum score...")
    scored   = sorted([(s, score_stock(f)) for s,f in quick_feat.items()],
                      key=lambda x: x[1], reverse=True)
    selected = [s for s,_ in scored[:TOP_N_STOCKS]]
    logger.info(f"  → {len(selected)} stocks selected")

    with open(DOCS_DIR / "selected_stocks.json","w") as f:
        json.dump({"last_updated": datetime.now().strftime("%d %b %Y %H:%M IST"),
                   "selected": selected,
                   "scores": {s: round(sc,4) for s,sc in scored[:TOP_N_STOCKS]}}, f)

    # 5. Full quant pipeline
    logger.info(f"\n[5/6] Running full quant pipeline on {len(selected)} stocks...")
    results = []
    failed  = []

    for i, sym in enumerate(selected, 1):
        logger.info(f"\n[{i:02d}/{len(selected)}] {sym}")
        df = fetch_stock_data(sym, years=10)
        if df is None:
            failed.append(sym); continue

        feat     = compute_features(df)
        fin_info = fetch_financial_info(sym)
        meta     = next((s for s in nifty50 if s["symbol"] == sym), {})
        fin_info["name"]   = meta.get("name", sym)
        fin_info["sector"] = fin_info.get("sector") or meta.get("sector", SECTOR_MAP.get(sym,"Other"))

        # Technical analysis on full OHLCV data
        logger.info(f"    Computing technical indicators...")
        tech = compute_technical_indicators(df)

        result = run_full_pipeline(sym, feat, fin_info, nifty_returns)
        if result:
            result.update(tech)
            # Fundamental score
            fund_score = compute_fundamental_score(fin_info, result)
            result.update(fund_score)
            results.append(result)
        else:
            failed.append(sym)

    # 6. Save outputs
    logger.info(f"\n[6/6] Saving outputs...")
    results.sort(key=lambda x: (x.get("score",0), x.get("expected_return",0)), reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i

    # Separate chart data
    charts = {r["symbol"]: {"path_charts": r.pop("path_charts",{}),
                             "histogram"  : r.pop("histogram",{})} for r in results}

    output = {
        "last_updated"  : datetime.now().strftime("%d %b %Y %H:%M IST"),
        "total_stocks"  : len(results),
        "failed_stocks" : failed,
        "runtime_mins"  : round((time.time()-t0)/60, 1),
        "stocks"        : results,
    }

    with open(DOCS_DIR / "metrics.json","w") as f:
        json.dump(output, f, separators=(',',':'))
    with open(DOCS_DIR / "charts.json","w") as f:
        json.dump(charts, f, separators=(',',':'))

    summary = {
        "last_updated": output["last_updated"],
        "total_stocks": len(results),
        "summary": [{
            "symbol": r["symbol"], "name": r["name"], "sector": r["sector"],
            "price": r["price"], "expected_return_pct": r["expected_return_pct"],
            "prob_up": r["prob_up"], "signal": r["signal"],
            "signal_color": r["signal_color"], "confidence": r["confidence"],
            "sharpe": r["sharpe"], "var_95": r["var_95"],
            "sigma_annual": r["sigma_annual"], "mispricing_pct": r["mispricing_pct"],
            "mom_1m": r["mom_1m"], "mom_3m": r["mom_3m"],
            "week52_high": r["week52_high"], "week52_low": r["week52_low"],
            "market_cap": r.get("market_cap"), "pe_ratio": r.get("pe_ratio"),
            "beta_nifty": r.get("beta_nifty"), "rank": r["rank"],
            "mu_annual": r.get("mu_annual"), "score": r.get("score",0),
            # Technical indicators for screener display
            "rsi_14": r.get("rsi_14"), "rsi_signal": r.get("rsi_signal"),
            "macd_cross": r.get("macd_cross"), "tech_signal": r.get("tech_signal"),
            "tech_score": r.get("tech_score"),
            "above_sma50": r.get("above_sma50"), "above_sma200": r.get("above_sma200"),
            "golden_cross": r.get("golden_cross"),
            "bb_position": r.get("bb_position"), "bb_signal": r.get("bb_signal"),
            "stoch_k": r.get("stoch_k"), "vol_ratio": r.get("vol_ratio"),
            "sma_50": r.get("sma_50"), "sma_200": r.get("sma_200"),
            "fundamental_grade": r.get("fundamental_grade"),
            "fundamental_score": r.get("fundamental_score"),
            # Fields needed by modal quick metrics strip
            "mean_price": r.get("mean_price"), "median_price": r.get("median_price"),
            "ci_5": r.get("ci_5"), "ci_25": r.get("ci_25"),
            "ci_75": r.get("ci_75"), "ci_95": r.get("ci_95"),
            "n_simulations": r.get("n_simulations", 10000),
            "horizon_days": r.get("horizon_days", 252),
            "prob_10up": r.get("prob_10up"), "prob_20up": r.get("prob_20up"),
            "prob_10down": r.get("prob_10down"),
            "bull_median": r.get("bull_median"), "base_median": r.get("base_median"),
            "bear_median": r.get("bear_median"),
            "max_drawdown": r.get("max_drawdown"), "sortino": r.get("sortino"),
            "calmar": r.get("calmar"), "cvar_95": r.get("cvar_95"),
            "var_99": r.get("var_99"), "drift_method": r.get("drift_method",""),
        } for r in results]
    }

    with open(DOCS_DIR / "summary.json","w") as f:
        json.dump(summary, f, separators=(',',':'))

    avg_ret  = np.mean([r["expected_return_pct"] for r in results]) if results else 0
    buy_cnt  = sum(1 for r in results if "BUY" in r.get("signal",""))
    logger.info("\n" + "="*65)
    logger.info(f"  Processed     : {len(results)} / {len(selected)}")
    logger.info(f"  Failed        : {len(failed)} ({', '.join(failed[:5])})")
    logger.info(f"  BUY signals   : {buy_cnt}")
    logger.info(f"  Avg exp. ret  : {avg_ret:+.1f}%")
    logger.info(f"  Runtime       : {(time.time()-t0)/60:.1f} min")
    logger.info(f"  summary.json  : {(DOCS_DIR/'summary.json').stat().st_size/1024:.1f} KB")
    logger.info("="*65)


if __name__ == "__main__":
    main()
