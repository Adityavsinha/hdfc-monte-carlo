"""
scripts/run_pipeline.py
-----------------------
Main daily pipeline orchestrator.
Selects top stocks by momentum/volume, runs full quant engine,
saves optimised JSON to docs/ for frontend.

Usage: python run_pipeline.py
"""

import json
import logging
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Setup logging ────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT  = SCRIPT_DIR.parent
DOCS_DIR   = REPO_ROOT / "docs"
LOGS_DIR   = REPO_ROOT / "logs"
DATA_DIR   = REPO_ROOT / "data"

DOCS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

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
from quant_engine import run_full_pipeline


# ════════════════════════════════════════════
#  STOCK SELECTOR — top stocks by score
# ════════════════════════════════════════════

def score_stock(features: dict) -> float:
    """
    Lightweight score to rank stocks for quant processing.
    Based on momentum, volume, volatility.
    Higher = more interesting for analysis.
    """
    score = 0.0
    score += features.get("mom_3m", 0)   * 2.0   # 3-month momentum
    score += features.get("mom_1m", 0)   * 1.0   # 1-month momentum
    score += features.get("mom_1y", 0)   * 0.5   # 1-year trend
    # Penalise extreme volatility slightly
    vol = features.get("sigma_annual", 0.3)
    if vol > 0.5: score -= 0.1
    return score


# ════════════════════════════════════════════
#  MAIN PIPELINE
# ════════════════════════════════════════════

def main():
    start_time = time.time()
    logger.info("=" * 65)
    logger.info("QUANTEDGE ANALYTICS — DAILY QUANT PIPELINE")
    logger.info(f"Started: {datetime.now().strftime('%d %b %Y %H:%M IST')}")
    logger.info("=" * 65)

    # ── 1. Fetch live Nifty 50 composition ──
    logger.info("\n[1/5] Fetching live Nifty 50 composition...")
    nifty50_stocks = fetch_live_nifty50()
    logger.info(f"  → {len(nifty50_stocks)} stocks in current Nifty 50")

    # Save composition for frontend
    with open(DOCS_DIR / "nifty50_composition.json", "w") as f:
        json.dump({
            "last_updated": datetime.now().strftime("%d %b %Y %H:%M IST"),
            "count"       : len(nifty50_stocks),
            "stocks"      : nifty50_stocks
        }, f)

    # All symbols to process
    all_symbols = [s["symbol"] for s in nifty50_stocks]

    # ── 2. Quick data fetch for all stocks (features only) ──
    logger.info(f"\n[2/5] Quick feature scan for stock selection...")
    quick_features = {}
    for sym in all_symbols:
        try:
            df = fetch_stock_data(sym, years=1)  # Only 1 year for selection
            if df is not None and len(df) > 60:
                quick_features[sym] = compute_features(df)
        except Exception as e:
            logger.warning(f"  {sym}: quick scan failed — {e}")

    # ── 3. Select top stocks ──
    logger.info(f"\n[3/5] Selecting top {TOP_N_STOCKS} stocks by momentum score...")
    scored = [(sym, score_stock(feat)) for sym, feat in quick_features.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [sym for sym, _ in scored[:TOP_N_STOCKS]]
    logger.info(f"  → Selected: {', '.join(selected[:10])}... and {len(selected)-10} more")

    # Save selection
    with open(DOCS_DIR / "selected_stocks.json", "w") as f:
        json.dump({
            "last_updated": datetime.now().strftime("%d %b %Y %H:%M IST"),
            "selected"    : selected,
            "scores"      : {sym: round(sc, 4) for sym, sc in scored[:TOP_N_STOCKS]}
        }, f)

    # ── 4. Full quant pipeline on selected stocks ──
    logger.info(f"\n[4/5] Running full quant pipeline on {len(selected)} stocks...")
    results = []
    failed  = []

    for i, sym in enumerate(selected, 1):
        t0 = time.time()
        logger.info(f"\n[{i:02d}/{len(selected)}] {sym}")

        # Full 10-year data
        df = fetch_stock_data(sym, years=10)
        if df is None:
            failed.append(sym)
            continue

        # Features
        features = compute_features(df)

        # Financial info
        logger.info(f"    Fetching financial info...")
        fin_info = fetch_financial_info(sym)

        # Get stock name from nifty50 list
        stock_meta = next((s for s in nifty50_stocks if s["symbol"] == sym), {})
        fin_info["name"] = stock_meta.get("name", sym)
        if not fin_info.get("sector"):
            fin_info["sector"] = stock_meta.get("sector", SECTOR_MAP.get(sym, "Other"))

        # Run pipeline
        result = run_full_pipeline(sym, features, fin_info)

        if result:
            results.append(result)
            elapsed = time.time() - t0
            logger.info(f"  ✅ Done in {elapsed:.1f}s | "
                        f"₹{result['price']:,.2f} → E[₹{result['mean_price']:,.2f}] | "
                        f"Signal: {result['signal']} | "
                        f"P(↑): {result['prob_up']:.1%} | "
                        f"Sharpe: {result['sharpe']:.2f}")
        else:
            failed.append(sym)

    # ── 5. Sort + save output ──
    logger.info(f"\n[5/5] Saving outputs...")

    # Sort by signal score then expected return
    results.sort(key=lambda x: (x.get("score", 0), x.get("expected_return", 0)), reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i

    # Separate path_charts and histogram to keep metrics.json smaller
    charts_data = {}
    for r in results:
        charts_data[r["symbol"]] = {
            "path_charts": r.pop("path_charts", {}),
            "histogram"  : r.pop("histogram", {}),
        }

    # Main metrics file
    output = {
        "last_updated"  : datetime.now().strftime("%d %b %Y %H:%M IST"),
        "total_stocks"  : len(results),
        "failed_stocks" : failed,
        "runtime_mins"  : round((time.time() - start_time) / 60, 1),
        "stocks"        : results,
    }

    with open(DOCS_DIR / "metrics.json", "w") as f:
        json.dump(output, f, separators=(',', ':'))  # Compact JSON

    with open(DOCS_DIR / "charts.json", "w") as f:
        json.dump(charts_data, f, separators=(',', ':'))

    # Summary JSON (used for homepage — tiny file)
    summary = {
        "last_updated": output["last_updated"],
        "total_stocks": len(results),
        "summary"     : [{
            "symbol"          : r["symbol"],
            "name"            : r["name"],
            "sector"          : r["sector"],
            "price"           : r["price"],
            "expected_return_pct": r["expected_return_pct"],
            "prob_up"         : r["prob_up"],
            "signal"          : r["signal"],
            "signal_color"    : r["signal_color"],
            "confidence"      : r["confidence"],
            "sharpe"          : r["sharpe"],
            "var_95"          : r["var_95"],
            "sigma_annual"    : r["sigma_annual"],
            "mispricing_pct"  : r["mispricing_pct"],
            "mom_1m"          : r["mom_1m"],
            "mom_3m"          : r["mom_3m"],
            "week52_high"     : r["week52_high"],
            "week52_low"      : r["week52_low"],
            "market_cap"      : r.get("market_cap"),
            "pe_ratio"        : r.get("pe_ratio"),
            "rank"            : r["rank"],
        } for r in results]
    }

    with open(DOCS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, separators=(',', ':'))

    # ── Final summary ──
    total_time = time.time() - start_time
    buy_count  = sum(1 for r in results if "BUY" in r.get("signal", ""))
    avg_ret    = np.mean([r["expected_return_pct"] for r in results])

    logger.info("\n" + "=" * 65)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Stocks processed : {len(results)} / {len(selected)}")
    logger.info(f"  Failed           : {len(failed)} ({', '.join(failed[:5])})")
    logger.info(f"  BUY signals      : {buy_count}")
    logger.info(f"  Avg expected ret : {avg_ret:+.1f}%")
    logger.info(f"  Total time       : {total_time/60:.1f} minutes")
    logger.info(f"  metrics.json     : {(DOCS_DIR / 'metrics.json').stat().st_size/1024:.1f} KB")
    logger.info(f"  summary.json     : {(DOCS_DIR / 'summary.json').stat().st_size/1024:.1f} KB")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
