"""
scripts/backtest.py
-------------------
QuantEdge Analytics — Signal Storage & Backtesting Engine (Phase 1)

Two modes:
  python backtest.py store   → append today's signals to signal_history.json
  python backtest.py eval    → evaluate accuracy of signals >= 30 days old
  python backtest.py report  → generate backtest_results.json (for frontend)

Signal history schema per entry:
  {
    "date"            : "2026-04-08",
    "symbol"          : "POWERGRID",
    "signal"          : "STRONG BUY",
    "score"           : 8,
    "confidence"      : 85,
    "price_at_signal" : 325.40,
    "expected_return" : 20.1,
    "prob_up"         : 0.76,
    "sharpe"          : 0.53,
    "regime"          : "Bull",
    "sentiment_label" : "Positive",
    "evaluated"       : false,
    "price_after_30d" : null,
    "actual_return"   : null,
    "signal_correct"  : null,
  }
"""

import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT  = SCRIPT_DIR.parent
DOCS_DIR   = REPO_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

HISTORY_FILE  = DOCS_DIR / "signal_history.json"
RESULTS_FILE  = DOCS_DIR / "backtest_results.json"
HOLD_DAYS     = 30


# ════════════════════════════════════════
#  LOAD / SAVE HISTORY
# ════════════════════════════════════════

def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def save_history(history: list[dict]):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, separators=(",", ":"), default=str)


# ════════════════════════════════════════
#  STORE: append today's signals
# ════════════════════════════════════════

def store_signals(results: list[dict] | None = None):
    """
    Append today's signals to signal_history.json.
    If results not passed, reads from docs/summary.json.
    """
    if results is None:
        summary_path = DOCS_DIR / "summary.json"
        if not summary_path.exists():
            logger.error("summary.json not found — run pipeline first")
            return
        with open(summary_path) as f:
            data = json.load(f)
        results = data.get("summary", [])

    today     = datetime.now().strftime("%Y-%m-%d")
    history   = load_history()
    today_syms = {e["symbol"] for e in history if e["date"] == today}

    new_entries = []
    for r in results:
        sym = r.get("symbol", "")
        if sym in today_syms:
            continue   # already stored today

        new_entries.append({
            "date"            : today,
            "symbol"          : sym,
            "name"            : r.get("name", sym),
            "signal"          : r.get("signal", "HOLD"),
            "score"           : r.get("score", 0),
            "confidence"      : r.get("confidence", 50),
            "price_at_signal" : r.get("price", 0),
            "expected_return" : r.get("expected_return_pct", 0),
            "prob_up"         : r.get("prob_up", 0.5),
            "sharpe"          : r.get("sharpe", 0),
            "var_95"          : r.get("var_95", 0),
            "regime"          : r.get("regime", "Unknown"),
            "sentiment_label" : r.get("sentiment_label", "Neutral"),
            "sentiment_score" : r.get("sentiment_score", 0),
            "has_earnings"    : r.get("has_earnings_soon", False),
            "sector"          : r.get("sector", "Other"),
            "beta_nifty"      : r.get("beta_nifty", 1.0),
            "eval_date"       : (
                datetime.strptime(today, "%Y-%m-%d") + timedelta(days=HOLD_DAYS)
            ).strftime("%Y-%m-%d"),
            "evaluated"       : False,
            "price_after_30d" : None,
            "actual_return"   : None,
            "signal_correct"  : None,
        })

    if new_entries:
        history.extend(new_entries)
        save_history(history)
        logger.info(f"  Stored {len(new_entries)} new signals for {today}")
    else:
        logger.info(f"  Signals for {today} already stored")


# ════════════════════════════════════════
#  EVALUATE: fill in actual returns
# ════════════════════════════════════════

def evaluate_signals():
    """
    For entries where eval_date <= today and not yet evaluated,
    fetch current price and compute actual return + correctness.
    """
    history    = load_history()
    today_str  = datetime.now().strftime("%Y-%m-%d")
    to_eval    = [
        e for e in history
        if not e.get("evaluated", False) and e.get("eval_date", "9999-99-99") <= today_str
    ]

    if not to_eval:
        logger.info("  No signals ready for evaluation yet")
        return

    # Group by symbol to minimise API calls
    syms = list({e["symbol"] for e in to_eval})
    logger.info(f"  Evaluating {len(to_eval)} signals across {len(syms)} stocks...")

    prices = {}
    for sym in syms:
        try:
            ticker  = yf.Ticker(f"{sym}.NS")
            price_s = ticker.fast_info
            p       = getattr(price_s, "last_price", None)
            if p:
                prices[sym] = float(p)
            else:
                # Fallback: download last 5 days
                df = yf.download(f"{sym}.NS", period="5d",
                                 auto_adjust=True, progress=False)
                if not df.empty:
                    prices[sym] = float(df["Close"].squeeze().iloc[-1])
        except Exception as e:
            logger.warning(f"  Price fetch failed for {sym}: {e}")
        time.sleep(0.2)

    n_evaluated = 0
    for entry in history:
        if not entry["evaluated"] and entry.get("eval_date", "9999") <= today_str:
            sym = entry["symbol"]
            p   = prices.get(sym)
            if p and entry.get("price_at_signal", 0) > 0:
                entry["price_after_30d"] = round(p, 2)
                actual_ret = (p - entry["price_at_signal"]) / entry["price_at_signal"] * 100
                entry["actual_return"]   = round(actual_ret, 2)

                sig = entry.get("signal", "HOLD")
                if "BUY" in sig:
                    entry["signal_correct"] = bool(actual_ret > 0)
                elif sig == "AVOID":
                    entry["signal_correct"] = bool(actual_ret < 0)
                elif sig == "RISKY":
                    entry["signal_correct"] = bool(actual_ret < -2)
                else:
                    entry["signal_correct"] = None   # HOLD — ambiguous

                entry["evaluated"] = True
                n_evaluated += 1

    save_history(history)
    logger.info(f"  Evaluated {n_evaluated} signals")


# ════════════════════════════════════════
#  REPORT: generate backtest_results.json
# ════════════════════════════════════════

def generate_report():
    """
    Aggregates evaluated signals into statistics by signal type.
    Saves to docs/backtest_results.json for frontend display.
    """
    history   = load_history()
    evaluated = [e for e in history if e["evaluated"] and e["actual_return"] is not None]

    if len(evaluated) < 10:
        logger.info(f"  Only {len(evaluated)} evaluated signals — need 10+ for report")
        report = {
            "last_updated"    : datetime.now().strftime("%d %b %Y %H:%M IST"),
            "status"          : "insufficient_data",
            "total_evaluated" : len(evaluated),
            "message"         : f"Backtesting requires 10+ evaluated signals. Currently {len(evaluated)}.",
            "by_signal"       : {},
            "overall"         : {},
        }
        with open(RESULTS_FILE, "w") as f:
            json.dump(report, f, indent=2)
        return

    df = pd.DataFrame(evaluated)

    # ── By signal type ──────────────────────────────────────
    by_signal = {}
    for sig in ["STRONG BUY", "BUY", "HOLD", "RISKY", "AVOID"]:
        sub = df[df["signal"] == sig]
        if len(sub) == 0:
            continue
        returns = sub["actual_return"].dropna()
        correct = sub["signal_correct"].dropna()
        by_signal[sig] = {
            "count"           : int(len(sub)),
            "avg_return_pct"  : round(float(returns.mean()), 2),
            "median_return_pct": round(float(returns.median()), 2),
            "std_return_pct"  : round(float(returns.std()), 2),
            "win_rate_pct"    : round(float(correct.mean() * 100), 1) if len(correct) > 0 else None,
            "best_pct"        : round(float(returns.max()), 2),
            "worst_pct"       : round(float(returns.min()), 2),
            "positive_count"  : int((returns > 0).sum()),
            "negative_count"  : int((returns <= 0).sum()),
        }

    # ── Overall stats ───────────────────────────────────────
    all_returns = df["actual_return"].dropna()
    all_correct = df[df["signal"] != "HOLD"]["signal_correct"].dropna()

    # Buy&Hold Nifty benchmark over same period
    try:
        start = datetime.strptime(df["date"].min(), "%Y-%m-%d") - timedelta(days=5)
        end   = datetime.now()
        nifty = yf.download("^NSEI", start=start, end=end,
                            auto_adjust=True, progress=False)
        if not nifty.empty:
            nifty_close = nifty["Close"].squeeze()
            nifty_ret   = float((nifty_close.iloc[-1] / nifty_close.iloc[0] - 1) * 100)
        else:
            nifty_ret = None
    except Exception:
        nifty_ret = None

    # Best picks (STRONG BUY hits)
    best_picks = (
        df[df["signal"] == "STRONG BUY"]
        .nlargest(5, "actual_return")
        [["symbol", "date", "price_at_signal", "actual_return"]]
        .to_dict("records")
    )

    # Worst picks (to show transparency)
    worst_picks = (
        df[df["signal"].isin(["STRONG BUY","BUY"])]
        .nsmallest(3, "actual_return")
        [["symbol", "date", "price_at_signal", "actual_return"]]
        .to_dict("records")
    )

    # Monthly breakdown
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    monthly = (
        df.groupby("month")["actual_return"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_return", "count": "signals"})
        .round(2)
        .to_dict("index")
    )

    report = {
        "last_updated"    : datetime.now().strftime("%d %b %Y %H:%M IST"),
        "status"          : "active",
        "hold_days"       : HOLD_DAYS,
        "total_evaluated" : int(len(evaluated)),
        "date_range"      : {"from": df["date"].min(), "to": df["date"].max()},
        "overall": {
            "avg_return_pct"     : round(float(all_returns.mean()), 2),
            "median_return_pct"  : round(float(all_returns.median()), 2),
            "overall_accuracy_pct": round(float(all_correct.mean() * 100), 1) if len(all_correct) > 0 else None,
            "nifty_benchmark_pct": round(nifty_ret, 2) if nifty_ret else None,
            "alpha_pct"          : round(float(all_returns.mean()) - (nifty_ret or 0), 2),
            "best_signal"        : best_picks,
            "worst_signal"       : worst_picks,
        },
        "by_signal" : by_signal,
        "by_month"  : monthly,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\n{'='*55}")
    logger.info(f"  Backtest Report ({len(evaluated)} signals, {HOLD_DAYS}-day hold)")
    logger.info(f"  Overall avg return : {report['overall']['avg_return_pct']:+.2f}%")
    logger.info(f"  Overall accuracy   : {report['overall'].get('overall_accuracy_pct','—')}%")
    logger.info(f"  vs Nifty benchmark : {nifty_ret:+.2f}%" if nifty_ret else "  Nifty benchmark: N/A")
    for sig, stats in by_signal.items():
        logger.info(
            f"  {sig:<12}: {stats['count']:>3} signals | "
            f"avg {stats['avg_return_pct']:+.1f}% | "
            f"accuracy {stats.get('win_rate_pct','—')}%"
        )
    logger.info(f"{'='*55}")
    logger.info(f"  Saved: {RESULTS_FILE}")


# ════════════════════════════════════════
#  CLI ENTRY POINT
# ════════════════════════════════════════

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "store"

    if mode == "store":
        logger.info("Storing today's signals...")
        store_signals()

    elif mode == "eval":
        logger.info("Evaluating matured signals...")
        evaluate_signals()

    elif mode == "report":
        logger.info("Generating backtest report...")
        evaluate_signals()
        generate_report()

    elif mode == "all":
        logger.info("Full backtest cycle: store → eval → report")
        store_signals()
        evaluate_signals()
        generate_report()

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python backtest.py [store|eval|report|all]")
        sys.exit(1)
