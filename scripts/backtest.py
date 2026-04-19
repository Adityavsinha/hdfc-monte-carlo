"""
scripts/backtest.py
-------------------
Stores daily signals and evaluates accuracy after 30/60/90 days.

HOW IT WORKS:
  1. Every day, saves today's signals to docs/signal_history.json
  2. After 30 days, compares predicted signals to actual returns
  3. Outputs accuracy stats → shown on website as credibility proof

USAGE:
  python backtest.py store    → saves today's signals
  python backtest.py evaluate → computes accuracy of past signals
  python backtest.py report   → generates HTML report
"""
import json, sys, logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent.resolve()
REPO_ROOT   = SCRIPT_DIR.parent
DOCS_DIR    = REPO_ROOT / "docs"
HISTORY_FILE = DOCS_DIR / "signal_history.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_history() -> list:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(history: list):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, separators=(",", ":"))


def store_todays_signals():
    """Save today's signals from summary.json to history."""
    summary_path = DOCS_DIR / "summary.json"
    if not summary_path.exists():
        logger.error("summary.json not found — run pipeline first")
        return

    with open(summary_path) as f:
        data = json.load(f)

    today     = datetime.now().strftime("%Y-%m-%d")
    history   = load_history()
    existing  = {(r["date"], r["symbol"]) for r in history}

    added = 0
    for s in data.get("summary", []):
        key = (today, s["symbol"])
        if key in existing:
            continue
        history.append({
            "date"            : today,
            "symbol"          : s["symbol"],
            "signal"          : s.get("signal", ""),
            "price_at_signal" : s.get("price", 0),
            "expected_return" : s.get("expected_return_pct", 0),
            "prob_up"         : s.get("prob_up", 0),
            "sharpe"          : s.get("sharpe", 0),
        })
        added += 1

    save_history(history)
    logger.info(f"Stored {added} signals for {today}. Total history: {len(history)}")


def evaluate_signals(lookback_days: int = 30):
    """Fetch actual returns for signals older than lookback_days."""
    history = load_history()
    if not history:
        logger.info("No signal history yet.")
        return {}

    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    # Signals old enough to evaluate
    to_eval = [r for r in history if r["date"] <= cutoff and "actual_return" not in r]

    if not to_eval:
        logger.info(f"No signals to evaluate (need signals older than {lookback_days} days).")
    else:
        logger.info(f"Evaluating {len(to_eval)} signals...")

        # Get unique symbols
        symbols = list({r["symbol"] for r in to_eval})
        price_data = {}

        for sym in symbols:
            try:
                df = yf.download(sym+".NS", period="6mo", interval="1d",
                                 auto_adjust=True, progress=False)
                if df.empty: continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                price_data[sym] = df["Close"].squeeze().dropna()
            except Exception:
                pass

        evaluated = 0
        for rec in history:
            if rec.get("actual_return") is not None:
                continue
            if rec["date"] > cutoff:
                continue

            sym = rec["symbol"]
            if sym not in price_data:
                continue

            prices = price_data[sym]
            try:
                # Find price lookback_days after signal date
                sig_date   = pd.Timestamp(rec["date"])
                future     = prices[prices.index > sig_date]
                if len(future) < lookback_days // 2:
                    continue

                actual_idx  = min(lookback_days - 1, len(future) - 1)
                price_then  = rec["price_at_signal"]
                price_now   = float(future.iloc[actual_idx])
                actual_ret  = (price_now - price_then) / price_then * 100

                rec["actual_return"]  = round(actual_ret, 2)
                rec["actual_price"]   = round(price_now, 2)
                rec["eval_days"]      = lookback_days
                rec["correct"]        = (
                    (rec["signal"] in ("STRONG BUY","BUY") and actual_ret > 0) or
                    (rec["signal"] == "AVOID"              and actual_ret < 0) or
                    (rec["signal"] in ("HOLD","RISKY")     and abs(actual_ret) < 5)
                )
                evaluated += 1
            except Exception:
                pass

        save_history(history)
        logger.info(f"Evaluated {evaluated} signals.")

    # Compute stats
    evaluated_recs = [r for r in history if r.get("actual_return") is not None]
    if not evaluated_recs:
        return {}

    df_eval = pd.DataFrame(evaluated_recs)

    stats = {}
    for sig in df_eval["signal"].unique():
        sub = df_eval[df_eval["signal"] == sig]
        stats[sig] = {
            "count"        : len(sub),
            "avg_return"   : round(sub["actual_return"].mean(), 2),
            "win_rate"     : round(sub["correct"].mean() * 100, 1),
            "best"         : round(sub["actual_return"].max(), 2),
            "worst"        : round(sub["actual_return"].min(), 2),
            "std"          : round(sub["actual_return"].std(), 2),
        }

    overall_accuracy = round(df_eval["correct"].mean() * 100, 1)

    result = {
        "last_evaluated"   : datetime.now().strftime("%d %b %Y %H:%M IST"),
        "total_signals"    : len(evaluated_recs),
        "overall_accuracy" : overall_accuracy,
        "by_signal"        : stats,
        "lookback_days"    : lookback_days,
    }

    # Save to docs for website
    with open(DOCS_DIR / "backtest_results.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info("\n" + "="*50)
    logger.info(f"BACKTEST RESULTS ({lookback_days}-day lookback):")
    logger.info(f"Overall accuracy: {overall_accuracy}%")
    for sig, s in stats.items():
        logger.info(f"  {sig:12s}: avg {s['avg_return']:+.1f}% | win rate {s['win_rate']}% | n={s['count']}")
    logger.info("="*50)

    return result


def generate_report():
    """Generate a simple HTML report of backtest results."""
    results_path = DOCS_DIR / "backtest_results.json"
    if not results_path.exists():
        logger.error("No backtest results yet. Run: python backtest.py evaluate")
        return

    with open(results_path) as f:
        r = json.load(f)

    html = f"""<!DOCTYPE html>
<html><head>
<title>QuantEdge Signal Accuracy Report</title>
<meta charset="UTF-8">
<style>
body{{font-family:'Plus Jakarta Sans',sans-serif;background:#F0F4FB;padding:40px;color:#111827}}
.card{{background:#fff;border-radius:12px;padding:24px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
h1{{color:#2F80ED;font-size:24px;margin-bottom:4px}}
.big{{font-size:48px;font-weight:800;color:#2F80ED}}
table{{width:100%;border-collapse:collapse;margin-top:12px}}
th{{background:#F0F4FB;padding:10px 14px;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:#6B7280}}
td{{padding:10px 14px;border-bottom:1px solid #E5E7EB;font-size:13px}}
.up{{color:#059669;font-weight:700}}.dn{{color:#DC2626;font-weight:700}}
</style></head>
<body>
<div class="card">
  <h1>QuantEdge Signal Accuracy Report</h1>
  <p style="color:#6B7280">Last evaluated: {r['last_evaluated']} · {r['lookback_days']}-day lookback · {r['total_signals']} signals</p>
  <div class="big">{r['overall_accuracy']}%</div>
  <p style="color:#6B7280">Overall accuracy (signal direction correct)</p>
</div>
<div class="card">
  <h2 style="font-size:16px;margin-bottom:12px">Results by Signal Type</h2>
  <table>
    <tr><th>Signal</th><th>Avg Return</th><th>Win Rate</th><th>Best</th><th>Worst</th><th>Count</th></tr>
    {''.join(f"<tr><td><strong>{s}</strong></td><td class='{'up' if v['avg_return']>0 else 'dn'}'>{v['avg_return']:+.1f}%</td><td>{v['win_rate']}%</td><td class='up'>{v['best']:+.1f}%</td><td class='dn'>{v['worst']:+.1f}%</td><td>{v['count']}</td></tr>" for s,v in r['by_signal'].items())}
  </table>
</div>
<div class="card" style="font-size:11px;color:#9CA3AF">
  ⚠️ For educational purposes only. Past signal accuracy does not guarantee future returns.
  This is a quantitative model backtested on historical data. Not financial advice.
</div>
</body></html>"""

    out = DOCS_DIR / "backtest_report.html"
    with open(out, "w") as f:
        f.write(html)
    logger.info(f"Report saved: {out}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "store"

    if cmd == "store":
        store_todays_signals()
    elif cmd == "evaluate":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        evaluate_signals(days)
    elif cmd == "report":
        generate_report()
    else:
        print("Usage: python backtest.py [store|evaluate|report]")
