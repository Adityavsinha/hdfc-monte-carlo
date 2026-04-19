"""
scripts/post_telegram.py
-------------------------
Posts daily top signals to your Telegram channel.
Runs automatically via GitHub Actions after the daily pipeline.

SETUP:
  1. Create a Telegram bot: talk to @BotFather on Telegram
     - Send /newbot
     - Give it a name like "QuantEdge Signals Bot"
     - Copy the token it gives you

  2. Create a Telegram channel (e.g. @quantedge_signals)
     - Add your bot as admin of the channel

  3. Add to GitHub Secrets:
     TELEGRAM_BOT_TOKEN = your_bot_token
     TELEGRAM_CHANNEL_ID = @quantedge_signals  (or numeric ID)

  4. The daily.yml workflow calls this script automatically.

LOCAL TESTING:
  Create .env with those values, then:
  python scripts/post_telegram.py
"""
import json, os, requests, logging
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()
DOCS_DIR   = SCRIPT_DIR.parent / "docs"

BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", "")


def send_message(text: str, parse_mode: str = "Markdown"):
    if not BOT_TOKEN or not CHANNEL_ID:
        logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHANNEL_ID not set — skipping")
        return False

    url  = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHANNEL_ID, "text": text, "parse_mode": parse_mode,
            "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=data, timeout=10)
        if r.status_code == 200:
            logger.info("Telegram message sent successfully")
            return True
        else:
            logger.error(f"Telegram error: {r.status_code} — {r.text[:200]}")
            return False
    except Exception as e:
        logger.error(f"Telegram request failed: {e}")
        return False


def build_daily_message(summary_data: dict) -> str:
    stocks  = summary_data.get("summary", [])
    updated = summary_data.get("last_updated", datetime.now().strftime("%d %b %Y"))

    # Top strong buys
    strong_buys = [s for s in stocks if s.get("signal") == "STRONG BUY"][:3]
    avoids      = [s for s in stocks if s.get("signal") == "AVOID"][:2]

    # Avg return across all
    avg_ret = sum(s.get("expected_return_pct", 0) for s in stocks) / max(len(stocks), 1)
    bullish = sum(1 for s in stocks if s.get("prob_up", 0) > 0.55)

    msg  = f"📊 *QuantEdge Daily Signals* — {updated}\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"*Market Overview:*\n"
    msg += f"  Stocks simulated: {len(stocks)}\n"
    msg += f"  Bullish (P↑ > 55%): {bullish}\n"
    msg += f"  Avg expected return: {avg_ret:+.1f}%\n\n"

    if strong_buys:
        msg += f"*🟢 Top STRONG BUY signals:*\n"
        for s in strong_buys:
            msg += f"  *{s['symbol']}* — P(↑): {s.get('prob_up',0)*100:.0f}% | E[Return]: {s.get('expected_return_pct',0):+.1f}% | Sharpe: {s.get('sharpe',0):.2f}\n"
        msg += "\n"

    if avoids:
        msg += f"*🔴 AVOID signals:*\n"
        for s in avoids:
            msg += f"  *{s['symbol']}* — P(↑): {s.get('prob_up',0)*100:.0f}% | E[Return]: {s.get('expected_return_pct',0):+.1f}%\n"
        msg += "\n"

    msg += f"*Full analysis with charts:*\n"
    msg += f"🔗 https://quantedgeanalytics.co.in\n\n"
    msg += f"_⚠️ Not financial advice. Educational model only._"

    return msg


def build_weekly_accuracy_message(backtest_data: dict) -> str:
    if not backtest_data:
        return ""

    msg  = f"📈 *QuantEdge Weekly Accuracy Report*\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"*{backtest_data.get('lookback_days', 30)}-day Signal Accuracy:*\n"
    msg += f"Overall: *{backtest_data.get('overall_accuracy', 0)}%*\n\n"

    by_sig = backtest_data.get("by_signal", {})
    for sig, stats in by_sig.items():
        emoji = "🟢" if "BUY" in sig else "🔴" if sig == "AVOID" else "🟡"
        msg += f"{emoji} *{sig}:* avg {stats['avg_return']:+.1f}% | win rate {stats['win_rate']}% (n={stats['count']})\n"

    msg += f"\n_Based on {backtest_data.get('total_signals',0)} historical signals_\n"
    msg += f"🔗 https://quantedgeanalytics.co.in/backtest_report.html"
    return msg


def main():
    # Load summary
    summary_path = DOCS_DIR / "summary.json"
    if not summary_path.exists():
        logger.error("summary.json not found")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    # Send daily signals
    msg = build_daily_message(summary)
    send_message(msg)

    # On Mondays — also send accuracy report
    if datetime.now().weekday() == 0:
        backtest_path = DOCS_DIR / "backtest_results.json"
        if backtest_path.exists():
            with open(backtest_path) as f:
                bt = json.load(f)
            accuracy_msg = build_weekly_accuracy_message(bt)
            if accuracy_msg:
                send_message(accuracy_msg)


if __name__ == "__main__":
    main()
