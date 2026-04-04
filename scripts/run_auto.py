"""
run_auto.py
-----------
This is the file that runs automatically every day via Task Scheduler.
It does EVERYTHING in one shot:
  1. Fetch latest HDFC Bank price (rolling 10-year window)
  2. Run Monte Carlo simulation
  3. Generate all charts
  4. Export to Excel
  5. Upload to Google Drive
  6. Update Google Sheets dashboard
  7. Save shareable links

You can also run this manually anytime:
    python run_auto.py
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Logging (to file + screen) ───────────────
Path("logs").mkdir(exist_ok=True)
log_file = f"logs/auto_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ],
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    cfg_path = Path("config.json")
    if not cfg_path.exists():
        logger.error("config.json not found! Please run: python setup_google.py first.")
        sys.exit(1)
    with open(cfg_path) as f:
        return json.load(f)


def main():
    logger.info("=" * 60)
    logger.info("HDFC MONTE CARLO — AUTOMATED DAILY RUN")
    logger.info(f"Started: {datetime.now().strftime('%d %b %Y %H:%M IST')}")
    logger.info("=" * 60)

    config = load_config()

    # ── 1. DATA ─────────────────────────────
    logger.info("[1/6] Fetching latest price data...")
    from data_ingestion import load_or_fetch, compute_returns
    df = load_or_fetch(ticker=config.get("ticker", "HDFCBANK.NS"))
    log_returns = compute_returns(df)
    close_val   = df["Close"].iloc[-1]
    current_price = float(close_val.iloc[0]) if hasattr(close_val, "iloc") else float(close_val)
    logger.info(f"      Latest close = ₹{current_price:,.2f} | {len(df)} trading days loaded")

    # ── 2. SIMULATION ───────────────────────
    logger.info("[2/6] Running Monte Carlo simulation...")
    from simulation_engine import SimConfig, run_simulation
    cfg = SimConfig(
        n_simulations = config.get("simulations", 10_000),
        horizon_days  = config.get("horizon_days", 252),
        use_fat_tails = True,
    )
    results = run_simulation(log_returns, current_price, cfg)

    # ── 3. CHARTS ───────────────────────────
    logger.info("[3/6] Generating charts...")
    try:
        import matplotlib
        matplotlib.use("Agg")   # headless — no screen needed
        from visualization import render_all
        render_all(results, log_returns)
        logger.info("      Charts saved to outputs/")
    except Exception as e:
        logger.warning(f"      Chart generation failed: {e}")

    # HTML dashboard
    try:
        from html_dashboard import generate_html_dashboard
        html_path = generate_html_dashboard(results, log_returns)
        logger.info(f"      HTML dashboard → {html_path}")
    except Exception as e:
        logger.warning(f"      HTML dashboard failed: {e}")

    # ── 4. EXCEL ────────────────────────────
    logger.info("[4/6] Creating Excel dashboard...")
    from dashboard_export import export_to_excel
    export_to_excel(results, log_returns)

    # ── 5. GOOGLE DRIVE ─────────────────────
    logger.info("[5/6] Uploading to Google Drive...")
    try:
        from google_sync import get_credentials, upload_all_outputs
        creds       = get_credentials()
        drive_links = upload_all_outputs(config["drive_folder_id"], creds)
        logger.info(f"      {len(drive_links)} files uploaded to Drive")
    except Exception as e:
        logger.error(f"      Drive upload failed: {e}")
        drive_links = {}
        creds       = None

    # ── 6. GOOGLE SHEETS ────────────────────
    logger.info("[6/6] Updating Google Sheets dashboard...")
    sheets_url = ""
    if creds:
        try:
            from google_sync import write_sheets_dashboard, save_links
            sheets_url = write_sheets_dashboard(
                config["spreadsheet_id"], results, drive_links, creds
            )
            save_links(drive_links, sheets_url)
            logger.info(f"      Dashboard updated → {sheets_url}")
        except Exception as e:
            logger.error(f"      Sheets update failed: {e}")

    # ── SUMMARY ─────────────────────────────
    S0 = results.S0
    T  = results.config.horizon_days
    logger.info("")
    logger.info("══════════════ RESULTS SUMMARY ══════════════")
    logger.info(f"  Current price       : ₹{S0:,.2f}")
    logger.info(f"  Expected (Day {T:3d})  : ₹{results.mean_price:,.2f}  ({(results.mean_price/S0-1)*100:+.1f}%)")
    logger.info(f"  Best  case (95th)   : ₹{results.ci_95:,.2f}  ({(results.ci_95/S0-1)*100:+.1f}%)")
    logger.info(f"  Worst case (5th)    : ₹{results.ci_5:,.2f}  ({(results.ci_5/S0-1)*100:+.1f}%)")
    logger.info(f"  VaR 95%             : ₹{results.var_95:,.2f}")
    logger.info(f"  P(price goes up)    : {results.prob_increase:.1%}")
    logger.info(f"  Annual volatility   : {results.sigma*(252**0.5)*100:.2f}%")
    logger.info("══════════════════════════════════════════════")
    if sheets_url:
        logger.info(f"  📊 Live Dashboard   : {sheets_url}")
    logger.info(f"  ✅ Run complete: {datetime.now().strftime('%H:%M IST')}")
    logger.info("")


if __name__ == "__main__":
    main()
