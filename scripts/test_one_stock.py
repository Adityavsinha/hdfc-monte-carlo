"""
scripts/test_one_stock.py
--------------------------
Quick test: run full pipeline on ONE stock to verify everything works.
Usage: python test_one_stock.py RELIANCE
       python test_one_stock.py HDFCBANK
       python test_one_stock.py  (defaults to NTPC)
"""
import sys, json, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT  = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))

def test_stock(symbol: str = "NTPC"):
    logger.info(f"Testing full pipeline on: {symbol}")
    logger.info("="*50)

    # Step 1: Fetch data
    logger.info("[1/5] Fetching stock data...")
    from data_ingestion import fetch_stock_data, compute_features, fetch_financial_info
    df = fetch_stock_data(symbol, years=5)
    if df is None or df.empty:
        logger.error(f"Could not fetch data for {symbol}")
        return
    logger.info(f"  Got {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    logger.info(f"  Current price: ₹{df['Close'].iloc[-1]:,.2f}")

    # Step 2: Compute features
    logger.info("[2/5] Computing features...")
    features = compute_features(df)
    logger.info(f"  Sigma (daily): {features['sigma_daily']:.4f}")
    logger.info(f"  Mom 1M: {features['mom_1m']:.2%}")
    logger.info(f"  Mom 3M: {features['mom_3m']:.2%}")

    # Step 3: Fetch Nifty returns
    logger.info("[3/5] Fetching Nifty 50 index returns...")
    import yfinance as yf
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    end   = datetime.today()
    start = end - timedelta(days=1000)
    nifty_df = yf.download("^NSEI", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(nifty_df.columns, pd.MultiIndex):
        nifty_df.columns = nifty_df.columns.get_level_values(0)
    close = pd.to_numeric(nifty_df["Close"].squeeze(), errors="coerce").dropna()
    nifty_returns = np.log(close / close.shift(1)).dropna()
    logger.info(f"  Nifty returns: {len(nifty_returns)} days")

    # Step 4: Financial info
    logger.info("[4/5] Fetching financial info...")
    fin_info = fetch_financial_info(symbol)
    logger.info(f"  Name: {fin_info.get('name', symbol)}")
    logger.info(f"  Sector: {fin_info.get('sector', 'Unknown')}")
    logger.info(f"  P/E: {fin_info.get('pe_ratio', 'N/A')}")
    logger.info(f"  Beta (yfinance): {fin_info.get('beta', 'N/A')}")

    # Step 5: Run full pipeline
    logger.info("[5/5] Running Monte Carlo simulation...")
    from quant_engine import run_full_pipeline
    result = run_full_pipeline(symbol, features, fin_info, nifty_returns)

    if result:
        logger.info("")
        logger.info("="*50)
        logger.info(f"RESULT FOR {symbol}:")
        logger.info(f"  Price:           ₹{result['price']:,.2f}")
        logger.info(f"  Expected Price:  ₹{result['mean_price']:,.2f}")
        logger.info(f"  Expected Return: {result['expected_return_pct']:+.1f}%")
        logger.info(f"  P(up):           {result['prob_up']:.1%}")
        logger.info(f"  Signal:          {result['signal']} (confidence: {result['confidence']}%)")
        logger.info(f"  Nifty Beta:      {result['beta_nifty']:.3f}")
        logger.info(f"  Sharpe Ratio:    {result['sharpe']:.3f}")
        logger.info(f"  VaR 95%:         {result['var_95']:.2%}")
        logger.info(f"  Annual Sigma:    {result['sigma_annual']:.2%}")
        logger.info(f"  RSI (14):        {result.get('rsi_14', 'N/A')}")
        logger.info(f"  MACD Cross:      {result.get('macd_cross', 'N/A')}")
        logger.info(f"  Fund. Grade:     {result.get('fundamental_grade', 'N/A')}")
        logger.info(f"  Drift Method:    {result['drift_method']}")
        logger.info(f"  CI 5th–95th:     ₹{result['ci_5']:,.2f} – ₹{result['ci_95']:,.2f}")
        logger.info("="*50)
        logger.info("TEST PASSED - Pipeline working correctly!")

        # Save test output
        out_path = REPO_ROOT / "docs" / f"test_{symbol}.json"
        with open(out_path, "w") as f:
            # Remove large path_charts for test output
            save = {k: v for k, v in result.items() if k not in ['path_charts','histogram']}
            json.dump(save, f, indent=2, default=str)
        logger.info(f"  Test output saved: {out_path}")
    else:
        logger.error("TEST FAILED - pipeline returned None")


if __name__ == "__main__":
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "NTPC"
    test_stock(symbol)
