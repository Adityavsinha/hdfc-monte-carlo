"""
scripts/config.py — QuantEdge Analytics Phase 1
"""

# Core Simulation
N_SIMULATIONS    = 5_000
HORIZON_DAYS     = 252
ROLLING_VOL_WIN  = 63
RISK_FREE_RATE   = 0.065
T_DOF            = 5
TOP_N_STOCKS     = 100

# Signal Thresholds
BUY_PROB_THRESHOLD    = 0.60
RISKY_VAR_THRESHOLD   = 0.20
HOLD_SHARPE_MIN       = 0.3

# Phase 1 Feature Flags
USE_FAMA_FRENCH       = True
USE_REGIME_DETECTION  = True
USE_SENTIMENT         = True
USE_EARNINGS_RISK     = True
USE_CORRELATED_MC     = True

# Fama-French
FF3_LOOKBACK_DAYS     = 504
SMB_ANNUAL_PREMIUM    = 0.030
HML_ANNUAL_PREMIUM    = 0.035
MOM_ANNUAL_PREMIUM    = 0.040

# Regime Detection
REGIME_LOOKBACK_DAYS  = 756
REGIME_BULL_ADJ       = 0.020 / 252
REGIME_BEAR_ADJ       = -0.025 / 252

# Earnings Risk
EARNINGS_WINDOW_DAYS  = 30
EARNINGS_VOL_MULT     = 1.35

# Backtesting
BACKTEST_HOLD_DAYS    = 30
SIGNAL_HISTORY_FILE   = "docs/signal_history.json"
BACKTEST_RESULTS_FILE = "docs/backtest_results.json"

# Correlated MC
CORR_MC_N             = 2_000
CORR_MC_MIN_STOCKS    = 5

# Market Parameters (Indian market calibrated)
MARKET_RETURN_ANNUAL  = 0.15
EQUITY_PREMIUM        = MARKET_RETURN_ANNUAL - RISK_FREE_RATE
MIN_MU_ANNUAL         = -0.05
MAX_MU_ANNUAL         = 0.32

# Nifty 50 Symbols
NIFTY50_SYMBOLS = [
    "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK",
    "INFOSYS","SBIN","HINDUNILVR","ITC","LT",
    "KOTAKBANK","AXISBANK","ASIANPAINT","MARUTI","SUNPHARMA",
    "TITAN","ULTRACEMCO","BAJFINANCE","WIPRO","ONGC",
    "NTPC","POWERGRID","HCLTECH","M&M","TATAMOTORS",
    "TATASTEEL","ADANIENT","ADANIPORTS","COALINDIA","BAJAJFINSV",
    "NESTLEIND","CIPLA","DRREDDY","HEROMOTOCO","BRITANNIA",
    "TECHM","GRASIM","BPCL","EICHERMOT","DIVISLAB",
    "APOLLOHOSP","TATACONSUM","BAJAJ-AUTO","HINDALCO","JSWSTEEL",
    "SHRIRAMFIN","INDUSINDBK","SBILIFE","HDFCLIFE","BEL"
]

NIFTY_NEXT50 = [
    "PIDILITIND","HAVELLS","SIEMENS","ABB","TORNTPHARM",
    "MUTHOOTFIN","LUPIN","AMBUJACEM","TRENT","IRCTC",
    "CONCOR","PRESTIGE","CANBK","BANKBARODA","FEDERALBNK",
    "IDFCFIRSTB","BANDHANBNK","MFSL","MAXHEALTH","FORTIS",
    "GODREJCP","DABUR","EMAMILTD","NHPC","SAIL",
    "NATIONALUM","VEDL","JINDALSTEL","APLAPOLLO","DMART",
    "ZYDUSLIFE","AUROPHARMA","ALKEM","BIOCON","SYNGENE",
    "LAURUSLABS","GRANULES","LODHA","NYKAA","CHOLAFIN",
    "JIOFIN","INDIGO","RECLTD","PFC","IRFC",
    "ADANIGREEN","JSWENERGY","ETERNAL","TATAPOWER","IDFCFIRSTB"
]

FULL_UNIVERSE = list(dict.fromkeys(NIFTY50_SYMBOLS + NIFTY_NEXT50))

SECTOR_MAP = {
    "RELIANCE":"Energy","TCS":"Technology","HDFCBANK":"Financial Services",
    "BHARTIARTL":"Communication Services","ICICIBANK":"Financial Services",
    "INFOSYS":"Technology","SBIN":"Financial Services","HINDUNILVR":"Consumer Defensive",
    "ITC":"Consumer Defensive","LT":"Industrials","KOTAKBANK":"Financial Services",
    "AXISBANK":"Financial Services","ASIANPAINT":"Consumer Cyclical","MARUTI":"Consumer Cyclical",
    "SUNPHARMA":"Healthcare","TITAN":"Consumer Cyclical","ULTRACEMCO":"Basic Materials",
    "BAJFINANCE":"Financial Services","WIPRO":"Technology","ONGC":"Energy",
    "NTPC":"Utilities","POWERGRID":"Utilities","HCLTECH":"Technology",
    "M&M":"Consumer Cyclical","TATAMOTORS":"Consumer Cyclical","TATASTEEL":"Basic Materials",
    "ADANIENT":"Industrials","ADANIPORTS":"Industrials","COALINDIA":"Energy",
    "BAJAJFINSV":"Financial Services","NESTLEIND":"Consumer Defensive","CIPLA":"Healthcare",
    "DRREDDY":"Healthcare","HEROMOTOCO":"Consumer Cyclical","BRITANNIA":"Consumer Defensive",
    "TECHM":"Technology","GRASIM":"Basic Materials","BPCL":"Energy",
    "EICHERMOT":"Consumer Cyclical","DIVISLAB":"Healthcare","APOLLOHOSP":"Healthcare",
    "TATACONSUM":"Consumer Defensive","BAJAJ-AUTO":"Consumer Cyclical",
    "HINDALCO":"Basic Materials","JSWSTEEL":"Basic Materials",
    "SHRIRAMFIN":"Financial Services","INDUSINDBK":"Financial Services",
    "SBILIFE":"Financial Services","HDFCLIFE":"Financial Services","BEL":"Industrials",
    "ETERNAL":"Consumer Cyclical","NYKAA":"Consumer Cyclical","IRCTC":"Industrials",
    "DMART":"Consumer Defensive","TRENT":"Consumer Cyclical","TATAPOWER":"Utilities",
    "VEDL":"Basic Materials","SAIL":"Basic Materials","BANKBARODA":"Financial Services",
    "CANBK":"Financial Services","FEDERALBNK":"Financial Services",
    "IDFCFIRSTB":"Financial Services","BANDHANBNK":"Financial Services",
    "PIDILITIND":"Basic Materials","HAVELLS":"Industrials","SIEMENS":"Industrials",
    "ABB":"Industrials","TORNTPHARM":"Healthcare","MUTHOOTFIN":"Financial Services",
    "LUPIN":"Healthcare","AMBUJACEM":"Basic Materials","MAXHEALTH":"Healthcare",
    "FORTIS":"Healthcare","GODREJCP":"Consumer Defensive","DABUR":"Consumer Defensive",
    "EMAMILTD":"Consumer Defensive","NHPC":"Utilities","NATIONALUM":"Basic Materials",
    "JINDALSTEL":"Basic Materials","APLAPOLLO":"Basic Materials","BIOCON":"Healthcare",
    "LAURUSLABS":"Healthcare","LODHA":"Real Estate","PRESTIGE":"Real Estate",
    "MFSL":"Financial Services","CHOLAFIN":"Financial Services","JIOFIN":"Financial Services",
    "INDIGO":"Consumer Cyclical","RECLTD":"Financial Services","PFC":"Financial Services",
    "IRFC":"Financial Services","ADANIGREEN":"Energy","JSWENERGY":"Utilities",
    "ZYDUSLIFE":"Healthcare","AUROPHARMA":"Healthcare","ALKEM":"Healthcare",
    "SYNGENE":"Healthcare","GRANULES":"Healthcare","CONCOR":"Industrials",
}
