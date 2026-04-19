"""
scripts/config.py
-----------------
Central configuration for QuantEdge Analytics platform.
Stock universe auto-updates from NSE.
"""

# ── Simulation Parameters ─────────────────────
N_SIMULATIONS    = 10_000
HORIZON_DAYS     = 252
ROLLING_VOL_WIN  = 63
RISK_FREE_RATE   = 0.070   # RBI repo rate annualised (April 2026)
T_DOF            = 5       # Student-t degrees of freedom
TOP_N_STOCKS     = 100     # Max stocks for heavy quant computation

# ── Signal Thresholds ─────────────────────────
BUY_PROB_THRESHOLD    = 0.60   # P(gain) > 60% → BUY
RISKY_VAR_THRESHOLD   = 0.20   # VaR > 20% → RISKY
HOLD_SHARPE_MIN       = 0.3    # Sharpe > 0.3 → HOLD eligible

# ── Regime Detection ─────────────────────────
REGIME_LOOKBACK_DAYS   = 756   # HMM training window (~3 years)
REGIME_MIN_DATA_DAYS   = 252   # Minimum data required
REGIME_BULL_THRESHOLD  = 0.60  # Bull probability > 60% = Bull regime

# ── Extended Stock Universe (NSE) ─────────────
# This is the full universe shown in the screener
# Quant engine only processes top 100 by momentum/volume

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
    "MUTHOOTFIN","LUPIN","AMBUJACEM","GLAND","TRENT",
    "LODHA","NYKAA","ZOMATO","PAYTM","POLICYBZR",
    "IRCTC","CONCOR","OBEROIRLTY","PRESTIGE","SOBHA",
    "CANBK","BANKBARODA","FEDERALBNK","IDFCFIRSTB","BANDHANBNK",
    "MFSL","MAXHEALTH","FORTIS","LALPATHLAB","METROPOLIS",
    "GODREJCP","DABUR","EMAMILTD","TATAPOWER","NHPC",
    "SAIL","NATIONALUM","VEDL","JINDALSTEL","APLAPOLLO",
    "DMART","ZYDUSLIFE","AUROPHARMA","ALKEM","IPCALAB",
    "BIOCON","SYNGENE","LAURUSLABS","GRANULES","SUNPHARMA"
]

# Full universe for screener (frontend only - no compute cost)
FULL_UNIVERSE = list(set(NIFTY50_SYMBOLS + NIFTY_NEXT50))

SECTOR_MAP = {
    "RELIANCE":"Energy","TCS":"IT","HDFCBANK":"Banking","BHARTIARTL":"Telecom",
    "ICICIBANK":"Banking","INFOSYS":"IT","SBIN":"Banking","HINDUNILVR":"FMCG",
    "ITC":"FMCG","LT":"Infra","KOTAKBANK":"Banking","AXISBANK":"Banking",
    "ASIANPAINT":"Consumer","MARUTI":"Auto","SUNPHARMA":"Pharma","TITAN":"Consumer",
    "ULTRACEMCO":"Cement","BAJFINANCE":"Finance","WIPRO":"IT","ONGC":"Energy",
    "NTPC":"Power","POWERGRID":"Power","HCLTECH":"IT","M&M":"Auto",
    "TATAMOTORS":"Auto","TATASTEEL":"Metal","ADANIENT":"Conglomerate",
    "ADANIPORTS":"Infra","COALINDIA":"Mining","BAJAJFINSV":"Finance",
    "NESTLEIND":"FMCG","CIPLA":"Pharma","DRREDDY":"Pharma","HEROMOTOCO":"Auto",
    "BRITANNIA":"FMCG","TECHM":"IT","GRASIM":"Cement","BPCL":"Energy",
    "EICHERMOT":"Auto","DIVISLAB":"Pharma","APOLLOHOSP":"Healthcare",
    "TATACONSUM":"FMCG","BAJAJ-AUTO":"Auto","HINDALCO":"Metal","JSWSTEEL":"Metal",
    "SHRIRAMFIN":"Finance","INDUSINDBK":"Banking","SBILIFE":"Insurance",
    "HDFCLIFE":"Insurance","BEL":"Defence","ZOMATO":"Tech","PAYTM":"Fintech",
    "NYKAA":"Retail","IRCTC":"Travel","DMART":"Retail","TRENT":"Retail",
    "TATAPOWER":"Power","VEDL":"Metal","SAIL":"Metal","BANKBARODA":"Banking",
    "CANBK":"Banking","FEDERALBNK":"Banking","IDFCFIRSTB":"Banking",
    "BANDHANBNK":"Banking","PIDILITIND":"Consumer","HAVELLS":"Consumer",
    "SIEMENS":"Industrial","ABB":"Industrial","TORNTPHARM":"Pharma",
    "MUTHOOTFIN":"Finance","LUPIN":"Pharma","AMBUJACEM":"Cement",
}


