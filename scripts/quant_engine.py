"""
scripts/quant_engine.py
-----------------------
Full institutional-grade quant pipeline:
  1. ARIMA forecast → drift (μ)
  2. GARCH volatility → σ
  3. Correlated Monte Carlo simulation (with Cholesky decomposition)
  4. Fama-French 3-Factor Model for drift estimation
  5. Market Regime Detection (HMM) for regime-aware signals
  6. Risk engine (VaR, CVaR, Sharpe, Max Drawdown)
  7. Signal engine (BUY / HOLD / RISKY)
  8. Mispricing indicator
  9. Earnings risk flag
  10. Sentiment score from news
  11. Options implied volatility
  12. Backtesting engine for signal accuracy tracking
"""

import numpy as np
import pandas as pd
import warnings
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from scipy.stats import t as student_t
from scipy.stats import norm
from config import (N_SIMULATIONS, HORIZON_DAYS, RISK_FREE_RATE, T_DOF,
                    BUY_PROB_THRESHOLD, RISKY_VAR_THRESHOLD, HOLD_SHARPE_MIN,
                    REGIME_LOOKBACK_DAYS, REGIME_MIN_DATA_DAYS, REGIME_BULL_THRESHOLD)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Global regime state (cached)
_regime_cache = {"regime": None, "bull_prob": 0.5, "drift_adj": 0.0, "last_update": None}


# ════════════════════════════════════════════
#  CORRELATED MONTE CARLO (Cholesky decomposition)
# ════════════════════════════════════════════

def correlated_monte_carlo(stocks_data: dict, nifty_returns: pd.Series, N: int = 5000, T: int = 252) -> dict:
    """
    Simulate all stocks together using their real correlation structure.
    Uses Cholesky decomposition on the return covariance matrix.
    
    Args:
        stocks_data: dict of {symbol: {'log_returns': pd.Series, 'mu_daily': float, 'sigma_daily': float, 'price': float}}
        nifty_returns: Nifty 50 returns for market correlation
        N: number of simulations
        T: time horizon in days
    
    Returns:
        dict of {symbol: np.ndarray} with simulated paths (T+1 x N)
    """
    symbols = list(stocks_data.keys())
    n = len(symbols)
    
    if n < 2:
        # Fallback to independent simulation for single stock
        sym = symbols[0]
        paths = run_monte_carlo(
            stocks_data[sym]['price'],
            stocks_data[sym]['mu_daily'],
            stocks_data[sym]['sigma_daily'],
            T=T, N=N
        )
        return {sym: paths}
    
    # Build return matrix (T_hist × n_stocks)
    min_len = min(len(stocks_data[s]['log_returns'].dropna()) for s in symbols)
    lookback = min(min_len, 504)  # Max 2 years of data
    
    returns_matrix = np.column_stack([
        stocks_data[s]['log_returns'].values[-lookback:]
        for s in symbols
    ])
    
    # Covariance matrix from historical returns
    cov_matrix = np.cov(returns_matrix.T)
    
    # Add small regularization for numerical stability
    cov_matrix = cov_matrix + 1e-8 * np.eye(n)
    
    try:
        # Cholesky decomposition — creates correlated shocks
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        logger.warning("  Cholesky failed, using independent simulation")
        # Fallback to independent
        results = {}
        for sym in symbols:
            results[sym] = run_monte_carlo(
                stocks_data[sym]['price'],
                stocks_data[sym]['mu_daily'],
                stocks_data[sym]['sigma_daily'],
                T=T, N=N
            )
        return results
    
    # Generate correlated random shocks
    rng = np.random.default_rng(42)
    Z_ind = rng.standard_normal((T, N, n))  # independent
    Z_cor = Z_ind @ L.T                      # correlated
    
    # Simulate correlated paths for each stock
    results = {}
    for i, sym in enumerate(symbols):
        mu    = stocks_data[sym]['mu_daily']
        sigma = stocks_data[sym]['sigma_daily']
        S0    = stocks_data[sym]['price']
        
        shocks = mu - 0.5 * sigma**2 + sigma * Z_cor[:, :, i]
        paths  = S0 * np.exp(np.cumsum(shocks, axis=0))
        results[sym] = paths
    
    logger.debug(f"    Correlated Monte Carlo: {n} stocks, {N} paths")
    return results


def compute_portfolio_var(stocks_data: dict, final_prices: dict, weights: dict = None) -> dict:
    """
    Compute portfolio-level VaR using correlated simulations.
    
    Args:
        stocks_data: dict of stock data
        final_prices: dict of {symbol: final_price_array}
        weights: dict of {symbol: weight}, default equal weight
    
    Returns:
        dict with portfolio VaR metrics
    """
    symbols = list(final_prices.keys())
    n = len(symbols)
    
    if weights is None:
        weights = {s: 1.0/n for s in symbols}
    
    # Calculate portfolio returns for each simulation
    n_sims = len(next(iter(final_prices.values())))
    portfolio_returns = np.zeros(n_sims)
    
    for sym in symbols:
        S0 = stocks_data[sym]['price']
        ret = (final_prices[sym] - S0) / S0
        portfolio_returns += weights.get(sym, 0) * ret
    
    # Portfolio VaR
    var_95 = float(-np.percentile(portfolio_returns, 5))
    var_99 = float(-np.percentile(portfolio_returns, 1))
    
    # CVaR
    tail_mask = portfolio_returns <= np.percentile(portfolio_returns, 5)
    cvar_95 = float(-portfolio_returns[tail_mask].mean())
    
    return {
        "portfolio_var_95": round(var_95, 4),
        "portfolio_var_99": round(var_99, 4),
        "portfolio_cvar_95": round(cvar_95, 4),
    }


# ════════════════════════════════════════════
#  FAMA-FRENCH 3-FACTOR MODEL
# ════════════════════════════════════════════

def fama_french_drift(
    log_returns: pd.Series,
    nifty_returns: pd.Series,
    smb_returns: pd.Series = None,
    hml_returns: pd.Series = None,
    beta_nifty: float = 1.0
) -> float:
    """
    μ = Rf + β_market×ERP + β_size×SMB + β_value×HML
    
    Computes expected daily return using Fama-French 3-factor model.
    If SMB/HML not provided, estimates from Nifty data.
    
    Args:
        log_returns: Stock log returns
        nifty_returns: Nifty 50 returns (market factor)
        smb_returns: Small Minus Big returns (size factor), optional
        hml_returns: High Minus Low returns (value factor), optional
        beta_nifty: Stock's beta vs Nifty
    
    Returns:
        Expected daily return (drift)
    """
    try:
        from sklearn.linear_model import LinearRegression
        
        # Use last 504 days (2 years) for regression
        n = min(len(log_returns), len(nifty_returns), 504)
        
        if smb_returns is None or hml_returns is None:
            # Estimate SMB/HML from Nifty data (simplified)
            # In production, would use Nifty 500 data
            smb_returns = nifty_returns * 0.3  # Approximate small cap premium
            hml_returns = nifty_returns * 0.2  # Approximate value premium
        
        # Align data
        df = pd.DataFrame({
            'stock': log_returns.values[-n:],
            'market': nifty_returns.values[-n:],
            'smb': smb_returns.values[-n:],
            'hml': hml_returns.values[-n:],
        }).dropna()
        
        if len(df) < 60:
            # Fallback to CAPM
            return float(log_returns.mean())
        
        X = df[['market', 'smb', 'hml']].values
        y = df['stock'].values
        
        reg = LinearRegression().fit(X, y)
        b_mkt, b_smb, b_hml = reg.coef_
        
        # Expected daily return components
        ERP = 0.065 / 252        # Equity risk premium daily (~6.5% annual)
        SMB_premium = 0.03 / 252  # Small cap premium ~3%/yr
        HML_premium = 0.04 / 252  # Value premium ~4%/yr
        
        # Fama-French expected return
        mu_daily = (0.065/252) + b_mkt * ERP + b_smb * SMB_premium + b_hml * HML_premium
        
        # Sanity check — clamp to reasonable range
        mu_daily = float(np.clip(mu_daily, -0.05/252, 0.25/252))
        
        logger.debug(f"    FF3 μ = {mu_daily:.6f} (β_mkt={b_mkt:.2f}, β_smb={b_smb:.2f}, β_hml={b_hml:.2f})")
        return mu_daily
        
    except Exception as e:
        logger.debug(f"    FF3 failed ({e}), using CAPM drift")
        # Fallback to CAPM
        ERP = 0.065 / 252
        return beta_nifty * ERP


# ════════════════════════════════════════════
#  MARKET REGIME DETECTION (Hidden Markov Model)
# ════════════════════════════════════════════

def detect_market_regime(nifty_returns: pd.Series) -> dict:
    """
    2-state HMM: Bull (high return, low vol) vs Bear (low return, high vol)
    Returns current regime probability and adjusts drift accordingly.
    
    Args:
        nifty_returns: Nifty 50 daily returns
    
    Returns:
        dict with regime, bull_prob, drift_adj
    """
    global _regime_cache
    
    # Check cache (valid for 1 hour)
    if _regime_cache["last_update"]:
        cache_age = datetime.now() - _regime_cache["last_update"]
        if cache_age < timedelta(hours=1):
            return {
                "regime": _regime_cache["regime"],
                "bull_prob": _regime_cache["bull_prob"],
                "drift_adj": _regime_cache["drift_adj"]
            }
    
    try:
        from hmmlearn import hmm
        
        # Use last 3 years of data
        returns = nifty_returns.values[-REGIME_LOOKBACK_DAYS:].reshape(-1, 1)
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < REGIME_MIN_DATA_DAYS:
            return {"regime": "Unknown", "bull_prob": 0.5, "drift_adj": 0.0}
        
        # Fit Gaussian HMM with 2 states
        model = hmm.GaussianHMM(
            n_components=2,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        model.fit(returns)
        
        # Get current state probability
        states = model.predict(returns)
        current_state = states[-1]
        
        # Identify which state is "bull" (higher mean)
        means = [model.means_[i][0] for i in range(2)]
        variances = [model.covars_[i][0][0] for i in range(2)]
        bull_state = np.argmax(means)
        
        is_bull = (current_state == bull_state)
        bull_prob = float(model.predict_proba(returns)[-1][bull_state])
        
        # Regime-based drift adjustment
        if is_bull:
            drift_adj = 0.02 / 252   # Bull regime: optimistic bias
        else:
            drift_adj = -0.03 / 252  # Bear regime: pessimistic
        
        # Update cache
        _regime_cache = {
            "regime": "Bull" if is_bull else "Bear",
            "bull_prob": bull_prob,
            "drift_adj": drift_adj,
            "last_update": datetime.now()
        }
        
        logger.info(f"    Regime: {'Bull' if is_bull else 'Bear'} (prob: {bull_prob:.1%})")
        
        return {
            "regime": "Bull" if is_bull else "Bear",
            "bull_prob": bull_prob,
            "drift_adj": drift_adj,
            "bull_mean": float(means[bull_state]),
            "bear_mean": float(means[1 - bull_state]),
            "bull_vol": float(np.sqrt(variances[bull_state])),
            "bear_vol": float(np.sqrt(variances[1 - bull_state])),
        }
        
    except Exception as e:
        logger.debug(f"    HMM failed ({e}), using neutral regime")
        return {"regime": "Unknown", "bull_prob": 0.5, "drift_adj": 0.0}


# ════════════════════════════════════════════
#  EARNINGS & CORPORATE ACTION RISK
# ════════════════════════════════════════════

def get_earnings_risk_flag(symbol: str, price: float = None) -> dict:
    """
    Flag stocks with earnings in next 30 days — widen confidence intervals.
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE')
        price: Current price (optional)
    
    Returns:
        dict with earnings risk info
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol + ".NS")
        
        # Get earnings calendar
        cal = ticker.calendar
        if cal is None or cal.empty:
            return {"has_earnings_soon": False, "vol_multiplier": 1.0}
        
        # Next earnings date
        next_earnings = pd.Timestamp(cal.columns[0])
        days_to_earn = (next_earnings - pd.Timestamp.today()).days
        
        if 0 < days_to_earn < 30:
            # Get historical earnings moves
            hist = ticker.earnings_history
            if hist is not None and not hist.empty:
                avg_move = hist['surprisePercent'].abs().mean() / 100
            else:
                avg_move = 0.07  # Default 7% move
            
            return {
                "has_earnings_soon": True,
                "days_to_earnings": days_to_earn,
                "vol_multiplier": 1 + avg_move,
                "avg_earnings_move": round(avg_move * 100, 1)
            }
        
        return {"has_earnings_soon": False, "vol_multiplier": 1.0}
        
    except Exception as e:
        logger.debug(f"    Earnings check failed for {symbol}: {e}")
        return {"has_earnings_soon": False, "vol_multiplier": 1.0}


# ════════════════════════════════════════════
#  SENTIMENT SCORE FROM NEWS (Google News + VADER)
# ════════════════════════════════════════════

def get_sentiment_score(symbol: str, company_name: str = None) -> dict:
    """
    Use Google News RSS (free) + VADER sentiment analysis.
    
    Args:
        symbol: Stock symbol
        company_name: Company name for news search
    
    Returns:
        dict with sentiment score and label
    """
    if company_name is None:
        company_name = symbol
    
    try:
        import feedparser
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        # Google News RSS query
        query = company_name.replace(' ', '+')
        rss_url = f"https://news.google.com/rss/search?q={query}+stock&hl=en-IN&gl=IN&ceid=IN:en"
        
        feed = feedparser.parse(rss_url)
        
        if not feed.entries:
            return {"sentiment_score": 0.0, "sentiment_label": "Neutral", "news_count": 0}
        
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        
        for entry in feed.entries[:10]:  # Last 10 news items
            text = entry.title + ' ' + entry.get('summary', '')
            score = analyzer.polarity_scores(text)['compound']
            scores.append(score)
        
        avg_sentiment = np.mean(scores) if scores else 0
        
        return {
            "sentiment_score": round(avg_sentiment, 3),
            "sentiment_label": "Positive" if avg_sentiment > 0.1 else 
                              "Negative" if avg_sentiment < -0.1 else "Neutral",
            "news_count": len(scores)
        }
        
    except Exception as e:
        logger.debug(f"    Sentiment check failed for {symbol}: {e}")
        return {"sentiment_score": 0.0, "sentiment_label": "Neutral", "news_count": 0}


# ════════════════════════════════════════════
#  OPTIONS IMPLIED VOLATILITY (F&O stocks)
# ════════════════════════════════════════════

def get_implied_volatility(symbol: str) -> float:
    """
    For F&O stocks, use actual implied volatility from options.
    Falls back to GARCH if unavailable.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Annualized IV or None if unavailable
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol + ".NS")
        
        if not ticker.options:
            return None  # No options chain available
        
        # Get nearest expiry options chain
        expiry = ticker.options[0]
        chain = ticker.option_chain(expiry)
        spot = ticker.fast_info.last_price
        
        # Find ATM call
        calls = chain.calls
        atm_idx = (calls['strike'] - spot).abs().idxmin()
        iv = calls.loc[atm_idx, 'impliedVolatility']
        
        # Sanity check
        if iv and 0.05 < iv < 2.0:
            logger.debug(f"    IV: {iv:.2%} (ATM)")
            return float(iv)
        
        return None
        
    except Exception as e:
        logger.debug(f"    IV check failed for {symbol}: {e}")
        return None


# ════════════════════════════════════════════
#  BACKTESTING ENGINE
# ════════════════════════════════════════════

SIGNALS_HISTORY_FILE = Path("docs/signal_history.json")

def save_signal_for_backtest(symbol: str, signal: str, price: float, date: str = None) -> None:
    """
    Save signal to history for backtesting.
    
    Args:
        symbol: Stock symbol
        signal: Signal string (STRONG BUY, BUY, HOLD, RISKY, AVOID)
        price: Price at signal time
        date: Date string (default: today)
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Load existing history
    history = []
    if SIGNALS_HISTORY_FILE.exists():
        try:
            with open(SIGNALS_HISTORY_FILE) as f:
                history = json.load(f)
        except:
            history = []
    
    # Add new signal
    history.append({
        "date": date,
        "symbol": symbol,
        "signal": signal,
        "price_at_signal": round(price, 2)
    })
    
    # Keep last 2 years of history
    cutoff = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    history = [h for h in history if h["date"] >= cutoff]
    
    # Save
    with open(SIGNALS_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def backtest_signals(price_data: dict, holding_days: int = 30) -> dict:
    """
    For each past signal, measure actual return over next N days.
    
    Args:
        price_data: dict of {symbol: pd.Series with prices}
        holding_days: Days to hold before measuring return
    
    Returns:
        dict with accuracy stats per signal type
    """
    if not SIGNALS_HISTORY_FILE.exists():
        return {"error": "No signal history found"}
    
    try:
        with open(SIGNALS_HISTORY_FILE) as f:
            history = json.load(f)
    except:
        return {"error": "Could not load signal history"}
    
    results = []
    
    for record in history:
        sym = record["symbol"]
        date = record["date"]
        sig = record["signal"]
        p_buy = record["price_at_signal"]
        
        if sym not in price_data:
            continue
        
        # Get price N days later
        prices = price_data[sym]
        try:
            future_prices = prices.loc[date:]
            if len(future_prices) > holding_days:
                p_future = future_prices.iloc[holding_days]
                actual_return = (p_future - p_buy) / p_buy * 100
                
                # Determine if signal was "correct"
                is_buy_signal = "BUY" in sig
                is_correct = (is_buy_signal and actual_return > 0) or \
                            (sig == "AVOID" and actual_return < 0)
                
                results.append({
                    "signal": sig,
                    "actual_return": round(actual_return, 2),
                    "correct": is_correct,
                    "days": holding_days
                })
        except:
            continue
    
    if not results:
        return {"error": "Insufficient data for backtest"}
    
    # Aggregate stats per signal type
    df = pd.DataFrame(results)
    stats = df.groupby("signal").agg({
        "actual_return": ["mean", "std", "count"],
        "correct": "mean"
    }).round(3)
    
    # Convert to dict
    stats_dict = {}
    for sig in stats.index:
        stats_dict[sig] = {
            "avg_return": float(stats.loc[sig, ("actual_return", "mean")]),
            "std_return": float(stats.loc[sig, ("actual_return", "std")]),
            "count": int(stats.loc[sig, ("actual_return", "count")]),
            "accuracy": float(stats.loc[sig, ("correct", "mean")])
        }
    
    return stats_dict


# ════════════════════════════════════════════
#  ARIMA FORECAST (μ estimation)
# ════════════════════════════════════════════


# ════════════════════════════════════════════
#  ARIMA FORECAST (μ estimation)
# ════════════════════════════════════════════

def arima_drift(log_returns: pd.Series) -> float:
    """
    Uses ARIMA(1,0,1) to forecast expected daily return.
    Falls back to historical mean if ARIMA fails.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        # Use last 252 days for speed
        series = log_returns.iloc[-252:].dropna()
        if len(series) < 60:
            return float(log_returns.mean())

        model  = ARIMA(series, order=(1, 0, 1))
        result = model.fit()
        # 1-step ahead forecast
        forecast = result.forecast(steps=1)
        mu = float(forecast.iloc[0])
        # Sanity check — clamp extreme values
        mu = np.clip(mu, -0.01, 0.01)
        logger.debug(f"    ARIMA μ = {mu:.6f}")
        return mu
    except Exception as e:
        logger.debug(f"    ARIMA failed ({e}), using historical mean")
        return float(log_returns.mean())


# ════════════════════════════════════════════
#  GARCH VOLATILITY (σ estimation)
# ════════════════════════════════════════════

def garch_volatility(log_returns: pd.Series) -> float:
    """
    Uses GARCH(1,1) to estimate conditional volatility.
    Falls back to EWMA if GARCH fails.
    """
    try:
        from arch import arch_model
        series  = log_returns.iloc[-504:].dropna() * 100  # Scale for GARCH
        if len(series) < 100:
            return float(log_returns.ewm(span=63).std().iloc[-1])

        model   = arch_model(series, vol="Garch", p=1, q=1,
                             dist="StudentsT", rescale=False)
        result  = model.fit(disp="off", show_warning=False)
        # Forecast 1-step ahead variance
        forecast    = result.forecast(horizon=1)
        cond_var    = forecast.variance.values[-1, 0]
        sigma_daily = float(np.sqrt(cond_var) / 100)  # Rescale back
        sigma_daily = np.clip(sigma_daily, 0.005, 0.06)
        logger.debug(f"    GARCH σ = {sigma_daily:.6f}")
        return sigma_daily
    except Exception as e:
        logger.debug(f"    GARCH failed ({e}), using EWMA")
        return float(log_returns.ewm(span=63).std().iloc[-1])


# ════════════════════════════════════════════
#  MONTE CARLO SIMULATION (GBM)
# ════════════════════════════════════════════

def run_monte_carlo(
    S0    : float,
    mu    : float,
    sigma : float,
    T     : int   = HORIZON_DAYS,
    N     : int   = N_SIMULATIONS,
    seed  : int   = 42
) -> np.ndarray:
    """
    Vectorised GBM Monte Carlo simulation with Student-t shocks.
    Returns terminal prices array of shape (N,).
    Also returns path percentiles for charting.
    """
    rng = np.random.default_rng(seed)

    # Student-t shocks (fat tails — more realistic)
    Z     = student_t.rvs(df=T_DOF, size=(T, N), random_state=seed)
    scale = np.sqrt((T_DOF - 2) / T_DOF)
    Z     = Z * scale

    drift  = (mu - 0.5 * sigma**2)
    paths  = S0 * np.exp(np.cumsum(drift + sigma * Z, axis=0))

    return paths


def extract_path_percentiles(paths: np.ndarray, sample_every: int = 5) -> dict:
    """Extract percentile paths for frontend charting (sampled for size)."""
    idx = range(0, paths.shape[0], sample_every)
    return {
        "p5"  : np.percentile(paths[idx], 5,  axis=1).round(2).tolist(),
        "p25" : np.percentile(paths[idx], 25, axis=1).round(2).tolist(),
        "p50" : np.percentile(paths[idx], 50, axis=1).round(2).tolist(),
        "p75" : np.percentile(paths[idx], 75, axis=1).round(2).tolist(),
        "p95" : np.percentile(paths[idx], 95, axis=1).round(2).tolist(),
        "days": list(range(sample_every, paths.shape[0]+1, sample_every)),
    }


def extract_histogram(final_prices: np.ndarray, bins: int = 50) -> dict:
    """Extract histogram data for distribution chart."""
    counts, edges = np.histogram(final_prices, bins=bins)
    centers = ((edges[:-1] + edges[1:]) / 2).round(2).tolist()
    return {"labels": centers, "counts": counts.tolist()}


# ════════════════════════════════════════════
#  RISK ENGINE
# ════════════════════════════════════════════

def compute_risk_metrics(
    final_prices : np.ndarray,
    S0           : float,
    mu_daily     : float,
    sigma_daily  : float,
    log_returns  : pd.Series,
) -> dict:
    """
    Computes institutional-grade risk metrics.
    VaR, CVaR, Sharpe, Sortino, Max Drawdown, Calmar.
    """
    pnl_pct  = (final_prices - S0) / S0

    # VaR (% loss not exceeded at confidence level)
    var_95   = float(-np.percentile(pnl_pct, 5))
    var_99   = float(-np.percentile(pnl_pct, 1))

    # CVaR / Expected Shortfall (avg loss beyond VaR)
    tail_mask = pnl_pct <= np.percentile(pnl_pct, 5)
    cvar_95   = float(-pnl_pct[tail_mask].mean())

    # Annual metrics
    mu_annual    = float(mu_daily * 252)
    sigma_annual = float(sigma_daily * np.sqrt(252))

    # Sharpe Ratio
    excess_return = mu_annual - RISK_FREE_RATE
    sharpe        = excess_return / sigma_annual if sigma_annual > 0 else 0.0

    # Sortino Ratio (downside deviation only)
    neg_returns    = log_returns[log_returns < 0]
    downside_std   = float(neg_returns.std() * np.sqrt(252)) if len(neg_returns) > 10 else sigma_annual
    sortino        = excess_return / downside_std if downside_std > 0 else 0.0

    # Max Drawdown from historical data
    close        = log_returns.cumsum().apply(np.exp)
    rolling_max  = close.cummax()
    drawdown     = (close - rolling_max) / rolling_max
    max_dd       = float(drawdown.min())

    # Calmar Ratio
    calmar = mu_annual / abs(max_dd) if max_dd != 0 else 0.0

    # Z-score (how far current price is from expected)
    expected_price = float(np.mean(final_prices))
    price_zscore   = (expected_price - S0) / (S0 * sigma_annual) if sigma_annual > 0 else 0.0

    return {
        "var_95"       : round(var_95, 4),
        "var_99"       : round(var_99, 4),
        "cvar_95"      : round(cvar_95, 4),
        "sharpe"       : round(sharpe, 3),
        "sortino"      : round(sortino, 3),
        "calmar"       : round(calmar, 3),
        "max_drawdown" : round(max_dd, 4),
        "mu_annual"    : round(mu_annual, 4),
        "sigma_annual" : round(sigma_annual, 4),
        "price_zscore" : round(price_zscore, 3),
    }


# ════════════════════════════════════════════
#  SIGNAL ENGINE
# ════════════════════════════════════════════

def generate_signal(
    prob_up      : float,
    var_95       : float,
    sharpe       : float,
    expected_ret : float,
    mispricing   : float,
) -> dict:
    """
    Generates BUY / HOLD / RISKY / SELL signal with confidence score.
    Mispricing: positive = undervalued, negative = overvalued.
    """
    score = 0  # -10 to +10

    # Probability of gain
    if prob_up > 0.70:   score += 3
    elif prob_up > 0.60: score += 2
    elif prob_up > 0.50: score += 1
    elif prob_up < 0.40: score -= 3
    elif prob_up < 0.50: score -= 1

    # Sharpe ratio
    if sharpe > 1.0:   score += 2
    elif sharpe > 0.5: score += 1
    elif sharpe < 0.0: score -= 2
    elif sharpe < 0.3: score -= 1

    # VaR
    if var_95 < 0.10:   score += 2
    elif var_95 < 0.15: score += 1
    elif var_95 > 0.25: score -= 2
    elif var_95 > 0.20: score -= 1

    # Expected return
    if expected_ret > 0.15:  score += 2
    elif expected_ret > 0.08: score += 1
    elif expected_ret < 0:    score -= 2
    elif expected_ret < 0.04: score -= 1

    # Mispricing (undervalued = positive signal)
    if mispricing > 0.10:   score += 1
    elif mispricing < -0.10: score -= 1

    # Convert score to signal
    if score >= 5:
        signal = "STRONG BUY"
        color  = "#00c853"
    elif score >= 2:
        signal = "BUY"
        color  = "#2ecc8a"
    elif score >= -1:
        signal = "HOLD"
        color  = "#f0b429"
    elif score >= -4:
        signal = "RISKY"
        color  = "#ff9800"
    else:
        signal = "AVOID"
        color  = "#e02d3c"

    # Confidence = normalised score
    confidence = min(100, max(0, int((score + 10) / 20 * 100)))

    return {
        "signal"    : signal,
        "signal_color": color,
        "score"     : score,
        "confidence": confidence,
    }


# ════════════════════════════════════════════
#  FULL PIPELINE FOR ONE STOCK
# ════════════════════════════════════════════

def run_full_pipeline(
    symbol   : str,
    features : dict,
    fin_info : dict,
    nifty_returns : pd.Series = None,
) -> dict | None:
    """
    Runs complete quant pipeline for one stock with all advanced features.
    Returns full metrics dict ready for JSON output.
    
    Features integrated:
    - Fama-French 3-Factor Model for drift
    - Market Regime Detection (HMM)
    - Earnings Risk Flag
    - Sentiment Score from News
    - Options Implied Volatility
    - Backtesting signal storage
    """
    try:
        log_returns   = features["log_returns"]
        S0            = features["current_price"]
        sigma_ewma    = features["sigma_daily"]
        beta          = fin_info.get("beta", 1.0)

        if len(log_returns) < 100:
            logger.warning(f"  {symbol}: not enough returns ({len(log_returns)})")
            return None

        # ── 0. Market Regime Detection ──────────
        regime_info = {"regime": "Unknown", "bull_prob": 0.5, "drift_adj": 0.0}
        if nifty_returns is not None and len(nifty_returns) > 252:
            logger.info(f"    Regime Detection...")
            regime_info = detect_market_regime(nifty_returns)

        # ── 1. Fama-French 3-Factor drift ───────
        logger.info(f"    Fama-French 3-Factor...")
        if nifty_returns is not None and len(nifty_returns) > 100:
            mu = fama_french_drift(log_returns, nifty_returns, beta_nifty=beta)
        else:
            mu = arima_drift(log_returns)
        
        # Apply regime adjustment
        mu = mu + regime_info["drift_adj"]

        # ── 2. GARCH volatility ─────────────────
        logger.info(f"    GARCH...")
        sigma = garch_volatility(log_returns)
        
        # Try to get IV from options for F&O stocks
        iv = get_implied_volatility(symbol)
        if iv is not None:
            logger.info(f"    Using IV from options: {iv:.2%}")
            sigma = iv / np.sqrt(252)  # Convert annual IV to daily
        
        # Check earnings risk
        earnings_risk = get_earnings_risk_flag(symbol, S0)
        if earnings_risk["has_earnings_soon"]:
            logger.info(f"    Earnings in {earnings_risk['days_to_earnings']} days - widening CI")
            sigma = sigma * earnings_risk["vol_multiplier"]
        
        # Blend GARCH with EWMA for stability
        sigma = 0.6 * sigma + 0.4 * sigma_ewma

        # ── 3. Monte Carlo ──────────────────────
        logger.info(f"    Monte Carlo ({N_SIMULATIONS:,} paths)...")
        paths       = run_monte_carlo(S0, mu, sigma)
        final       = paths[-1, :]
        path_charts = extract_path_percentiles(paths)
        histogram   = extract_histogram(final)

        # ── 4. Risk metrics ─────────────────────
        risk = compute_risk_metrics(final, S0, mu, sigma, log_returns)

        # ── 5. Distribution stats ───────────────
        mean_price   = float(np.mean(final))
        median_price = float(np.median(final))
        ci_5         = float(np.percentile(final, 5))
        ci_25        = float(np.percentile(final, 25))
        ci_75        = float(np.percentile(final, 75))
        ci_95        = float(np.percentile(final, 95))

        prob_up       = float(np.mean(final > S0))
        prob_10up     = float(np.mean(final > S0 * 1.10))
        prob_20up     = float(np.mean(final > S0 * 1.20))
        prob_10down   = float(np.mean(final < S0 * 0.90))

        expected_ret  = (mean_price / S0) - 1

        # ── 6. Mispricing indicator ─────────────
        mispricing = (mean_price - S0) / S0   # + = undervalued

        # ── 7. Sentiment Score ───────────────────
        logger.info(f"    Sentiment Analysis...")
        sentiment = get_sentiment_score(symbol, fin_info.get("name"))
        
        # Adjust mispricing based on sentiment
        sentiment_adj = sentiment["sentiment_score"] * 0.05  # Max 5% adjustment
        mispricing = mispricing + sentiment_adj

        # ── 8. Scenario medians ─────────────────
        def scen(mu_m, sig_m):
            p = run_monte_carlo(S0, mu * mu_m, sigma * sig_m, N=1000)
            return float(np.median(p[-1, :]))

        # ── 9. Signal (with sentiment adjustment) ─
        sig_result = generate_signal(
            prob_up, risk["var_95"],
            risk["sharpe"], expected_ret, mispricing
        )

        # ── 10. Save signal for backtesting ─────
        save_signal_for_backtest(symbol, sig_result["signal"], S0)

        return {
            # Identity
            "symbol"          : symbol,
            "name"            : fin_info.get("name", symbol),
            "sector"          : fin_info.get("sector", features.get("sector", "Other")),
            "industry"        : fin_info.get("industry", ""),
            "description"     : fin_info.get("description", ""),
            "website"         : fin_info.get("website", ""),

            # Price
            "price"           : round(S0, 2),
            "week52_high"     : round(features["week52_high"], 2),
            "week52_low"      : round(features["week52_low"], 2),

            # Financial info
            "market_cap"      : fin_info.get("market_cap"),
            "pe_ratio"        : fin_info.get("pe_ratio"),
            "pb_ratio"        : fin_info.get("pb_ratio"),
            "eps"             : fin_info.get("eps"),
            "revenue"         : fin_info.get("revenue"),
            "net_income"      : fin_info.get("net_income"),
            "roe"             : fin_info.get("roe"),
            "roa"             : fin_info.get("roa"),
            "debt_equity"     : fin_info.get("debt_equity"),
            "current_ratio"   : fin_info.get("current_ratio"),
            "dividend_yield"  : fin_info.get("dividend_yield"),
            "beta"            : fin_info.get("beta"),
            "book_value"      : fin_info.get("book_value"),
            "employees"       : fin_info.get("employees"),

            # Monte Carlo outputs
            "mean_price"      : round(mean_price, 2),
            "median_price"    : round(median_price, 2),
            "ci_5"            : round(ci_5, 2),
            "ci_25"           : round(ci_25, 2),
            "ci_75"           : round(ci_75, 2),
            "ci_95"           : round(ci_95, 2),
            "expected_return" : round(expected_ret, 4),
            "expected_return_pct": round(expected_ret * 100, 2),
            "mispricing_pct"  : round(mispricing * 100, 2),

            # Probabilities
            "prob_up"         : round(prob_up, 4),
            "prob_10up"       : round(prob_10up, 4),
            "prob_20up"       : round(prob_20up, 4),
            "prob_10down"     : round(prob_10down, 4),

            # Risk metrics
            **risk,

            # Signal
            **sig_result,

            # Model parameters
            "mu_arima"        : round(mu, 6),
            "sigma_garch"     : round(sigma, 6),
            "mu_daily"        : round(features["mu_daily"], 6),
            "mu_fama_french"  : round(mu - regime_info["drift_adj"], 6) if "drift_adj" in regime_info else round(mu, 6),

            # Momentum
            "mom_1m"          : round(features["mom_1m"] * 100, 2),
            "mom_3m"          : round(features["mom_3m"] * 100, 2),
            "mom_6m"          : round(features["mom_6m"] * 100, 2),
            "mom_1y"          : round(features["mom_1y"] * 100, 2),

            # Chart data
            "path_charts"     : path_charts,
            "histogram"       : histogram,

            # Scenarios
            "bull_median"     : round(scen(1.3, 0.8), 2),
            "base_median"     : round(scen(1.0, 1.0), 2),
            "bear_median"     : round(scen(0.7, 1.4), 2),

            # Advanced features
            "regime"          : regime_info.get("regime", "Unknown"),
            "bull_prob"       : regime_info.get("bull_prob", 0.5),
            "regime_drift_adj": regime_info.get("drift_adj", 0.0),
            
            "earnings_risk"   : earnings_risk.get("has_earnings_soon", False),
            "days_to_earnings": earnings_risk.get("days_to_earnings", None),
            "vol_multiplier"  : earnings_risk.get("vol_multiplier", 1.0),
            
            "sentiment_score" : sentiment.get("sentiment_score", 0.0),
            "sentiment_label" : sentiment.get("sentiment_label", "Neutral"),
            "news_count"      : sentiment.get("news_count", 0),
            
            "implied_vol"     : round(iv, 4) if iv else None,

            # Meta
            "n_simulations"   : N_SIMULATIONS,
            "horizon_days"    : HORIZON_DAYS,
            "model"           : "ARIMA-GARCH-GBM-StudentT-FF3-HMM",
        }

    except Exception as e:
        logger.error(f"  {symbol}: pipeline failed — {e}")
        import traceback; traceback.print_exc()
        return None
