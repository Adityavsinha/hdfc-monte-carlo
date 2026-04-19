"""
scripts/quant_engine.py - Fixed with Nifty-50 relative beta
"""
import numpy as np
import pandas as pd
import warnings
import logging
from scipy.stats import t as student_t
from config import (N_SIMULATIONS, HORIZON_DAYS, RISK_FREE_RATE, T_DOF,
                    BUY_PROB_THRESHOLD, RISKY_VAR_THRESHOLD, HOLD_SHARPE_MIN,
                    REGIME_LOOKBACK_DAYS, REGIME_MIN_DATA_DAYS, REGIME_BULL_THRESHOLD)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

MARKET_RETURN_ANNUAL = 0.13
EQUITY_PREMIUM       = MARKET_RETURN_ANNUAL - RISK_FREE_RATE
MIN_MU_ANNUAL        = -0.05
MAX_MU_ANNUAL        = +0.22

# Fama-French factor premiums (annual)
SMB_PREMIUM_ANNUAL   = 0.03   # Small cap premium ~3%/yr
HML_PREMIUM_ANNUAL   = 0.04   # Value premium ~4%/yr


def compute_nifty_beta(stock_returns, nifty_returns):
    try:
        df = pd.DataFrame({"s": stock_returns, "n": nifty_returns}).dropna()
        if len(df) < 60:
            return 1.0
        df = df.iloc[-504:]
        cov = df.cov()
        beta = cov.loc["s","n"] / cov.loc["n","n"]
        return float(np.clip(beta, 0.3, 2.2))
    except:
        return 1.0


def estimate_drift(log_returns, nifty_returns=None):
    beta = compute_nifty_beta(log_returns, nifty_returns) if nifty_returns is not None else 1.0
    mu_annual = RISK_FREE_RATE + beta * EQUITY_PREMIUM
    # Small momentum overlay
    if len(log_returns) >= 252:
        mom3  = log_returns.iloc[-63:].mean()
        mom12 = log_returns.iloc[-252:].mean()
        adj   = float(np.clip((mom3 - mom12) * 0.3, -0.0002, 0.0002))
    else:
        adj = 0.0
    mu_daily  = mu_annual / 252 + adj
    mu_annual = mu_daily * 252
    mu_annual = float(np.clip(mu_annual, MIN_MU_ANNUAL, MAX_MU_ANNUAL))
    mu_daily  = mu_annual / 252
    method    = f"CAPM(β={beta:.2f})+Mom → {mu_annual:.2%}/yr"
    logger.info(f"    β={beta:.2f} μ={mu_annual:.2%}/yr")
    return float(mu_daily), float(beta), method


def garch_volatility(log_returns, ewma_sigma):
    try:
        from arch import arch_model
        series = log_returns.iloc[-504:].dropna() * 100
        if len(series) < 100:
            return ewma_sigma
        model  = arch_model(series, vol="Garch", p=1, q=1, dist="StudentsT", rescale=False)
        result = model.fit(disp="off", show_warning=False)
        fc     = result.forecast(horizon=1)
        sg     = float(np.sqrt(fc.variance.values[-1, 0]) / 100)
        if sg < 0.004 or sg > 0.07:
            return ewma_sigma
        return float(0.6 * sg + 0.4 * ewma_sigma)
    except:
        return ewma_sigma


def detect_market_regime(nifty_returns: pd.Series) -> dict:
    """
    Detect market regime using Hidden Markov Model (HMM) or fallback method.
    
    Args:
        nifty_returns: Pandas Series of Nifty-50 daily log returns
        
    Returns:
        dict with regime (Bull/Bear), bull_prob, and drift_adjustment
    """
    try:
        from hmmlearn.hmm import GaussianHMM
        
        # Use last 756 days (approximately 3 years of trading days)
        returns = nifty_returns.dropna().iloc[-REGIME_LOOKBACK_DAYS:].values.reshape(-1, 1)
        
        if len(returns) < REGIME_MIN_DATA_DAYS:
            logger.warning("  Regime detection: insufficient data, using neutral")
            return {"regime": "Neutral", "bull_prob": 0.5, "drift_adjustment": 0.0}
        
        # Fit Gaussian HMM with 2 states
        model = GaussianHMM(
            n_components=2,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        model.fit(returns)
        
        # Get hidden states and predict probabilities
        hidden_states = model.predict(returns)
        proba = model.predict_proba(returns)
        
        # Calculate mean return for each state to identify Bull vs Bear
        state_means = [returns[hidden_states == i].mean() for i in range(2)]
        
        # State with higher mean return is Bull (state 1), lower is Bear (state 0)
        if state_means[1] > state_means[0]:
            bull_state = 1
            bear_state = 0
        else:
            bull_state = 0
            bear_state = 1
        
        # Current regime based on last observation
        current_state = hidden_states[-1]
        
        # Probability of being in bull regime (average of recent window)
        bull_prob = float(np.mean(proba[-21:, bull_state]))  # Last ~1 month
        
        # Determine regime based on probability threshold
        regime = "Bull" if bull_prob > REGIME_BULL_THRESHOLD else "Bear" if bull_prob < (1 - REGIME_BULL_THRESHOLD) else "Neutral"
        
        # Drift adjustment: positive for Bull, negative for Bear, neutral for Neutral
        if regime == "Bull":
            drift_adjustment = float(np.clip(state_means[bull_state] * 0.5, 0.0001, 0.0003))
        elif regime == "Bear":
            drift_adjustment = float(np.clip(state_means[bear_state] * 0.5, -0.0003, -0.0001))
        else:
            drift_adjustment = 0.0
        
        logger.info(f"  Regime: {regime} (bull_prob={bull_prob:.2%}, drift_adj={drift_adjustment:.6f})")
        
        return {
            "regime": regime,
            "bull_prob": round(bull_prob, 4),
            "drift_adjustment": round(drift_adjustment, 6),
        }
        
    except ImportError:
        # Fallback: Use simple momentum-based regime detection
        return _detect_regime_fallback(nifty_returns)
    except Exception as e:
        logger.warning(f"  Regime detection failed: {e}, using neutral")
        return {"regime": "Neutral", "bull_prob": 0.5, "drift_adjustment": 0.0}


def _detect_regime_fallback(nifty_returns: pd.Series) -> dict:
    """
    Fallback regime detection using moving average crossover and volatility.
    Used when hmmlearn is not available.
    """
    try:
        returns = nifty_returns.dropna().iloc[-252:]
        
        if len(returns) < 60:
            return {"regime": "Neutral", "bull_prob": 0.5, "drift_adjustment": 0.0}
        
        # Calculate metrics
        sma_short = returns.rolling(20).mean().iloc[-1]
        sma_long = returns.rolling(60).mean().iloc[-1]
        recent_mean = returns.iloc[-21:].mean()
        volatility = returns.iloc[-60:].std()
        
        # Determine regime based on trend and momentum
        trend_score = 0
        
        # SMA crossover signal
        if sma_short > sma_long:
            trend_score += 1
        else:
            trend_score -= 1
        
        # Recent momentum
        if recent_mean > 0:
            trend_score += 1
        else:
            trend_score -= 1
        
        # Volatility regime (low vol = bull, high vol = bear)
        avg_vol = returns.rolling(60).std().mean()
        if volatility < avg_vol * 0.8:
            trend_score += 1
        elif volatility > avg_vol * 1.2:
            trend_score -= 1
        
        # Determine regime
        if trend_score >= 2:
            regime = "Bull"
            bull_prob = 0.75
            drift_adjustment = 0.0002
        elif trend_score <= -2:
            regime = "Bear"
            bull_prob = 0.25
            drift_adjustment = -0.0002
        else:
            regime = "Neutral"
            bull_prob = 0.5
            drift_adjustment = 0.0
        
        logger.info(f"  Regime (fallback): {regime} (bull_prob={bull_prob:.2%}, drift_adj={drift_adjustment:.6f})")
        
        return {
            "regime": regime,
            "bull_prob": round(bull_prob, 4),
            "drift_adjustment": round(drift_adjustment, 6),
        }
        
    except Exception as e:
        logger.warning(f"  Fallback regime detection failed: {e}")
        return {"regime": "Neutral", "bull_prob": 0.5, "drift_adjustment": 0.0}


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
    Returns price paths array of shape (T+1, N).
    """
    # Student-t shocks (fat tails — more realistic)
    Z     = student_t.rvs(df=T_DOF, size=(T, N), random_state=seed)
    scale = np.sqrt((T_DOF - 2) / T_DOF)
    Z     = Z * scale

    drift  = (mu - 0.5 * sigma**2)
    shocks = drift + sigma * Z
    paths  = S0 * np.exp(np.cumsum(shocks, axis=0))
    
    # Add initial price as first row
    paths = np.vstack([np.full(N, S0), paths])
    
    return paths


# ════════════════════════════════════════════
#  CORRELATED MONTE CARLO SIMULATION
# ════════════════════════════════════════════

def correlated_monte_carlo(stocks_data: dict, nifty_returns: pd.Series, 
                           N: int = 5000, T: int = 252, seed: int = 42):
    """
    Simulate all stocks together using their real correlation structure.
    Uses Cholesky decomposition on the return covariance matrix.
    
    Args:
        stocks_data: dict with symbol -> {log_returns, mu_daily, sigma_daily, price}
        nifty_returns: pandas Series of Nifty log returns (for correlation context)
        N: number of simulations
        T: time horizon in days
        seed: random seed
        
    Returns:
        dict with symbol -> simulated price paths (T+1 x N)
    """
    try:
        symbols = list(stocks_data.keys())
        n = len(symbols)
        
        if n < 2:
            # Fallback to independent simulation for single stock
            results = {}
            for sym, data in stocks_data.items():
                paths = run_monte_carlo(data['price'], data['mu_daily'], 
                                        data['sigma_daily'], T=T, N=N, seed=seed)
                results[sym] = paths
            return results
        
        # Build return matrix (last 504 days × n_stocks)
        returns_matrix = np.column_stack([
            stocks_data[s]['log_returns'].values[-504:] for s in symbols
        ])
        
        # Covariance matrix from historical returns
        cov_matrix = np.cov(returns_matrix.T)
        
        # Add small regularization for numerical stability
        cov_matrix = cov_matrix + 1e-6 * np.eye(n)
        
        # Cholesky decomposition — creates correlated shocks
        L = np.linalg.cholesky(cov_matrix)
        
        # Set random seed and generate correlated random shocks
        np.random.seed(seed)
        Z_ind = np.random.standard_normal((T, N, n))  # independent: (T, N, n)
        
        # Apply Cholesky transformation: Z_cor[i,j,k] = sum_l(L[i,l] * Z_ind[j,k,l])
        # Shape: (T, N, n) = (T, N, n) @ (n, n).T
        Z_cor = np.zeros((T, N, n))
        for t in range(T):
            Z_cor[t, :, :] = Z_ind[t, :, :] @ L.T
        
        # Simulate correlated paths for each stock
        results = {}
        for i, sym in enumerate(symbols):
            mu    = stocks_data[sym]['mu_daily']
            sigma = stocks_data[sym]['sigma_daily']
            S0    = stocks_data[sym]['price']
            
            # Apply correlated shocks
            shocks = mu - 0.5 * sigma**2 + sigma * Z_cor[:, :, i]
            paths  = S0 * np.exp(np.cumsum(shocks, axis=0))
            results[sym] = paths
            
        logger.info(f"  📊 Correlated MC: {n} stocks, {N} sims, {T} days")
        return results
        
    except Exception as e:
        logger.warning(f"  Correlated MC failed: {e}, using independent simulation")
        # Fallback to independent simulation
        results = {}
        for sym, data in stocks_data.items():
            paths = run_monte_carlo(data['price'], data['mu_daily'], 
                                    data['sigma_daily'], T=T, N=N, seed=seed)
            results[sym] = paths
        return results


def compute_portfolio_var(results: dict, weights: dict, confidence: float = 0.95):
    """
    Compute portfolio-level VaR from correlated simulations.
    
    Args:
        results: dict from correlated_monte_carlo
        weights: dict of symbol -> weight (e.g., {"RELIANCE": 0.2, "TCS": 0.15, ...})
        confidence: VaR confidence level (default 0.95)
        
    Returns:
        portfolio VaR at specified confidence
    """
    try:
        final_prices = {}
        for sym, paths in results.items():
            final_prices[sym] = paths[-1, :]
        
        # Build portfolio returns
        symbols = list(final_prices.keys())
        initial_values = np.array([results[sym][0, 0] for sym in symbols])
        total_value = sum(weights.get(sym, 0) * initial_values[i] for i, sym in enumerate(symbols))
        
        # Calculate portfolio value at end for each simulation
        portfolio_values = np.zeros(final_prices[symbols[0]].shape)
        for i, sym in enumerate(symbols):
            weight = weights.get(sym, 0)
            portfolio_values += weight * final_prices[sym]
        
        # Portfolio returns
        portfolio_returns = (portfolio_values - total_value) / total_value
        
        # VaR
        var = float(-np.percentile(portfolio_returns, (1 - confidence) * 100))
        return round(var, 4)
        
    except Exception as e:
        logger.warning(f"  Portfolio VaR failed: {e}")
        return None


# ════════════════════════════════════════════
#  FAMA-FRENCH 3-FACTOR MODEL
# ════════════════════════════════════════════

def fama_french_drift(log_returns: pd.Series, nifty_returns: pd.Series,
                      smb_returns: pd.Series = None, hml_returns: pd.Series = None,
                      beta_nifty: float = 1.0) -> tuple:
    """
    Compute drift using Fama-French 3-Factor Model:
    μ = Rf + β_market×ERP + β_size×SMB + β_value×HML
    
    Args:
        log_returns: Stock log returns
        nifty_returns: Market (Nifty) returns
        smb_returns: Small Minus Big returns (if None, uses default)
        hml_returns: High Minus Low returns (if None, uses default)
        beta_nifty: Pre-computed market beta
        
    Returns:
        (mu_daily, method_string, factor_betas)
    """
    try:
        from sklearn.linear_model import LinearRegression
        
        # Use default factor returns if not provided
        if smb_returns is None:
            smb_returns = pd.Series(np.random.normal(SMB_PREMIUM_ANNUAL/252, 0.005, len(nifty_returns)),
                                    index=nifty_returns.index)
        if hml_returns is None:
            hml_returns = pd.Series(np.random.normal(HML_PREMIUM_ANNUAL/252, 0.006, len(nifty_returns)),
                                    index=nifty_returns.index)
        
        # Align data
        df = pd.DataFrame({
            'stock': log_returns.values[-504:],
            'market': nifty_returns.values[-504:],
            'smb': smb_returns.values[-504:],
            'hml': hml_returns.values[-504:],
        }).dropna()
        
        if len(df) < 100:
            # Fallback to CAPM
            mu_annual = RISK_FREE_RATE + beta_nifty * EQUITY_PREMIUM
            mu_daily = mu_annual / 252
            return float(mu_daily), "CAPM(fallback)", {'beta_market': beta_nifty, 'beta_smb': 0, 'beta_hml': 0}
        
        X = df[['market', 'smb', 'hml']].values
        y = df['stock'].values
        
        reg = LinearRegression().fit(X, y)
        b_mkt, b_smb, b_hml = reg.coef_
        
        # Factor premiums (daily)
        ERP = EQUITY_PREMIUM / 252       # Equity risk premium
        SMB_premium = SMB_PREMIUM_ANNUAL / 252
        HML_premium = HML_PREMIUM_ANNUAL / 252
        
        # Expected daily return
        mu_daily = (RISK_FREE_RATE / 252) + b_mkt * ERP + b_smb * SMB_premium + b_hml * HML_premium
        mu_daily = float(np.clip(mu_daily, MIN_MU_ANNUAL/252, MAX_MU_ANNUAL/252))
        
        method = f"FF3(β={b_mkt:.2f},β_smb={b_smb:.2f},β_hml={b_hml:.2f})"
        logger.info(f"    FF3: β_mkt={b_mkt:.2f}, β_smb={b_smb:.2f}, β_hml={b_hml:.2f} → μ={mu_daily*252:.2%}/yr")
        
        return mu_daily, method, {'beta_market': round(b_mkt, 3), 'beta_smb': round(b_smb, 3), 'beta_hml': round(b_hml, 3)}
        
    except Exception as e:
        logger.warning(f"  FF3 failed: {e}, using CAPM")
        mu_annual = RISK_FREE_RATE + beta_nifty * EQUITY_PREMIUM
        mu_daily = mu_annual / 252
        return float(mu_daily), "CAPM(fallback)", {'beta_market': beta_nifty, 'beta_smb': 0, 'beta_hml': 0}


# ════════════════════════════════════════════
#  EARNINGS RISK FLAG
# ════════════════════════════════════════════

def get_earnings_risk_flag(symbol: str) -> dict:
    """
    Flag stocks with earnings in next 30 days — widen confidence intervals.
    
    Returns:
        dict with earnings risk info and volatility multiplier
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol + ".NS")
        
        # Try to get earnings calendar
        try:
            cal = ticker.calendar
            if cal is None or cal.empty:
                return {'has_earnings_soon': False, 'vol_multiplier': 1.0, 'days_to_earnings': None}
            
            # Get next earnings date
            next_earnings = pd.Timestamp(cal.columns[0])
            days_to_earn = (next_earnings - pd.Timestamp.today()).days
            
            if 0 < days_to_earn < 30:
                # Try to get historical earnings moves
                try:
                    hist = ticker.earnings_history
                    if hist is not None and not hist.empty and 'surprisePercent' in hist.columns:
                        avg_move = hist['surprisePercent'].abs().mean() / 100
                    else:
                        avg_move = 0.07  # Default 7% move
                except:
                    avg_move = 0.07
                
                return {
                    'has_earnings_soon': True,
                    'days_to_earnings': days_to_earn,
                    'vol_multiplier': 1 + avg_move,
                    'avg_earnings_move': round(avg_move * 100, 1)
                }
        except:
            pass
            
        return {'has_earnings_soon': False, 'vol_multiplier': 1.0, 'days_to_earnings': None}
        
    except Exception as e:
        logger.debug(f"  Earnings risk check failed for {symbol}: {e}")
        return {'has_earnings_soon': False, 'vol_multiplier': 1.0, 'days_to_earnings': None}


# ════════════════════════════════════════════
#  SENTIMENT SCORE FROM NEWS
# ════════════════════════════════════════════

def get_sentiment_score(symbol: str, company_name: str = None) -> dict:
    """
    Use Google News RSS (free) + VADER sentiment analysis.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE")
        company_name: Company name for news search (if None, uses symbol)
        
    Returns:
        dict with sentiment_score, sentiment_label, news_count
    """
    try:
        import feedparser
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        if company_name is None:
            company_name = symbol
        
        # Google News RSS search
        query = company_name.replace(' ', '+')
        rss_url = f"https://news.google.com/rss/search?q={query}+stock&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        
        if not feed.entries:
            return {'sentiment_score': 0.0, 'sentiment_label': 'Neutral', 'news_count': 0}
        
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        
        for entry in feed.entries[:10]:  # Last 10 news items
            text = entry.title + ' ' + entry.get('summary', '')
            score = analyzer.polarity_scores(text)['compound']
            scores.append(score)
        
        avg_sentiment = np.mean(scores) if scores else 0
        
        return {
            'sentiment_score': round(avg_sentiment, 3),
            'sentiment_label': 'Positive' if avg_sentiment > 0.1 else 
                              'Negative' if avg_sentiment < -0.1 else 'Neutral',
            'news_count': len(scores)
        }
        
    except ImportError:
        logger.warning("  feedparser or vaderSentiment not installed, skipping sentiment")
        return {'sentiment_score': 0.0, 'sentiment_label': 'Unknown', 'news_count': 0, 'error': 'missing_lib'}
    except Exception as e:
        logger.debug(f"  Sentiment check failed for {symbol}: {e}")
        return {'sentiment_score': 0.0, 'sentiment_label': 'Neutral', 'news_count': 0}


# ════════════════════════════════════════════
#  OPTIONS IMPLIED VOLATILITY
# ════════════════════════════════════════════

def get_implied_volatility(symbol: str) -> float:
    """
    For F&O stocks, get ATM implied volatility from options market.
    Falls back to None if unavailable (non-F&O stock).
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE")
        
    Returns:
        Implied volatility (annualized) or None
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol + ".NS")
        
        # Check if options are available
        if not ticker.options:
            return None
        
        # Get nearest expiry options chain
        expiry = ticker.options[0]
        chain = ticker.option_chain(expiry)
        spot = ticker.fast_info.last_price
        
        if spot is None or spot <= 0:
            return None
        
        # Find ATM call
        calls = chain.calls
        if calls.empty:
            return None
            
        atm_idx = (calls['strike'] - spot).abs().idxmin()
        iv = calls.loc[atm_idx, 'impliedVolatility']
        
        # Sanity check (5% to 200% IV is reasonable)
        if iv and 0.05 < iv < 2.0:
            logger.info(f"  IV: {symbol} ATM IV = {iv:.1%}")
            return float(iv)
        
        return None
        
    except Exception as e:
        logger.debug(f"  IV check failed for {symbol}: {e}")
        return None


# ════════════════════════════════════════════
#  BACKTESTING ENGINE
# ════════════════════════════════════════════

def backtest_signals(signals_df: pd.DataFrame, price_data: dict, 
                     holding_days: int = 30) -> pd.DataFrame:
    """
    For each past signal, measure actual return over next N days.
    
    Args:
        signals_df: DataFrame with columns [date, symbol, signal, price_at_signal]
        price_data: dict of symbol -> price Series
        holding_days: Number of days to hold position
        
    Returns:
        DataFrame with accuracy stats per signal type
    """
    results = []
    
    for _, row in signals_df.iterrows():
        sym = row['symbol']
        date = row['date']
        sig = row['signal']
        p_buy = row['price_at_signal']
        
        if sym not in price_data:
            continue
            
        prices = price_data[sym]
        
        # Find price at signal date and after holding period
        try:
            future_prices = prices.loc[date:]
            if len(future_prices) > holding_days:
                p_future = future_prices.iloc[holding_days]
                actual_return = (p_future - p_buy) / p_buy * 100
                
                # Determine if signal was correct
                is_buy_signal = 'BUY' in sig
                is_correct = (is_buy_signal and actual_return > 0) or (sig == 'AVOID' and actual_return < 0)
                
                results.append({
                    'signal': sig,
                    'actual_return': actual_return,
                    'correct': is_correct,
                    'symbol': sym,
                    'holding_days': holding_days
                })
        except:
            continue
    
    if not results:
        return pd.DataFrame()
    
    # Aggregate stats per signal type
    stats = pd.DataFrame(results).groupby('signal').agg({
        'actual_return': ['mean', 'std', 'count'],
        'correct': 'mean'
    }).round(2)
    
    stats.columns = ['avg_return', 'std_return', 'count', 'accuracy']
    stats = stats.reset_index()
    
    logger.info(f"  📈 Backtest: {len(results)} signals analyzed")
    return stats


def save_daily_signals(symbol: str, signal_data: dict, filepath: str = "docs/signals_history.csv"):
    """
    Append daily signal to CSV for backtesting over time.
    """
    import os
    from datetime import datetime
    
    row = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'symbol': symbol,
        'signal': signal_data.get('signal', 'HOLD'),
        'price_at_signal': signal_data.get('price', 0),
        'score': signal_data.get('score', 0),
        'sharpe': signal_data.get('sharpe', 0),
        'prob_up': signal_data.get('prob_up', 0)
    }
    
    df = pd.DataFrame([row])
    
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath)
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv(filepath, index=False)
    logger.info(f"  💾 Signal saved to {filepath}")


def extract_path_percentiles(paths, step=5):
    idx = list(range(0, paths.shape[0], step))
    return {
        "p5" : np.percentile(paths[idx], 5,  axis=1).round(2).tolist(),
        "p25": np.percentile(paths[idx], 25, axis=1).round(2).tolist(),
        "p50": np.percentile(paths[idx], 50, axis=1).round(2).tolist(),
        "p75": np.percentile(paths[idx], 75, axis=1).round(2).tolist(),
        "p95": np.percentile(paths[idx], 95, axis=1).round(2).tolist(),
        "days": [i+step for i in idx],
    }


def extract_histogram(final, bins=50):
    counts, edges = np.histogram(final, bins=bins)
    return {"labels": ((edges[:-1]+edges[1:])/2).round(2).tolist(), "counts": counts.tolist()}


def compute_risk_metrics(final, S0, mu_daily, sigma_daily, log_returns):
    pnl       = (final - S0) / S0
    var_95    = float(-np.percentile(pnl, 5))
    var_99    = float(-np.percentile(pnl, 1))
    cvar_95   = float(-pnl[pnl <= np.percentile(pnl, 5)].mean())
    mu_ann    = float(mu_daily * 252)
    sigma_ann = float(sigma_daily * np.sqrt(252))
    excess    = mu_ann - RISK_FREE_RATE
    sharpe    = excess / sigma_ann if sigma_ann > 0 else 0.0
    neg       = log_returns[log_returns < 0]
    dstd      = float(neg.std() * np.sqrt(252)) if len(neg) > 10 else sigma_ann
    sortino   = excess / dstd if dstd > 0 else 0.0
    cs        = log_returns.cumsum().apply(np.exp)
    max_dd    = float(((cs - cs.cummax()) / cs.cummax()).min())
    calmar    = mu_ann / abs(max_dd) if max_dd != 0 else 0.0
    zscore    = (float(np.mean(final)) - S0) / (S0 * sigma_ann) if sigma_ann > 0 else 0.0
    return {
        "var_95": round(var_95,4), "var_99": round(var_99,4), "cvar_95": round(cvar_95,4),
        "sharpe": round(sharpe,3), "sortino": round(sortino,3), "calmar": round(calmar,3),
        "max_drawdown": round(max_dd,4), "mu_annual": round(mu_ann,4),
        "sigma_annual": round(sigma_ann,4), "price_zscore": round(zscore,3),
    }


def generate_signal(prob_up, var_95, sharpe, expected_ret, mispricing, beta):
    score = 0
    if prob_up > 0.68:    score += 3
    elif prob_up > 0.58:  score += 2
    elif prob_up > 0.50:  score += 1
    elif prob_up < 0.40:  score -= 3
    elif prob_up < 0.50:  score -= 1
    if sharpe > 0.8:    score += 2
    elif sharpe > 0.4:  score += 1
    elif sharpe < -0.1: score -= 2
    elif sharpe < 0.2:  score -= 1
    if var_95 < 0.10:    score += 2
    elif var_95 < 0.15:  score += 1
    elif var_95 > 0.25:  score -= 2
    elif var_95 > 0.20:  score -= 1
    if expected_ret > 0.12:   score += 2
    elif expected_ret > 0.07: score += 1
    elif expected_ret < 0:    score -= 2
    elif expected_ret < 0.03: score -= 1
    if mispricing > 0.08:    score += 1
    elif mispricing < -0.08: score -= 1
    if score >= 5:    signal, color = "STRONG BUY", "#00c853"
    elif score >= 2:  signal, color = "BUY", "#2ecc8a"
    elif score >= -1: signal, color = "HOLD", "#f0b429"
    elif score >= -4: signal, color = "RISKY", "#ff9800"
    else:             signal, color = "AVOID", "#e02d3c"
    return {"signal": signal, "signal_color": color, "score": score,
            "confidence": min(100, max(0, int((score+10)/20*100)))}


def run_full_pipeline(symbol, features, fin_info, nifty_returns=None):
    try:
        log_ret    = features["log_returns"]
        S0         = features["current_price"]
        sigma_ewma = features["sigma_daily"]
        if len(log_ret) < 100:
            return None

        mu, beta, method = estimate_drift(log_ret, nifty_returns)
        sigma  = garch_volatility(log_ret, sigma_ewma)
        paths  = run_monte_carlo(S0, mu, sigma)
        final  = paths[-1, :]
        pc     = extract_path_percentiles(paths)
        hist   = extract_histogram(final)
        risk   = compute_risk_metrics(final, S0, mu, sigma, log_ret)

        mean_p = float(np.mean(final))
        exp_r  = (mean_p / S0) - 1
        misp   = (mean_p - S0) / S0
        pu     = float(np.mean(final > S0))
        p10u   = float(np.mean(final > S0*1.1))
        p20u   = float(np.mean(final > S0*1.2))
        p10d   = float(np.mean(final < S0*0.9))
        sig    = generate_signal(pu, risk["var_95"], risk["sharpe"], exp_r, misp, beta)

        def sc(mm, sm):
            return float(np.median(run_monte_carlo(S0, mu*mm, sigma*sm, N=500)[-1, :]))

        logger.info(
            f"  ✅ {symbol}: ₹{S0:,.2f}→E[₹{mean_p:,.2f}] ({exp_r:+.1%}) | "
            f"{sig['signal']} | P(↑):{pu:.1%} | Sharpe:{risk['sharpe']:.2f} | β={beta:.2f}"
        )

        return {
            "symbol": symbol, "name": fin_info.get("name", symbol),
            "sector": fin_info.get("sector","Other"), "industry": fin_info.get("industry",""),
            "description": fin_info.get("description",""), "website": fin_info.get("website",""),
            "price": round(S0,2), "week52_high": round(features["week52_high"],2),
            "week52_low": round(features["week52_low"],2),
            "market_cap": fin_info.get("market_cap"), "pe_ratio": fin_info.get("pe_ratio"),
            "pb_ratio": fin_info.get("pb_ratio"), "eps": fin_info.get("eps"),
            "revenue": fin_info.get("revenue"), "net_income": fin_info.get("net_income"),
            "roe": fin_info.get("roe"), "roa": fin_info.get("roa"),
            "debt_equity": fin_info.get("debt_equity"), "current_ratio": fin_info.get("current_ratio"),
            "dividend_yield": fin_info.get("dividend_yield"),
            "book_value": fin_info.get("book_value"), "employees": fin_info.get("employees"),
            "mean_price": round(mean_p,2), "median_price": round(float(np.percentile(final,50)),2),
            "ci_5": round(float(np.percentile(final,5)),2), "ci_25": round(float(np.percentile(final,25)),2),
            "ci_75": round(float(np.percentile(final,75)),2), "ci_95": round(float(np.percentile(final,95)),2),
            "expected_return": round(exp_r,4), "expected_return_pct": round(exp_r*100,2),
            "mispricing_pct": round(misp*100,2),
            "prob_up": round(pu,4), "prob_10up": round(p10u,4),
            "prob_20up": round(p20u,4), "prob_10down": round(p10d,4),
            **risk, **sig,
            "mu_annual": round(mu*252,4), "sigma_annual": round(sigma*np.sqrt(252),4),
            "mu_daily": round(mu,6), "drift_method": method, "beta_nifty": round(beta,3),
            "mom_1m": round(features["mom_1m"]*100,2), "mom_3m": round(features["mom_3m"]*100,2),
            "mom_6m": round(features["mom_6m"]*100,2), "mom_1y": round(features["mom_1y"]*100,2),
            "path_charts": pc, "histogram": hist,
            "bull_median": round(sc(1.2,0.85),2), "base_median": round(sc(1.0,1.0),2),
            "bear_median": round(sc(0.8,1.3),2),
            "n_simulations": N_SIMULATIONS, "horizon_days": HORIZON_DAYS,
            "model": "CAPM(NiftyBeta)+Mom-GARCH-GBM-StudentT",
        }
    except Exception as e:
        logger.error(f"  {symbol}: failed — {e}")
        import traceback; traceback.print_exc()
        return None


# ════════════════════════════════════════════
#  TECHNICAL ANALYSIS ENGINE
# ════════════════════════════════════════════

def compute_technical_indicators(df: pd.DataFrame) -> dict:
    """
    Computes key technical indicators from OHLCV data.
    Returns dict of indicator values + signals.
    """
    try:
        close  = df["Close"].squeeze().astype(float)
        high   = df["High"].squeeze().astype(float)  if "High"   in df.columns else close
        low    = df["Low"].squeeze().astype(float)   if "Low"    in df.columns else close
        volume = df["Volume"].squeeze().astype(float) if "Volume" in df.columns else None

        result = {}

        # ── RSI (14) ────────────────────────────
        delta   = close.diff()
        gain    = delta.clip(lower=0).rolling(14).mean()
        loss    = (-delta.clip(upper=0)).rolling(14).mean()
        rs      = gain / loss.replace(0, 1e-10)
        rsi     = float((100 - 100/(1+rs)).iloc[-1])
        result["rsi_14"]        = round(rsi, 2)
        result["rsi_signal"]    = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"

        # ── MACD (12,26,9) ──────────────────────
        ema12   = close.ewm(span=12).mean()
        ema26   = close.ewm(span=26).mean()
        macd    = ema12 - ema26
        signal  = macd.ewm(span=9).mean()
        hist    = macd - signal
        result["macd"]          = round(float(macd.iloc[-1]), 4)
        result["macd_signal"]   = round(float(signal.iloc[-1]), 4)
        result["macd_hist"]     = round(float(hist.iloc[-1]), 4)
        result["macd_cross"]    = "Bullish" if float(macd.iloc[-1]) > float(signal.iloc[-1]) else "Bearish"

        # ── Bollinger Bands (20,2) ───────────────
        sma20   = close.rolling(20).mean()
        std20   = close.rolling(20).std()
        bb_up   = float((sma20 + 2*std20).iloc[-1])
        bb_mid  = float(sma20.iloc[-1])
        bb_low  = float((sma20 - 2*std20).iloc[-1])
        cur_p   = float(close.iloc[-1])
        bb_pct  = (cur_p - bb_low) / (bb_up - bb_low) if bb_up != bb_low else 0.5
        result["bb_upper"]      = round(bb_up, 2)
        result["bb_middle"]     = round(bb_mid, 2)
        result["bb_lower"]      = round(bb_low, 2)
        result["bb_position"]   = round(float(bb_pct), 3)
        result["bb_signal"]     = "Near Upper" if bb_pct > 0.8 else "Near Lower" if bb_pct < 0.2 else "Middle"

        # ── Moving Averages ──────────────────────
        sma_50  = float(close.rolling(50).mean().iloc[-1])  if len(close) >= 50  else None
        sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        ema_20  = float(close.ewm(span=20).mean().iloc[-1])
        result["sma_50"]        = round(sma_50,  2) if sma_50  else None
        result["sma_200"]       = round(sma_200, 2) if sma_200 else None
        result["ema_20"]        = round(ema_20,  2)
        result["above_sma50"]   = bool(cur_p > sma_50)  if sma_50  else None
        result["above_sma200"]  = bool(cur_p > sma_200) if sma_200 else None
        result["golden_cross"]  = bool(sma_50 > sma_200) if (sma_50 and sma_200) else None

        # ── ATR (14) ────────────────────────────
        tr1  = high - low
        tr2  = (high - close.shift()).abs()
        tr3  = (low  - close.shift()).abs()
        atr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
        result["atr_14"]        = round(float(atr.iloc[-1]), 2)
        result["atr_pct"]       = round(float(atr.iloc[-1]) / cur_p * 100, 2)

        # ── Stochastic (14,3) ────────────────────
        lo14 = low.rolling(14).min()
        hi14 = high.rolling(14).max()
        k    = 100 * (close - lo14) / (hi14 - lo14 + 1e-10)
        d    = k.rolling(3).mean()
        result["stoch_k"]       = round(float(k.iloc[-1]), 2)
        result["stoch_d"]       = round(float(d.iloc[-1]), 2)
        result["stoch_signal"]  = "Oversold" if float(k.iloc[-1]) < 20 else "Overbought" if float(k.iloc[-1]) > 80 else "Neutral"

        # ── Volume Analysis ──────────────────────
        if volume is not None:
            vol_sma20 = volume.rolling(20).mean()
            result["vol_ratio"]     = round(float(volume.iloc[-1]/vol_sma20.iloc[-1]), 2) if float(vol_sma20.iloc[-1]) > 0 else 1.0
            result["vol_trend"]     = "Above Average" if result["vol_ratio"] > 1.2 else "Below Average" if result["vol_ratio"] < 0.8 else "Normal"

        # ── Overall Technical Score ──────────────
        tech_score = 0
        if rsi < 35:                                tech_score += 2
        elif rsi < 45:                              tech_score += 1
        elif rsi > 75:                              tech_score -= 2
        elif rsi > 65:                              tech_score -= 1
        if result["macd_cross"] == "Bullish":       tech_score += 2
        else:                                       tech_score -= 1
        if bb_pct < 0.25:                           tech_score += 2
        elif bb_pct > 0.85:                         tech_score -= 2
        if result.get("above_sma50"):               tech_score += 1
        if result.get("above_sma200"):              tech_score += 1
        if result.get("golden_cross"):              tech_score += 1
        if result.get("vol_ratio", 1) > 1.5:       tech_score += 1

        result["tech_score"]    = tech_score
        result["tech_signal"]   = (
            "Strong Buy"  if tech_score >= 5 else
            "Buy"         if tech_score >= 2 else
            "Neutral"     if tech_score >= -1 else
            "Sell"        if tech_score >= -4 else
            "Strong Sell"
        )

        return result

    except Exception as e:
        logger.warning(f"  Technical analysis failed: {e}")
        return {}


# ════════════════════════════════════════════
#  FUNDAMENTAL ANALYSIS ENGINE
# ════════════════════════════════════════════

def compute_fundamental_score(fin_info: dict, risk_metrics: dict) -> dict:
    """
    Scores a stock on fundamental metrics.
    Returns score, grade, and individual metric assessments.
    """
    score = 0
    details = {}

    # P/E ratio
    pe = fin_info.get("pe_ratio")
    if pe:
        if pe < 15:      score += 2; details["pe"] = "Cheap"
        elif pe < 25:    score += 1; details["pe"] = "Fair"
        elif pe < 40:    score -= 1; details["pe"] = "Expensive"
        else:            score -= 2; details["pe"] = "Very Expensive"
    
    # P/B ratio
    pb = fin_info.get("pb_ratio")
    if pb:
        if pb < 1.5:     score += 2; details["pb"] = "Undervalued"
        elif pb < 3:     score += 1; details["pb"] = "Fair"
        elif pb < 6:     score -= 1; details["pb"] = "Premium"
        else:            score -= 2; details["pb"] = "Expensive"

    # ROE
    roe = fin_info.get("roe")
    if roe:
        roe_pct = roe * 100
        if roe_pct > 20:   score += 2; details["roe"] = "Excellent"
        elif roe_pct > 12: score += 1; details["roe"] = "Good"
        elif roe_pct > 5:  score -= 1; details["roe"] = "Below Average"
        else:              score -= 2; details["roe"] = "Poor"

    # Debt/Equity
    de = fin_info.get("debt_equity")
    if de is not None:
        if de < 0.3:     score += 2; details["debt"] = "Low Debt"
        elif de < 1.0:   score += 1; details["debt"] = "Moderate"
        elif de < 2.0:   score -= 1; details["debt"] = "High Debt"
        else:            score -= 2; details["debt"] = "Very High Debt"

    # Dividend yield
    dy = fin_info.get("dividend_yield")
    if dy:
        dy_pct = dy * 100
        if dy_pct > 3:   score += 1; details["dividend"] = "High Yield"
        elif dy_pct > 1: score += 0; details["dividend"] = "Moderate Yield"

    # Sharpe ratio (from risk metrics)
    sh = risk_metrics.get("sharpe", 0)
    if sh > 1.0:  score += 2; details["risk_adj"] = "Excellent"
    elif sh > 0.5: score += 1; details["risk_adj"] = "Good"
    elif sh < 0:   score -= 1; details["risk_adj"] = "Poor"

    grade = (
        "A+" if score >= 8 else "A"  if score >= 6 else
        "B+" if score >= 4 else "B"  if score >= 2 else
        "C+" if score >= 0 else "C"  if score >= -2 else "D"
    )

    return {
        "fundamental_score" : score,
        "fundamental_grade" : grade,
        "fundamental_details": details,
    }
