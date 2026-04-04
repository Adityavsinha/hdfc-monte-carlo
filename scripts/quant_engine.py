"""
scripts/quant_engine.py
-----------------------
Full institutional-grade quant pipeline:
  1. ARIMA forecast → drift (μ)
  2. GARCH volatility → σ
  3. Monte Carlo GBM simulation
  4. Risk engine (VaR, CVaR, Sharpe, Max Drawdown)
  5. Signal engine (BUY / HOLD / RISKY)
  6. Mispricing indicator
"""

import numpy as np
import pandas as pd
import warnings
import logging
from scipy.stats import t as student_t
from scipy.stats import norm
from config import (N_SIMULATIONS, HORIZON_DAYS, RISK_FREE_RATE, T_DOF,
                    BUY_PROB_THRESHOLD, RISKY_VAR_THRESHOLD, HOLD_SHARPE_MIN)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


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
) -> dict | None:
    """
    Runs complete quant pipeline for one stock.
    Returns full metrics dict ready for JSON output.
    """
    try:
        log_returns   = features["log_returns"]
        S0            = features["current_price"]
        sigma_ewma    = features["sigma_daily"]

        if len(log_returns) < 100:
            logger.warning(f"  {symbol}: not enough returns ({len(log_returns)})")
            return None

        # ── 1. ARIMA drift ──────────────────────
        logger.info(f"    ARIMA...")
        mu = arima_drift(log_returns)

        # ── 2. GARCH volatility ─────────────────
        logger.info(f"    GARCH...")
        sigma = garch_volatility(log_returns)
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

        # ── 7. Scenario medians ─────────────────
        def scen(mu_m, sig_m):
            p = run_monte_carlo(S0, mu * mu_m, sigma * sig_m, N=1000)
            return float(np.median(p[-1, :]))

        # ── 8. Signal ───────────────────────────
        sig_result = generate_signal(
            prob_up, risk["var_95"],
            risk["sharpe"], expected_ret, mispricing
        )

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

            # Meta
            "n_simulations"   : N_SIMULATIONS,
            "horizon_days"    : HORIZON_DAYS,
            "model"           : "ARIMA-GARCH-GBM-StudentT",
        }

    except Exception as e:
        logger.error(f"  {symbol}: pipeline failed — {e}")
        import traceback; traceback.print_exc()
        return None
