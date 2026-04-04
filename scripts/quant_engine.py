"""
scripts/quant_engine.py
-----------------------
Institutional-grade quant pipeline:
  1. CAPM drift (primary) + ARIMA adjustment (supplementary)
  2. GARCH(1,1) volatility with EWMA fallback
  3. GBM Monte Carlo with Student-t fat tails
  4. Risk engine: VaR, CVaR, Sharpe, Sortino, Calmar
  5. Signal engine: STRONG BUY / BUY / HOLD / RISKY / AVOID
  6. Mispricing indicator

DRIFT MODEL:
  Uses CAPM as the base drift to avoid historical mean extrapolation.
  μ = Rf + β × (Rm - Rf)
  India Rf = 6.5% (RBI repo), Rm = 13% (long-run Nifty CAGR)
  ARIMA adjusts the CAPM drift by short-term momentum signal.
  Final μ is clamped to [-15%, +30%] annualised — realistic range.
"""

import numpy as np
import pandas as pd
import warnings
import logging
from scipy.stats import t as student_t
from config import (N_SIMULATIONS, HORIZON_DAYS, RISK_FREE_RATE, T_DOF,
                    BUY_PROB_THRESHOLD, RISKY_VAR_THRESHOLD, HOLD_SHARPE_MIN)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── India market constants ─────────────────────
MARKET_RETURN_ANNUAL = 0.13   # Long-run Nifty 50 CAGR (~13%)
EQUITY_PREMIUM       = MARKET_RETURN_ANNUAL - RISK_FREE_RATE   # ~6.5%
MIN_MU_ANNUAL        = -0.15  # Floor: -15% annual
MAX_MU_ANNUAL        = +0.30  # Cap:   +30% annual


# ════════════════════════════════════════════
#  CAPM DRIFT (Primary μ estimator)
# ════════════════════════════════════════════

def capm_drift(log_returns: pd.Series, beta: float | None = None) -> float:
    """
    Estimates annualised expected return using CAPM.
    μ_annual = Rf + β × (Rm - Rf)
    Falls back to β=1.0 (market) if not provided.

    Returns daily drift.
    """
    if beta is None or np.isnan(beta) or beta <= 0:
        # Estimate beta from returns correlation with a proxy
        # Use simple 1.0 (market beta) as safe default
        beta = 1.0

    # Clamp beta to realistic range
    beta = np.clip(beta, 0.3, 2.5)

    mu_annual = RISK_FREE_RATE + beta * EQUITY_PREMIUM
    mu_daily  = mu_annual / 252

    logger.debug(f"    CAPM: β={beta:.2f} → μ_annual={mu_annual:.4f}")
    return float(mu_daily)


# ════════════════════════════════════════════
#  ARIMA MOMENTUM ADJUSTMENT (Supplementary)
# ════════════════════════════════════════════

def arima_adjustment(log_returns: pd.Series) -> float:
    """
    Uses ARIMA(1,0,1) on recent returns to get a momentum signal.
    Returns a SMALL daily adjustment to CAPM drift (-0.002 to +0.002).
    This is supplementary — not the primary drift source.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        series = log_returns.iloc[-126:].dropna()  # Last 6 months
        if len(series) < 40:
            return 0.0

        model    = ARIMA(series, order=(1, 0, 1))
        result   = model.fit()
        forecast = result.forecast(steps=1)
        adj      = float(forecast.iloc[0])

        # Only use as a small momentum nudge — clamp tightly
        adj = np.clip(adj, -0.002, 0.002)
        logger.debug(f"    ARIMA adj = {adj:.6f}")
        return adj

    except Exception as e:
        logger.debug(f"    ARIMA skipped ({type(e).__name__})")
        return 0.0


# ════════════════════════════════════════════
#  COMBINED DRIFT ESTIMATOR
# ════════════════════════════════════════════

def estimate_drift(
    log_returns : pd.Series,
    beta        : float | None = None,
) -> tuple[float, str]:
    """
    Final drift = CAPM base + ARIMA momentum adjustment
    Clamped to realistic annual range.
    Returns (mu_daily, method_used).
    """
    # 1. CAPM base
    mu_capm  = capm_drift(log_returns, beta)

    # 2. ARIMA momentum adjustment (small nudge)
    mu_arima_adj = arima_adjustment(log_returns)

    # 3. Combine
    mu_daily = mu_capm + mu_arima_adj

    # 4. Hard clamp to realistic annual range
    mu_annual_raw = mu_daily * 252
    mu_annual     = np.clip(mu_annual_raw, MIN_MU_ANNUAL, MAX_MU_ANNUAL)
    mu_daily      = mu_annual / 252

    method = f"CAPM(β)+ARIMA_adj → {mu_annual:.2%}/yr"
    logger.debug(f"    Drift: {method}")
    return float(mu_daily), method


# ════════════════════════════════════════════
#  GARCH VOLATILITY
# ════════════════════════════════════════════

def garch_volatility(log_returns: pd.Series, ewma_sigma: float) -> float:
    """
    GARCH(1,1) conditional volatility estimate.
    Blended with EWMA for stability.
    Falls back to pure EWMA on failure.
    """
    try:
        from arch import arch_model
        series = log_returns.iloc[-504:].dropna() * 100  # Scale for GARCH
        if len(series) < 100:
            return ewma_sigma

        model  = arch_model(series, vol="Garch", p=1, q=1,
                            dist="StudentsT", rescale=False)
        result = model.fit(disp="off", show_warning=False)

        forecast   = result.forecast(horizon=1)
        cond_var   = forecast.variance.values[-1, 0]
        sigma_garch = float(np.sqrt(cond_var) / 100)

        # Sanity check
        if sigma_garch < 0.003 or sigma_garch > 0.08:
            return ewma_sigma

        # Blend: 60% GARCH + 40% EWMA
        sigma_blended = 0.6 * sigma_garch + 0.4 * ewma_sigma
        logger.debug(f"    GARCH σ={sigma_garch:.5f}, EWMA σ={ewma_sigma:.5f} → blended={sigma_blended:.5f}")
        return float(sigma_blended)

    except Exception as e:
        logger.debug(f"    GARCH failed ({type(e).__name__}), using EWMA")
        return ewma_sigma


# ════════════════════════════════════════════
#  MONTE CARLO SIMULATION (GBM)
# ════════════════════════════════════════════

def run_monte_carlo(
    S0    : float,
    mu    : float,
    sigma : float,
    T     : int   = HORIZON_DAYS,
    N     : int   = N_SIMULATIONS,
    seed  : int   = 42,
) -> np.ndarray:
    """
    Vectorised GBM with Student-t shocks.
    Returns full path matrix (T, N).
    """
    Z     = student_t.rvs(df=T_DOF, size=(T, N), random_state=seed)
    scale = np.sqrt((T_DOF - 2) / T_DOF)
    Z     = Z * scale

    drift  = (mu - 0.5 * sigma**2)
    paths  = S0 * np.exp(np.cumsum(drift + sigma * Z, axis=0))
    return paths


def extract_path_percentiles(paths: np.ndarray, sample_every: int = 5) -> dict:
    idx = list(range(0, paths.shape[0], sample_every))
    return {
        "p5"  : np.percentile(paths[idx], 5,  axis=1).round(2).tolist(),
        "p25" : np.percentile(paths[idx], 25, axis=1).round(2).tolist(),
        "p50" : np.percentile(paths[idx], 50, axis=1).round(2).tolist(),
        "p75" : np.percentile(paths[idx], 75, axis=1).round(2).tolist(),
        "p95" : np.percentile(paths[idx], 95, axis=1).round(2).tolist(),
        "days": [i + sample_every for i in idx],
    }


def extract_histogram(final_prices: np.ndarray, bins: int = 50) -> dict:
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
    pnl_pct  = (final_prices - S0) / S0

    var_95   = float(-np.percentile(pnl_pct, 5))
    var_99   = float(-np.percentile(pnl_pct, 1))

    tail     = pnl_pct[pnl_pct <= np.percentile(pnl_pct, 5)]
    cvar_95  = float(-tail.mean())

    mu_annual    = float(mu_daily * 252)
    sigma_annual = float(sigma_daily * np.sqrt(252))

    excess = mu_annual - RISK_FREE_RATE
    sharpe = excess / sigma_annual if sigma_annual > 0 else 0.0

    neg_ret      = log_returns[log_returns < 0]
    down_std     = float(neg_ret.std() * np.sqrt(252)) if len(neg_ret) > 10 else sigma_annual
    sortino      = excess / down_std if down_std > 0 else 0.0

    close        = log_returns.cumsum().apply(np.exp)
    rolling_max  = close.cummax()
    drawdown     = (close - rolling_max) / rolling_max
    max_dd       = float(drawdown.min())

    calmar = mu_annual / abs(max_dd) if max_dd != 0 else 0.0

    expected_price = float(np.mean(final_prices))
    zscore = (expected_price - S0) / (S0 * sigma_annual) if sigma_annual > 0 else 0.0

    return {
        "var_95"      : round(var_95, 4),
        "var_99"      : round(var_99, 4),
        "cvar_95"     : round(cvar_95, 4),
        "sharpe"      : round(sharpe, 3),
        "sortino"     : round(sortino, 3),
        "calmar"      : round(calmar, 3),
        "max_drawdown": round(max_dd, 4),
        "mu_annual"   : round(mu_annual, 4),
        "sigma_annual": round(sigma_annual, 4),
        "price_zscore": round(zscore, 3),
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
    score = 0

    if prob_up > 0.70:   score += 3
    elif prob_up > 0.60: score += 2
    elif prob_up > 0.50: score += 1
    elif prob_up < 0.40: score -= 3
    elif prob_up < 0.50: score -= 1

    if sharpe > 1.0:   score += 2
    elif sharpe > 0.5: score += 1
    elif sharpe < 0.0: score -= 2
    elif sharpe < 0.3: score -= 1

    if var_95 < 0.10:   score += 2
    elif var_95 < 0.15: score += 1
    elif var_95 > 0.25: score -= 2
    elif var_95 > 0.20: score -= 1

    if expected_ret > 0.15:   score += 2
    elif expected_ret > 0.08: score += 1
    elif expected_ret < 0:    score -= 2
    elif expected_ret < 0.04: score -= 1

    if mispricing > 0.10:    score += 1
    elif mispricing < -0.10: score -= 1

    if score >= 5:
        signal, color = "STRONG BUY", "#00c853"
    elif score >= 2:
        signal, color = "BUY", "#2ecc8a"
    elif score >= -1:
        signal, color = "HOLD", "#f0b429"
    elif score >= -4:
        signal, color = "RISKY", "#ff9800"
    else:
        signal, color = "AVOID", "#e02d3c"

    confidence = min(100, max(0, int((score + 10) / 20 * 100)))
    return {"signal": signal, "signal_color": color, "score": score, "confidence": confidence}


# ════════════════════════════════════════════
#  FULL PIPELINE FOR ONE STOCK
# ════════════════════════════════════════════

def run_full_pipeline(symbol: str, features: dict, fin_info: dict) -> dict | None:
    try:
        log_returns = features["log_returns"]
        S0          = features["current_price"]
        sigma_ewma  = features["sigma_daily"]
        beta        = fin_info.get("beta")

        if len(log_returns) < 100:
            logger.warning(f"  {symbol}: insufficient returns ({len(log_returns)})")
            return None

        # ── 1. Drift (CAPM + ARIMA) ─────────────
        logger.info(f"    Estimating drift (CAPM + ARIMA)...")
        mu, drift_method = estimate_drift(log_returns, beta)

        # ── 2. Volatility (GARCH + EWMA) ────────
        logger.info(f"    Estimating volatility (GARCH)...")
        sigma = garch_volatility(log_returns, sigma_ewma)

        # ── 3. Monte Carlo ───────────────────────
        logger.info(f"    Running Monte Carlo ({N_SIMULATIONS:,} paths)...")
        paths    = run_monte_carlo(S0, mu, sigma)
        final    = paths[-1, :]
        p_charts = extract_path_percentiles(paths)
        hist     = extract_histogram(final)

        # ── 4. Risk metrics ──────────────────────
        risk = compute_risk_metrics(final, S0, mu, sigma, log_returns)

        # ── 5. Distribution stats ────────────────
        mean_price   = float(np.mean(final))
        median_price = float(np.median(final))
        ci_5         = float(np.percentile(final, 5))
        ci_25        = float(np.percentile(final, 25))
        ci_75        = float(np.percentile(final, 75))
        ci_95        = float(np.percentile(final, 95))

        prob_up    = float(np.mean(final > S0))
        prob_10up  = float(np.mean(final > S0 * 1.10))
        prob_20up  = float(np.mean(final > S0 * 1.20))
        prob_10down= float(np.mean(final < S0 * 0.90))

        expected_ret = (mean_price / S0) - 1
        mispricing   = (mean_price - S0) / S0

        # ── 6. Scenarios ─────────────────────────
        def scen_med(mu_m, sig_m):
            p = run_monte_carlo(S0, mu * mu_m, sigma * sig_m, N=500)
            return float(np.median(p[-1, :]))

        # ── 7. Signal ────────────────────────────
        sig = generate_signal(prob_up, risk["var_95"], risk["sharpe"],
                               expected_ret, mispricing)

        logger.info(
            f"  ✅ {symbol}: ₹{S0:,.2f} → E[₹{mean_price:,.2f}] "
            f"({expected_ret:+.1%}) | Signal:{sig['signal']} | "
            f"P(↑):{prob_up:.1%} | Sharpe:{risk['sharpe']:.2f} | "
            f"μ={mu*252:.2%}/yr | σ={sigma*252**.5:.2%}/yr | {drift_method}"
        )

        return {
            "symbol"          : symbol,
            "name"            : fin_info.get("name", symbol),
            "sector"          : fin_info.get("sector", features.get("sector", "Other")),
            "industry"        : fin_info.get("industry", ""),
            "description"     : fin_info.get("description", ""),
            "website"         : fin_info.get("website", ""),
            "price"           : round(S0, 2),
            "week52_high"     : round(features["week52_high"], 2),
            "week52_low"      : round(features["week52_low"], 2),
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
            "mean_price"      : round(mean_price, 2),
            "median_price"    : round(median_price, 2),
            "ci_5"            : round(ci_5, 2),
            "ci_25"           : round(ci_25, 2),
            "ci_75"           : round(ci_75, 2),
            "ci_95"           : round(ci_95, 2),
            "expected_return" : round(expected_ret, 4),
            "expected_return_pct": round(expected_ret * 100, 2),
            "mispricing_pct"  : round(mispricing * 100, 2),
            "prob_up"         : round(prob_up, 4),
            "prob_10up"       : round(prob_10up, 4),
            "prob_20up"       : round(prob_20up, 4),
            "prob_10down"     : round(prob_10down, 4),
            **risk,
            **sig,
            "mu_annual"       : round(mu * 252, 4),
            "sigma_annual"    : round(sigma * np.sqrt(252), 4),
            "mu_daily"        : round(mu, 6),
            "drift_method"    : drift_method,
            "mom_1m"          : round(features["mom_1m"] * 100, 2),
            "mom_3m"          : round(features["mom_3m"] * 100, 2),
            "mom_6m"          : round(features["mom_6m"] * 100, 2),
            "mom_1y"          : round(features["mom_1y"] * 100, 2),
            "path_charts"     : p_charts,
            "histogram"       : hist,
            "bull_median"     : round(scen_med(1.3, 0.8), 2),
            "base_median"     : round(scen_med(1.0, 1.0), 2),
            "bear_median"     : round(scen_med(0.7, 1.4), 2),
            "n_simulations"   : N_SIMULATIONS,
            "horizon_days"    : HORIZON_DAYS,
            "model"           : "CAPM+ARIMA-GARCH-GBM-StudentT",
        }

    except Exception as e:
        logger.error(f"  {symbol}: pipeline failed — {e}")
        import traceback; traceback.print_exc()
        return None
