"""
simulation_engine.py
--------------------
Geometric Brownian Motion Monte Carlo engine with:
  - Rolling volatility
  - Fat-tail (Student-t) option
  - CAPM drift adjustment
  - Scenario analysis (Bull / Base / Bear)
  - VaR / CVaR, confidence intervals, probability metrics
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG DATACLASS
# ─────────────────────────────────────────────
@dataclass
class SimConfig:
    n_simulations   : int   = 10_000
    horizon_days    : int   = 252        # 1 trading year
    rolling_vol_win : int   = 63         # ~3-month rolling window
    risk_free_rate  : float = 0.065      # RBI repo rate annualised (~6.5%)
    use_fat_tails   : bool  = True       # Student-t distribution
    t_dof           : int   = 5          # degrees of freedom for Student-t
    use_capm_drift  : bool  = False      # if True, uses risk-free rate as drift baseline
    random_seed     : int   = 42


@dataclass
class SimResults:
    paths            : np.ndarray   # shape (horizon, n_simulations)
    S0               : float
    config           : SimConfig
    mu               : float
    sigma            : float
    scenarios        : Dict[str, np.ndarray] = field(default_factory=dict)

    # ── derived metrics (filled by analyse()) ──
    mean_price       : float = 0.0
    median_price     : float = 0.0
    ci_5             : float = 0.0
    ci_50            : float = 0.0
    ci_95            : float = 0.0
    var_95           : float = 0.0   # Value at Risk (95%)
    var_99           : float = 0.0
    cvar_95          : float = 0.0   # Conditional VaR / Expected Shortfall
    prob_increase    : float = 0.0
    prob_10pct_up    : float = 0.0
    prob_10pct_down  : float = 0.0
    final_prices     : np.ndarray = field(default_factory=lambda: np.array([]))


# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────

def estimate_params(
    log_returns : pd.Series,
    cfg         : SimConfig,
    nifty_beta  : float = 1.0,
) -> Tuple[float, float]:
    """
    Estimate annualised drift (μ) and volatility (σ).
    Uses rolling volatility (ewm) for a more responsive σ estimate.
    Optionally adjusts drift via CAPM.
    """
    # Rolling volatility: exponentially-weighted std of last `rolling_vol_win` days
    rolling_vol = log_returns.ewm(span=cfg.rolling_vol_win).std()
    sigma_daily = rolling_vol.iloc[-1]
    sigma_annual = sigma_daily * np.sqrt(252)

    mu_daily   = log_returns.mean()
    mu_annual  = mu_daily * 252

    if cfg.use_capm_drift:
        # CAPM: E[R] = Rf + β*(Rm - Rf)  — use historical mu as market premium proxy
        market_premium = mu_annual - cfg.risk_free_rate
        mu_annual      = cfg.risk_free_rate + nifty_beta * market_premium
        mu_daily       = mu_annual / 252

    logger.info(
        f"Parameters → μ_annual={mu_annual:.4f} | σ_annual={sigma_annual:.4f} "
        f"| rolling σ_daily={sigma_daily:.6f}"
    )
    return mu_daily, sigma_daily


def _gbm_paths(
    S0      : float,
    mu      : float,
    sigma   : float,
    T       : int,
    N       : int,
    cfg     : SimConfig,
    rng     : np.random.Generator,
) -> np.ndarray:
    """
    Vectorised GBM path generation.
    S_t = S_{t-1} * exp( (μ - σ²/2)*dt  +  σ*√dt*Z )
    Returns shape (T, N).
    """
    dt = 1.0  # daily step

    if cfg.use_fat_tails:
        # Student-t shocks → fatter tails, more realistic for equity
        from scipy.stats import t as student_t
        Z = student_t.rvs(df=cfg.t_dof, size=(T, N), random_state=rng.integers(0, 2**31))
        # Rescale so variance matches σ²  (student-t var = dof/(dof-2))
        scale = np.sqrt((cfg.t_dof - 2) / cfg.t_dof)
        Z = Z * scale
    else:
        Z = rng.standard_normal((T, N))

    drift     = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_increments = drift + diffusion                    # (T, N)
    log_paths      = np.cumsum(log_increments, axis=0)   # cumulative log returns
    paths          = S0 * np.exp(log_paths)               # price paths
    return paths


def run_scenarios(
    S0          : float,
    mu_base     : float,
    sigma_base  : float,
    cfg         : SimConfig,
    rng         : np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Bull / Base / Bear scenario paths (1000 paths each for speed)."""
    n_scen = min(1000, cfg.n_simulations)
    T      = cfg.horizon_days
    scenarios = {}

    # Base
    scenarios["base"] = _gbm_paths(S0, mu_base, sigma_base, T, n_scen, cfg, rng)

    # Bull: +30% drift, -20% vol
    scenarios["bull"] = _gbm_paths(
        S0, mu_base * 1.30, sigma_base * 0.80, T, n_scen, cfg, rng
    )

    # Bear: -30% drift, +40% vol
    scenarios["bear"] = _gbm_paths(
        S0, mu_base * 0.70, sigma_base * 1.40, T, n_scen, cfg, rng
    )

    return scenarios


def analyse(results: SimResults) -> SimResults:
    """Compute all risk/return metrics from simulation paths."""
    final = results.paths[-1, :]          # terminal prices (N,)
    results.final_prices  = final
    results.mean_price    = float(np.mean(final))
    results.median_price  = float(np.median(final))
    results.ci_5          = float(np.percentile(final, 5))
    results.ci_50         = float(np.percentile(final, 50))
    results.ci_95         = float(np.percentile(final, 95))

    # Returns relative to S0
    pnl = (final - results.S0) / results.S0

    # VaR = worst loss at confidence level (expressed as % of S0)
    results.var_95  = float(-np.percentile(pnl, 5)  * results.S0)   # ₹ loss
    results.var_99  = float(-np.percentile(pnl, 1)  * results.S0)
    results.cvar_95 = float(-pnl[pnl <= np.percentile(pnl, 5)].mean() * results.S0)

    results.prob_increase   = float(np.mean(final > results.S0))
    results.prob_10pct_up   = float(np.mean(final > results.S0 * 1.10))
    results.prob_10pct_down = float(np.mean(final < results.S0 * 0.90))

    logger.info(
        f"Results → E[S_T]={results.mean_price:.2f}  "
        f"CI[5%,95%]=[{results.ci_5:.2f}, {results.ci_95:.2f}]  "
        f"VaR95=₹{results.var_95:.2f}  P(↑)={results.prob_increase:.2%}"
    )
    return results


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def run_simulation(
    log_returns : pd.Series,
    current_price: float,
    cfg         : SimConfig | None = None,
) -> SimResults:
    """
    Full simulation pipeline.
    Returns a populated SimResults object ready for visualisation.
    """
    if cfg is None:
        cfg = SimConfig()

    rng = np.random.default_rng(cfg.random_seed)

    mu, sigma = estimate_params(log_returns, cfg)

    logger.info(
        f"Running {cfg.n_simulations:,} simulations  "
        f"| horizon={cfg.horizon_days} days  "
        f"| fat_tails={cfg.use_fat_tails}"
    )

    paths = _gbm_paths(
        S0=current_price,
        mu=mu,
        sigma=sigma,
        T=cfg.horizon_days,
        N=cfg.n_simulations,
        cfg=cfg,
        rng=rng,
    )

    scenarios = run_scenarios(current_price, mu, sigma, cfg, rng)

    results = SimResults(
        paths=paths,
        S0=current_price,
        config=cfg,
        mu=mu,
        sigma=sigma,
        scenarios=scenarios,
    )
    return analyse(results)


if __name__ == "__main__":
    # Quick smoke-test with synthetic data
    np.random.seed(0)
    synthetic_returns = pd.Series(np.random.normal(0.0003, 0.015, 2520))
    res = run_simulation(synthetic_returns, current_price=1600.0)
    print(f"Mean terminal price : ₹{res.mean_price:.2f}")
    print(f"VaR 95%             : ₹{res.var_95:.2f}")
    print(f"P(price goes up)    : {res.prob_increase:.2%}")
