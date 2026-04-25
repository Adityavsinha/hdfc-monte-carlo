"""
scripts/quant_engine.py
-----------------------
QuantEdge Analytics — Phase 1 Quant Engine

Improvements over baseline:
  1. Fama-French 3-Factor drift (replaces basic CAPM)
  2. HMM Market Regime Detection (bull/bear overlay)
  3. Earnings Risk Flagging (widens CI near earnings)
  4. News Sentiment Score (Google RSS + VADER)
  5. Correlated Monte Carlo (Cholesky — portfolio VaR)
  6. Enhanced signal scoring (incorporates all new factors)
"""
import numpy as np
import pandas as pd
import warnings
import logging
from scipy.stats import t as student_t
from config import (
    N_SIMULATIONS, HORIZON_DAYS, RISK_FREE_RATE, T_DOF,
    BUY_PROB_THRESHOLD, RISKY_VAR_THRESHOLD, HOLD_SHARPE_MIN,
    MARKET_RETURN_ANNUAL, EQUITY_PREMIUM, MIN_MU_ANNUAL, MAX_MU_ANNUAL,
    SMB_ANNUAL_PREMIUM, HML_ANNUAL_PREMIUM, MOM_ANNUAL_PREMIUM,
    EARNINGS_WINDOW_DAYS, EARNINGS_VOL_MULT, FF3_LOOKBACK_DAYS,
    CORR_MC_N, CORR_MC_MIN_STOCKS,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════
#  1. FAMA-FRENCH 3-FACTOR DRIFT MODEL
# ════════════════════════════════════════════════════════

def compute_nifty_beta(stock_returns: pd.Series, nifty_returns: pd.Series) -> float:
    """Beta vs Nifty 50 index from covariance."""
    try:
        df = pd.DataFrame({"s": stock_returns, "n": nifty_returns}).dropna()
        if len(df) < 60:
            return 1.0
        df = df.iloc[-504:]
        cov = df.cov()
        beta = float(cov.loc["s", "n"] / cov.loc["n", "n"])
        return float(np.clip(beta, 0.2, 2.5))
    except Exception:
        return 1.0


def compute_ff3_loadings(
    stock_returns: pd.Series,
    market_excess: pd.Series,
    smb_returns:   pd.Series,
    hml_returns:   pd.Series,
) -> dict:
    """
    Run OLS regression of stock excess returns on 3 FF factors.
    Returns betas for market, SMB, HML plus R².
    Falls back to CAPM if factors unavailable.
    """
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        # Fallback: CAPM-only
        beta_m = compute_nifty_beta(stock_returns, market_excess)
        return {"b_market": beta_m, "b_smb": 0.0, "b_hml": 0.0, "r2": 0.0, "method": "CAPM"}

    try:
        # Align all series on common dates
        df = pd.DataFrame({
            "stock" : stock_returns,
            "mkt"   : market_excess,
            "smb"   : smb_returns,
            "hml"   : hml_returns,
        }).dropna().iloc[-FF3_LOOKBACK_DAYS:]

        if len(df) < 120:
            beta_m = compute_nifty_beta(stock_returns, market_excess)
            return {"b_market": beta_m, "b_smb": 0.0, "b_hml": 0.0, "r2": 0.0, "method": "CAPM-fallback"}

        X = df[["mkt", "smb", "hml"]].values
        y = df["stock"].values

        reg = LinearRegression(fit_intercept=True).fit(X, y)
        b_m, b_smb, b_hml = reg.coef_
        r2 = float(reg.score(X, y))

        return {
            "b_market": float(np.clip(b_m,   0.1, 2.5)),
            "b_smb"   : float(np.clip(b_smb, -1.0, 2.0)),
            "b_hml"   : float(np.clip(b_hml, -1.0, 2.0)),
            "r2"      : round(r2, 3),
            "method"  : "FF3",
        }
    except Exception as e:
        logger.debug(f"FF3 regression failed: {e}")
        beta_m = compute_nifty_beta(stock_returns, market_excess)
        return {"b_market": beta_m, "b_smb": 0.0, "b_hml": 0.0, "r2": 0.0, "method": "CAPM-err"}


def estimate_drift_ff3(
    log_returns:    pd.Series,
    market_excess:  pd.Series = None,
    smb_returns:    pd.Series = None,
    hml_returns:    pd.Series = None,
    regime_adj:     float = 0.0,
) -> tuple[float, float, str, dict]:
    """
    Compute expected daily drift using Fama-French 3-Factor model.
    Falls back to CAPM if factor data not available.

    Returns:
        (mu_daily, beta_nifty, method_str, ff3_loadings_dict)
    """
    # ── Factor regression ───────────────────────
    if (market_excess is not None and smb_returns is not None
            and hml_returns is not None):
        ff3 = compute_ff3_loadings(log_returns, market_excess, smb_returns, hml_returns)
    else:
        beta_m = compute_nifty_beta(log_returns, market_excess) if market_excess is not None else 1.0
        ff3 = {"b_market": beta_m, "b_smb": 0.0, "b_hml": 0.0, "r2": 0.0, "method": "CAPM"}

    b_m   = ff3["b_market"]
    b_smb = ff3["b_smb"]
    b_hml = ff3["b_hml"]

    # ── Expected annual return ─────────────────
    mu_annual = (
        RISK_FREE_RATE
        + b_m   * EQUITY_PREMIUM
        + b_smb * SMB_ANNUAL_PREMIUM
        + b_hml * HML_ANNUAL_PREMIUM
    )

    # ── Momentum overlay (short-term) ──────────
    if len(log_returns) >= 252:
        mom3  = float(log_returns.iloc[-63:].mean())
        mom12 = float(log_returns.iloc[-252:].mean())
        mom_overlay = float(np.clip((mom3 - mom12) * 0.25, -0.0002, 0.0002))
    else:
        mom_overlay = 0.0

    # ── Regime overlay ──────────────────────────
    mu_daily  = mu_annual / 252 + mom_overlay + regime_adj
    mu_annual = float(np.clip(mu_daily * 252, MIN_MU_ANNUAL, MAX_MU_ANNUAL))
    mu_daily  = mu_annual / 252

    method = (
        f"{ff3['method']}(β={b_m:.2f},SMB={b_smb:.2f},HML={b_hml:.2f})"
        f"+Mom+Regime → {mu_annual:.2%}/yr"
    )

    logger.debug(f"    {method}")
    return float(mu_daily), float(b_m), method, ff3


# ════════════════════════════════════════════════════════
#  2. GARCH VOLATILITY
# ════════════════════════════════════════════════════════

def garch_volatility(log_returns: pd.Series, ewma_sigma: float) -> float:
    """GARCH(1,1)-t blended 60/40 with EWMA. Falls back cleanly."""
    try:
        from arch import arch_model
        series = log_returns.iloc[-504:].dropna() * 100
        if len(series) < 100:
            return ewma_sigma
        model  = arch_model(series, vol="Garch", p=1, q=1,
                            dist="StudentsT", rescale=False)
        result = model.fit(disp="off", show_warning=False)
        fc     = result.forecast(horizon=1)
        sg     = float(np.sqrt(fc.variance.values[-1, 0]) / 100)
        if sg < 0.004 or sg > 0.07:
            return ewma_sigma
        return float(0.6 * sg + 0.4 * ewma_sigma)
    except Exception:
        return ewma_sigma


# ════════════════════════════════════════════════════════
#  3. MARKET REGIME DETECTION
# ════════════════════════════════════════════════════════

def detect_market_regime(nifty_returns: pd.Series) -> dict:
    """
    Detects current market regime (Bull / Bear / Sideways).

    Tries HMM (hmmlearn) first — falls back to rolling-window rule.
    Returns regime info and daily drift adjustment.
    """
    if nifty_returns is None or len(nifty_returns) < 60:
        return _default_regime()

    # ── Try HMM first ────────────────────────
    try:
        regime = _hmm_regime(nifty_returns)
        logger.info(f"  Regime (HMM): {regime['regime']} | Bull prob: {regime['bull_prob']:.1%}")
        return regime
    except Exception as e:
        logger.debug(f"  HMM failed ({e}), using rolling-window regime")

    # ── Fallback: rolling-window rule ────────
    return _rolling_regime(nifty_returns)


def _hmm_regime(nifty_returns: pd.Series) -> dict:
    """2-state Gaussian HMM on Nifty 50 daily returns."""
    from hmmlearn import hmm

    returns = nifty_returns.values[-756:].reshape(-1, 1)  # 3 years max
    model   = hmm.GaussianHMM(n_components=2, covariance_type="full",
                               n_iter=200, random_state=42)
    model.fit(returns)

    states      = model.predict(returns)
    state_means = [float(model.means_[i, 0]) for i in range(2)]
    bull_state  = int(np.argmax(state_means))
    bear_state  = 1 - bull_state

    current_state = int(states[-1])
    proba         = model.predict_proba(returns)[-1]
    bull_prob     = float(proba[bull_state])

    regime_str    = "Bull" if current_state == bull_state else "Bear"

    # Smooth regime: use prob as blend weight
    drift_adj = (
        bull_prob       * (MARKET_RETURN_ANNUAL * 0.02 / 252)     # bull boost
        + (1 - bull_prob) * (-MARKET_RETURN_ANNUAL * 0.025 / 252)  # bear penalty
    )

    return {
        "regime"          : regime_str,
        "bull_prob"       : round(bull_prob, 3),
        "bear_prob"       : round(1 - bull_prob, 3),
        "drift_adjustment": round(float(drift_adj), 6),
        "method"          : "HMM-2state",
        "mean_bull"       : round(state_means[bull_state] * 252 * 100, 2),
        "mean_bear"       : round(state_means[bear_state] * 252 * 100, 2),
    }


def _rolling_regime(nifty_returns: pd.Series) -> dict:
    """Simple rolling-window regime: bull if recent returns positive & vol low."""
    r60  = float(nifty_returns.iloc[-60:].mean()  * 252)  # annualised
    v60  = float(nifty_returns.iloc[-60:].std()   * np.sqrt(252))
    r20  = float(nifty_returns.iloc[-20:].mean()  * 252)

    # Regime scores
    if r60 > 0.05 and v60 < 0.22:
        regime     = "Bull"
        bull_prob  = 0.75
        drift_adj  = +0.0001
    elif r60 < -0.05 or v60 > 0.32:
        regime     = "Bear"
        bull_prob  = 0.25
        drift_adj  = -0.0001
    else:
        regime     = "Sideways"
        bull_prob  = 0.50
        drift_adj  = 0.0

    return {
        "regime"          : regime,
        "bull_prob"       : bull_prob,
        "bear_prob"       : 1 - bull_prob,
        "drift_adjustment": drift_adj,
        "method"          : "Rolling-window",
        "mean_bull"       : round(r60 * 100, 2),
        "mean_bear"       : round(r20 * 100, 2),
    }


def _default_regime() -> dict:
    return {
        "regime": "Unknown", "bull_prob": 0.5, "bear_prob": 0.5,
        "drift_adjustment": 0.0, "method": "default",
        "mean_bull": 0.0, "mean_bear": 0.0,
    }


# ════════════════════════════════════════════════════════
#  4. EARNINGS RISK FLAGGING
# ════════════════════════════════════════════════════════

def get_earnings_risk_flag(symbol: str) -> dict:
    """
    Checks if earnings are due in next EARNINGS_WINDOW_DAYS days.
    Widens volatility estimate near earnings.
    """
    base = {"has_earnings_soon": False, "vol_multiplier": 1.0,
            "days_to_earnings": None, "avg_earnings_move_pct": None}
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        cal    = ticker.calendar

        if cal is None:
            return base

        # calendar can be a dict or DataFrame depending on yfinance version
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            next_earn = pd.Timestamp(cal.columns[0])
        elif isinstance(cal, dict) and "Earnings Date" in cal:
            earn_dates = cal["Earnings Date"]
            if isinstance(earn_dates, list) and earn_dates:
                next_earn = pd.Timestamp(earn_dates[0])
            else:
                return base
        else:
            return base

        days_away = (next_earn - pd.Timestamp.today()).days
        if not (0 < days_away <= EARNINGS_WINDOW_DAYS):
            return base

        # Historical earnings move magnitude
        try:
            hist = ticker.earnings_history
            if hist is not None and "surprisePercent" in hist.columns:
                avg_move = float(hist["surprisePercent"].abs().dropna().mean())
                vol_mult = 1.0 + min(avg_move / 100, 0.60)   # cap at +60%
            else:
                avg_move = 7.0
                vol_mult = EARNINGS_VOL_MULT
        except Exception:
            avg_move = 7.0
            vol_mult = EARNINGS_VOL_MULT

        logger.info(f"    Earnings in {days_away}d → vol×{vol_mult:.2f}")
        return {
            "has_earnings_soon"    : True,
            "days_to_earnings"     : int(days_away),
            "vol_multiplier"       : round(vol_mult, 3),
            "avg_earnings_move_pct": round(avg_move, 2),
        }

    except Exception as e:
        logger.debug(f"  Earnings flag failed for {symbol}: {e}")
        return base


# ════════════════════════════════════════════════════════
#  5. NEWS SENTIMENT (Google RSS + VADER)
# ════════════════════════════════════════════════════════

def get_sentiment_score(symbol: str, company_name: str) -> dict:
    """
    Fetches recent news from Google News RSS and scores sentiment via VADER.
    Returns compound score in [-1, +1] and label.
    """
    base = {"sentiment_score": 0.0, "sentiment_label": "Neutral",
            "news_count": 0, "sentiment_source": "none"}
    try:
        import feedparser
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        import socket
        socket.setdefaulttimeout(6)

        query   = f"{company_name} stock NSE India".replace(" ", "+")
        rss_url = (f"https://news.google.com/rss/search?q={query}"
                   f"&hl=en-IN&gl=IN&ceid=IN:en")

        feed    = feedparser.parse(rss_url)
        entries = (feed.entries or [])[:12]

        if not entries:
            return base

        analyzer = SentimentIntensityAnalyzer()
        scores   = []
        for entry in entries:
            text  = entry.get("title", "") + " " + entry.get("summary", "")
            score = analyzer.polarity_scores(text)["compound"]
            scores.append(score)

        avg = float(np.mean(scores))
        label = (
            "Positive" if avg > 0.12  else
            "Negative" if avg < -0.12 else
            "Neutral"
        )

        return {
            "sentiment_score" : round(avg, 4),
            "sentiment_label" : label,
            "news_count"      : len(scores),
            "sentiment_source": "google_rss_vader",
        }

    except ImportError:
        logger.debug("  feedparser/vaderSentiment not installed — skipping sentiment")
        return base
    except Exception as e:
        logger.debug(f"  Sentiment failed for {symbol}: {e}")
        return base


# ════════════════════════════════════════════════════════
#  6. INDIVIDUAL MONTE CARLO SIMULATION
# ════════════════════════════════════════════════════════

def run_monte_carlo(
    S0:    float,
    mu:    float,
    sigma: float,
    T:     int   = HORIZON_DAYS,
    N:     int   = N_SIMULATIONS,
    seed:  int   = 42,
) -> np.ndarray:
    """GBM with Student-t fat tails. Returns (T × N) path matrix."""
    rng   = np.random.default_rng(seed)
    Z     = student_t.rvs(df=T_DOF, size=(T, N), random_state=seed)
    Z     = Z * np.sqrt((T_DOF - 2) / T_DOF)   # normalise to unit variance
    drift = (mu - 0.5 * sigma ** 2)
    paths = S0 * np.exp(np.cumsum(drift + sigma * Z, axis=0))
    return paths


def extract_path_percentiles(paths: np.ndarray, step: int = 5) -> dict:
    idx = list(range(0, paths.shape[0], step))
    return {
        "p5" : np.percentile(paths[idx],  5, axis=1).round(2).tolist(),
        "p25": np.percentile(paths[idx], 25, axis=1).round(2).tolist(),
        "p50": np.percentile(paths[idx], 50, axis=1).round(2).tolist(),
        "p75": np.percentile(paths[idx], 75, axis=1).round(2).tolist(),
        "p95": np.percentile(paths[idx], 95, axis=1).round(2).tolist(),
        "days": [i + step for i in idx],
    }


def extract_histogram(final: np.ndarray, bins: int = 50) -> dict:
    counts, edges = np.histogram(final, bins=bins)
    return {
        "labels": ((edges[:-1] + edges[1:]) / 2).round(2).tolist(),
        "counts": counts.tolist(),
    }


# ════════════════════════════════════════════════════════
#  7. CORRELATED MONTE CARLO (PORTFOLIO LEVEL)
# ════════════════════════════════════════════════════════

def run_correlated_mc(
    stocks_data: dict,      # {sym: {"S0": float, "mu": float, "sigma": float, "log_ret": Series}}
    N:           int = CORR_MC_N,
    T:           int = HORIZON_DAYS,
    seed:        int = 99,
) -> dict:
    """
    Simulate all stocks jointly using their historical return correlation.
    Uses Cholesky decomposition for correlated Student-t shocks.

    Returns:
        {
          "portfolio_var_95"       : float,   # % portfolio loss at 95%
          "portfolio_var_99"       : float,
          "portfolio_cvar_95"      : float,
          "correlation_matrix"     : dict,    # {sym: {sym: corr}}
          "corr_symbols"           : list,
          "portfolio_expected_ret" : float,
        }
    """
    symbols = list(stocks_data.keys())
    n       = len(symbols)

    if n < CORR_MC_MIN_STOCKS:
        logger.warning(f"  Correlated MC: only {n} stocks, skipping")
        return {}

    try:
        # ── Build return matrix ─────────────────────
        returns_list = []
        valid_syms   = []
        for sym in symbols:
            r = stocks_data[sym].get("log_ret")
            if r is not None and len(r) >= 252:
                returns_list.append(r.values[-504:])   # 2 years
                valid_syms.append(sym)

        if len(valid_syms) < CORR_MC_MIN_STOCKS:
            return {}

        # Align lengths
        min_len   = min(len(r) for r in returns_list)
        ret_matrix = np.column_stack([r[-min_len:] for r in returns_list])  # (T_hist × n)

        # ── Covariance + Cholesky ───────────────────
        cov = np.cov(ret_matrix.T)           # (n × n)
        # Regularise to ensure positive definite
        cov += 1e-8 * np.eye(len(valid_syms))

        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Fallback: nearest PD matrix
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-8)
            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L   = np.linalg.cholesky(cov)

        # ── Simulate correlated paths ───────────────
        rng    = np.random.default_rng(seed)
        Z_ind  = student_t.rvs(df=T_DOF, size=(T, N, len(valid_syms)),
                               random_state=seed)
        Z_ind  = Z_ind * np.sqrt((T_DOF - 2) / T_DOF)
        Z_cor  = Z_ind @ L.T                  # (T × N × n)

        # Equal-weight portfolio PnL
        port_pnl_paths = []
        for i, sym in enumerate(valid_syms):
            S0    = stocks_data[sym]["S0"]
            mu    = stocks_data[sym]["mu"]
            sigma = stocks_data[sym]["sigma"]
            shocks = (mu - 0.5 * sigma**2) + sigma * Z_cor[:, :, i]
            term   = S0 * np.exp(np.sum(shocks, axis=0))   # terminal price (N,)
            ret    = (term - S0) / S0                       # terminal return
            port_pnl_paths.append(ret)

        port_ret     = np.mean(port_pnl_paths, axis=0)    # equal weight
        var_95       = float(-np.percentile(port_ret, 5))
        var_99       = float(-np.percentile(port_ret, 1))
        cvar_95      = float(-port_ret[port_ret <= np.percentile(port_ret, 5)].mean())
        expected_ret = float(np.mean(port_ret))

        # ── Correlation matrix ──────────────────────
        corr_matrix = np.corrcoef(ret_matrix.T)
        corr_dict   = {}
        for i, si in enumerate(valid_syms):
            corr_dict[si] = {}
            for j, sj in enumerate(valid_syms):
                corr_dict[si][sj] = round(float(corr_matrix[i, j]), 3)

        logger.info(
            f"  Correlated MC ({len(valid_syms)} stocks): "
            f"Port VaR95={var_95:.1%} | E[Ret]={expected_ret:+.1%}"
        )

        return {
            "portfolio_var_95"       : round(var_95, 4),
            "portfolio_var_99"       : round(var_99, 4),
            "portfolio_cvar_95"      : round(cvar_95, 4),
            "portfolio_expected_ret" : round(expected_ret, 4),
            "correlation_matrix"     : corr_dict,
            "corr_symbols"           : valid_syms,
            "n_simulations"          : N,
        }

    except Exception as e:
        logger.warning(f"  Correlated MC failed: {e}")
        return {}


# ════════════════════════════════════════════════════════
#  8. RISK METRICS
# ════════════════════════════════════════════════════════

def compute_risk_metrics(
    final:      np.ndarray,
    S0:         float,
    mu_daily:   float,
    sigma_daily: float,
    log_returns: pd.Series,
) -> dict:
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

    return {
        "var_95"      : round(var_95,   4),
        "var_99"      : round(var_99,   4),
        "cvar_95"     : round(cvar_95,  4),
        "sharpe"      : round(sharpe,   3),
        "sortino"     : round(sortino,  3),
        "calmar"      : round(calmar,   3),
        "max_drawdown": round(max_dd,   4),
        "mu_annual"   : round(mu_ann,   4),
        "sigma_annual": round(sigma_ann,4),
    }


# ════════════════════════════════════════════════════════
#  9. SIGNAL GENERATOR (Phase 1 — incorporates all factors)
# ════════════════════════════════════════════════════════

def generate_signal(
    prob_up:        float,
    var_95:         float,
    sharpe:         float,
    expected_ret:   float,
    mispricing:     float,
    beta:           float,
    sentiment_score: float = 0.0,
    regime:          str   = "Unknown",
    has_earnings:    bool  = False,
    tech_score:      int   = 0,
    fundamental_score: int = 0,
) -> dict:
    """
    Weighted multi-factor signal scoring.
    Phase 1 adds: sentiment, regime, earnings risk, tech+fundamental integration.
    """
    score = 0

    # ── Core MC factors (weight: 10 pts max) ─────────────────
    if prob_up > 0.70:    score += 3
    elif prob_up > 0.60:  score += 2
    elif prob_up > 0.52:  score += 1
    elif prob_up < 0.38:  score -= 3
    elif prob_up < 0.48:  score -= 1

    if sharpe > 1.0:    score += 2
    elif sharpe > 0.5:  score += 1
    elif sharpe < -0.1: score -= 2
    elif sharpe < 0.2:  score -= 1

    if var_95 < 0.08:    score += 2
    elif var_95 < 0.14:  score += 1
    elif var_95 > 0.28:  score -= 2
    elif var_95 > 0.20:  score -= 1

    if expected_ret > 0.15:    score += 2
    elif expected_ret > 0.08:  score += 1
    elif expected_ret < 0:     score -= 2
    elif expected_ret < 0.03:  score -= 1

    if mispricing > 0.10:    score += 1
    elif mispricing < -0.10: score -= 1

    # ── Phase 1: Sentiment overlay (weight: 2 pts max) ───────
    if sentiment_score > 0.25:   score += 2
    elif sentiment_score > 0.10: score += 1
    elif sentiment_score < -0.25: score -= 2
    elif sentiment_score < -0.10: score -= 1

    # ── Phase 1: Regime overlay (weight: 2 pts max) ──────────
    if regime == "Bull":       score += 1
    elif regime == "Bear":     score -= 2
    # Sideways = no adjustment

    # ── Phase 1: Earnings risk penalty ──────────────────────
    if has_earnings:
        score -= 1   # uncertainty around earnings

    # ── Technical integration (weight: 2 pts max) ────────────
    if tech_score >= 5:    score += 2
    elif tech_score >= 2:  score += 1
    elif tech_score <= -4: score -= 2
    elif tech_score <= -2: score -= 1

    # ── Fundamental integration (weight: 2 pts max) ──────────
    if fundamental_score >= 6:    score += 2
    elif fundamental_score >= 3:  score += 1
    elif fundamental_score <= -3: score -= 2
    elif fundamental_score <= 0:  score -= 1

    # ── Beta risk cap ────────────────────────────────────────
    if beta > 1.8:
        score = max(score - 1, score)   # high-beta stocks capped

    # ── Map to signal ────────────────────────────────────────
    if   score >= 6:   signal, color = "STRONG BUY", "#00c853"
    elif score >= 2:   signal, color = "BUY",         "#2ecc8a"
    elif score >= -2:  signal, color = "HOLD",        "#f0b429"
    elif score >= -5:  signal, color = "RISKY",       "#ff9800"
    else:              signal, color = "AVOID",        "#e02d3c"

    # Confidence: normalise score to 0-100
    confidence = int(np.clip((score + 12) / 24 * 100, 0, 100))

    return {
        "signal"       : signal,
        "signal_color" : color,
        "score"        : score,
        "confidence"   : confidence,
    }


# ════════════════════════════════════════════════════════
#  10. FULL STOCK PIPELINE
# ════════════════════════════════════════════════════════

def run_full_pipeline(
    symbol:         str,
    features:       dict,
    fin_info:       dict,
    nifty_returns:  pd.Series = None,
    market_excess:  pd.Series = None,   # Phase 1: Nifty excess returns
    smb_returns:    pd.Series = None,   # Phase 1: FF SMB factor
    hml_returns:    pd.Series = None,   # Phase 1: FF HML factor
    regime_info:    dict      = None,   # Phase 1: from detect_market_regime
    earnings_info:  dict      = None,   # Phase 1: from get_earnings_risk_flag
    sentiment_info: dict      = None,   # Phase 1: from get_sentiment_score
) -> dict | None:
    """
    Phase 1 full pipeline:
      FF3 drift → GARCH vol → regime overlay → earnings risk → MC →
      risk metrics → signal (with sentiment + regime + tech + fundamentals)
    """
    try:
        log_ret    = features["log_returns"]
        S0         = features["current_price"]
        sigma_ewma = features["sigma_daily"]

        if len(log_ret) < 100:
            return None

        # ── Defaults for optional Phase 1 inputs ──────────
        regime_info    = regime_info    or _default_regime()
        earnings_info  = earnings_info  or {"has_earnings_soon": False, "vol_multiplier": 1.0}
        sentiment_info = sentiment_info or {"sentiment_score": 0.0, "sentiment_label": "Neutral", "news_count": 0}

        regime_adj = float(regime_info.get("drift_adjustment", 0.0))

        # ── 1. Drift: Fama-French 3-Factor ────────────────
        # NOTE: cannot use `or` with pandas Series — use explicit None check
        _mkt = (market_excess
                if (market_excess is not None and
                    not getattr(market_excess, 'empty', True) and
                    len(market_excess) > 10)
                else nifty_returns)
        mu, beta, method, ff3 = estimate_drift_ff3(
            log_returns   = log_ret,
            market_excess = _mkt,
            smb_returns   = smb_returns,
            hml_returns   = hml_returns,
            regime_adj    = regime_adj,
        )

        # ── 2. Volatility: GARCH blended ──────────────────
        sigma = garch_volatility(log_ret, sigma_ewma)

        # ── 3. Earnings risk: inflate vol if near earnings ─
        vol_mult = float(earnings_info.get("vol_multiplier", 1.0))
        sigma    = sigma * vol_mult

        # ── 4. Monte Carlo simulation ─────────────────────
        paths = run_monte_carlo(S0, mu, sigma)
        final = paths[-1, :]
        pc    = extract_path_percentiles(paths)
        hist  = extract_histogram(final)

        # ── 5. Risk metrics ───────────────────────────────
        risk  = compute_risk_metrics(final, S0, mu, sigma, log_ret)

        # ── 6. Derived stats ──────────────────────────────
        mean_p    = float(np.mean(final))
        exp_r     = (mean_p / S0) - 1
        misp      = (mean_p - S0) / S0
        pu        = float(np.mean(final > S0))
        p10u      = float(np.mean(final > S0 * 1.10))
        p20u      = float(np.mean(final > S0 * 1.20))
        p10d      = float(np.mean(final < S0 * 0.90))

        # Percentile prices
        ci_5,  ci_25 = float(np.percentile(final, 5)),  float(np.percentile(final, 25))
        ci_75, ci_95 = float(np.percentile(final, 75)), float(np.percentile(final, 95))
        med_p = float(np.median(final))

        # ── 7. Scenario medians (quick MC, 500 paths) ─────
        def scenario_median(mu_mult, sigma_mult):
            return float(np.median(
                run_monte_carlo(S0, mu * mu_mult, sigma * sigma_mult, N=500)[-1, :]
            ))

        bull_med = scenario_median(1.20, 0.85)
        base_med = scenario_median(1.00, 1.00)
        bear_med = scenario_median(0.60, 1.30)

        # ── 8. Fundamental score (pre-computed from fin_info) ──
        fund_result   = compute_fundamental_score(fin_info, risk)
        fund_score    = fund_result["fundamental_score"]

        # ── 9. Signal generation ──────────────────────────
        sig = generate_signal(
            prob_up           = pu,
            var_95            = risk["var_95"],
            sharpe            = risk["sharpe"],
            expected_ret      = exp_r,
            mispricing        = misp,
            beta              = beta,
            sentiment_score   = float(sentiment_info.get("sentiment_score", 0)),
            regime            = regime_info.get("regime", "Unknown"),
            has_earnings      = bool(earnings_info.get("has_earnings_soon", False)),
            tech_score        = 0,   # filled in by run_pipeline after tech analysis
            fundamental_score = fund_score,
        )

        logger.info(
            f"  {symbol}: ₹{S0:,.0f}→E[₹{mean_p:,.0f}] ({exp_r:+.1%}) | "
            f"{sig['signal']} | P(↑):{pu:.1%} | Sharpe:{risk['sharpe']:.2f} | "
            f"β={beta:.2f} | Regime:{regime_info.get('regime','?')} | "
            f"Sentiment:{sentiment_info.get('sentiment_label','?')}"
        )

        return {
            # Identity
            "symbol"       : symbol,
            "name"         : fin_info.get("name", symbol),
            "sector"       : fin_info.get("sector", "Other"),
            "industry"     : fin_info.get("industry", ""),
            "description"  : fin_info.get("description", ""),
            "website"      : fin_info.get("website", ""),
            # Price & horizon
            "price"        : round(S0, 2),
            "week52_high"  : round(features["week52_high"], 2),
            "week52_low"   : round(features["week52_low"],  2),
            "n_simulations": N_SIMULATIONS,
            "horizon_days" : HORIZON_DAYS,
            # Fundamentals (raw)
            "market_cap"   : fin_info.get("market_cap"),
            "pe_ratio"     : fin_info.get("pe_ratio"),
            "pb_ratio"     : fin_info.get("pb_ratio"),
            "eps"          : fin_info.get("eps"),
            "revenue"      : fin_info.get("revenue"),
            "net_income"   : fin_info.get("net_income"),
            "roe"          : fin_info.get("roe"),
            "roa"          : fin_info.get("roa"),
            "debt_equity"  : fin_info.get("debt_equity"),
            "current_ratio": fin_info.get("current_ratio"),
            "dividend_yield": fin_info.get("dividend_yield"),
            "book_value"   : fin_info.get("book_value"),
            "employees"    : fin_info.get("employees"),
            # MC output
            "mean_price"   : round(mean_p, 2),
            "median_price" : round(med_p,  2),
            "ci_5"         : round(ci_5,   2),
            "ci_25"        : round(ci_25,  2),
            "ci_75"        : round(ci_75,  2),
            "ci_95"        : round(ci_95,  2),
            "expected_return_pct" : round(exp_r * 100, 2),
            "expected_return"     : round(exp_r, 4),
            "mispricing_pct"      : round(misp * 100, 2),
            "prob_up"    : round(pu,   4),
            "prob_10up"  : round(p10u, 4),
            "prob_20up"  : round(p20u, 4),
            "prob_10down": round(p10d, 4),
            # Scenarios
            "bull_median": round(bull_med, 2),
            "base_median": round(base_med, 2),
            "bear_median": round(bear_med, 2),
            # Risk
            **risk,
            # Signal
            **sig,
            # Model metadata
            "beta_nifty"   : round(beta, 3),
            "mu_annual"    : round(mu * 252, 4),
            "sigma_daily"  : round(sigma, 6),
            "drift_method" : method,
            "ff3_b_market" : round(ff3["b_market"], 3),
            "ff3_b_smb"    : round(ff3["b_smb"],   3),
            "ff3_b_hml"    : round(ff3["b_hml"],   3),
            "ff3_r2"       : round(ff3.get("r2", 0), 3),
            # Phase 1 enrichments
            "regime"              : regime_info.get("regime", "Unknown"),
            "regime_bull_prob"    : regime_info.get("bull_prob", 0.5),
            "regime_method"       : regime_info.get("method", ""),
            "sentiment_score"     : sentiment_info.get("sentiment_score", 0),
            "sentiment_label"     : sentiment_info.get("sentiment_label", "Neutral"),
            "news_count"          : sentiment_info.get("news_count", 0),
            "has_earnings_soon"   : earnings_info.get("has_earnings_soon", False),
            "days_to_earnings"    : earnings_info.get("days_to_earnings"),
            "earnings_vol_mult"   : earnings_info.get("vol_multiplier", 1.0),
            # Momentum (from features)
            "mom_1m"   : round(features.get("mom_1m", 0), 4),
            "mom_3m"   : round(features.get("mom_3m", 0), 4),
            "mom_6m"   : round(features.get("mom_6m", 0), 4),
            "mom_1y"   : round(features.get("mom_1y", 0), 4),
            # Fundamental grade (filled in by pipeline)
            **fund_result,
            # Charts (heavy — separated into charts.json by pipeline)
            "path_charts": pc,
            "histogram"  : hist,
            "model"      : f"FF3+GARCH+MC({N_SIMULATIONS:,}sim,T={HORIZON_DAYS}d)",
        }

    except Exception as e:
        logger.error(f"  {symbol}: pipeline failed — {e}", exc_info=True)
        return None


# ════════════════════════════════════════════════════════
#  TECHNICAL ANALYSIS ENGINE (unchanged from baseline)
# ════════════════════════════════════════════════════════

def compute_technical_indicators(df: pd.DataFrame) -> dict:
    """Computes RSI, MACD, Bollinger, Stochastic, Moving Averages, ATR, Volume."""
    try:
        close  = df["Close"].squeeze().astype(float)
        high   = df["High"].squeeze().astype(float)   if "High"   in df.columns else close
        low    = df["Low"].squeeze().astype(float)    if "Low"    in df.columns else close
        volume = df["Volume"].squeeze().astype(float) if "Volume" in df.columns else None
        result = {}
        cur_p  = float(close.iloc[-1])

        # RSI (14)
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / loss.replace(0, 1e-10)
        rsi    = float((100 - 100 / (1 + rs)).iloc[-1])
        result["rsi_14"]       = round(rsi, 2)
        result["rsi_signal"]   = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"

        # MACD (12,26,9)
        ema12  = close.ewm(span=12).mean()
        ema26  = close.ewm(span=26).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        hist   = macd - signal
        result["macd"]         = round(float(macd.iloc[-1]),   4)
        result["macd_signal"]  = round(float(signal.iloc[-1]), 4)
        result["macd_hist"]    = round(float(hist.iloc[-1]),   4)
        result["macd_cross"]   = "Bullish" if float(macd.iloc[-1]) > float(signal.iloc[-1]) else "Bearish"

        # Bollinger Bands (20,2)
        sma20  = close.rolling(20).mean()
        std20  = close.rolling(20).std()
        bb_up  = float((sma20 + 2 * std20).iloc[-1])
        bb_mid = float(sma20.iloc[-1])
        bb_low = float((sma20 - 2 * std20).iloc[-1])
        bb_pct = (cur_p - bb_low) / (bb_up - bb_low) if bb_up != bb_low else 0.5
        result["bb_upper"]     = round(bb_up,  2)
        result["bb_middle"]    = round(bb_mid, 2)
        result["bb_lower"]     = round(bb_low, 2)
        result["bb_position"]  = round(float(bb_pct), 3)
        result["bb_signal"]    = "Near Upper" if bb_pct > 0.8 else "Near Lower" if bb_pct < 0.2 else "Middle"

        # Moving Averages
        sma_50  = float(close.rolling(50).mean().iloc[-1])  if len(close) >= 50  else None
        sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        ema_20  = float(close.ewm(span=20).mean().iloc[-1])
        result["sma_50"]       = round(sma_50,  2) if sma_50  else None
        result["sma_200"]      = round(sma_200, 2) if sma_200 else None
        result["ema_20"]       = round(ema_20,  2)
        result["above_sma50"]  = bool(cur_p > sma_50)  if sma_50  else None
        result["above_sma200"] = bool(cur_p > sma_200) if sma_200 else None
        result["golden_cross"] = bool(sma_50 > sma_200) if (sma_50 and sma_200) else None

        # ATR (14)
        tr1  = high - low
        tr2  = (high - close.shift()).abs()
        tr3  = (low  - close.shift()).abs()
        atr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
        result["atr_14"]       = round(float(atr.iloc[-1]), 2)
        result["atr_pct"]      = round(float(atr.iloc[-1]) / cur_p * 100, 2)

        # Stochastic (14,3)
        lo14 = low.rolling(14).min()
        hi14 = high.rolling(14).max()
        k    = 100 * (close - lo14) / (hi14 - lo14 + 1e-10)
        d    = k.rolling(3).mean()
        result["stoch_k"]      = round(float(k.iloc[-1]), 2)
        result["stoch_d"]      = round(float(d.iloc[-1]), 2)
        result["stoch_signal"] = ("Oversold" if float(k.iloc[-1]) < 20
                                  else "Overbought" if float(k.iloc[-1]) > 80
                                  else "Neutral")

        # Volume
        if volume is not None:
            vol_sma20 = volume.rolling(20).mean()
            vr = float(volume.iloc[-1] / vol_sma20.iloc[-1]) if float(vol_sma20.iloc[-1]) > 0 else 1.0
            result["vol_ratio"]  = round(vr, 2)
            result["vol_trend"]  = "Above Average" if vr > 1.2 else "Below Average" if vr < 0.8 else "Normal"

        # Overall technical score
        tech = 0
        if rsi < 35:                                 tech += 2
        elif rsi < 45:                               tech += 1
        elif rsi > 75:                               tech -= 2
        elif rsi > 65:                               tech -= 1
        if result["macd_cross"] == "Bullish":        tech += 2
        else:                                        tech -= 1
        if bb_pct < 0.25:                            tech += 2
        elif bb_pct > 0.85:                          tech -= 2
        if result.get("above_sma50"):                tech += 1
        if result.get("above_sma200"):               tech += 1
        if result.get("golden_cross"):               tech += 1
        if result.get("vol_ratio", 1) > 1.5:        tech += 1

        result["tech_score"]   = tech
        result["tech_signal"]  = (
            "Strong Buy"  if tech >= 5  else
            "Buy"         if tech >= 2  else
            "Neutral"     if tech >= -1 else
            "Sell"        if tech >= -4 else
            "Strong Sell"
        )
        return result

    except Exception as e:
        logger.warning(f"  Technical analysis failed: {e}")
        return {}


# ════════════════════════════════════════════════════════
#  FUNDAMENTAL SCORE (unchanged from baseline)
# ════════════════════════════════════════════════════════

def compute_fundamental_score(fin_info: dict, risk_metrics: dict) -> dict:
    score   = 0
    details = {}

    pe = fin_info.get("pe_ratio")
    if pe:
        if pe < 15:    score += 2; details["pe"] = "Cheap"
        elif pe < 25:  score += 1; details["pe"] = "Fair"
        elif pe < 40:  score -= 1; details["pe"] = "Expensive"
        else:          score -= 2; details["pe"] = "Very Expensive"

    pb = fin_info.get("pb_ratio")
    if pb:
        if pb < 1.5:   score += 2; details["pb"] = "Undervalued"
        elif pb < 3:   score += 1; details["pb"] = "Fair"
        elif pb < 6:   score -= 1; details["pb"] = "Premium"
        else:          score -= 2; details["pb"] = "Expensive"

    roe = fin_info.get("roe")
    if roe:
        roe_pct = roe * 100
        if roe_pct > 20:   score += 2; details["roe"] = "Excellent"
        elif roe_pct > 12: score += 1; details["roe"] = "Good"
        elif roe_pct > 5:  score -= 1; details["roe"] = "Below Average"
        else:              score -= 2; details["roe"] = "Poor"

    de = fin_info.get("debt_equity")
    if de is not None:
        if de < 0.3:   score += 2; details["debt"] = "Low Debt"
        elif de < 1.0: score += 1; details["debt"] = "Moderate"
        elif de < 2.0: score -= 1; details["debt"] = "High Debt"
        else:          score -= 2; details["debt"] = "Very High Debt"

    dy = fin_info.get("dividend_yield")
    if dy:
        dy_pct = dy * 100
        if dy_pct > 3:   score += 1; details["dividend"] = "High Yield"
        elif dy_pct > 1: details["dividend"] = "Moderate Yield"

    sh = risk_metrics.get("sharpe", 0)
    if sh > 1.0:   score += 2; details["risk_adj"] = "Excellent"
    elif sh > 0.5: score += 1; details["risk_adj"] = "Good"
    elif sh < 0:   score -= 1; details["risk_adj"] = "Poor"

    grade = (
        "A+" if score >= 8 else "A"  if score >= 6 else
        "B+" if score >= 4 else "B"  if score >= 2 else
        "C+" if score >= 0 else "C"  if score >= -2 else "D"
    )

    return {
        "fundamental_score"  : score,
        "fundamental_grade"  : grade,
        "fundamental_details": details,
    }
