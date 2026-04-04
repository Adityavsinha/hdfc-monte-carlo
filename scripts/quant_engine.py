"""
scripts/quant_engine.py - Fixed with Nifty-50 relative beta
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

MARKET_RETURN_ANNUAL = 0.13
EQUITY_PREMIUM       = MARKET_RETURN_ANNUAL - RISK_FREE_RATE
MIN_MU_ANNUAL        = -0.05
MAX_MU_ANNUAL        = +0.22


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


def run_monte_carlo(S0, mu, sigma, T=HORIZON_DAYS, N=N_SIMULATIONS, seed=42):
    Z     = student_t.rvs(df=T_DOF, size=(T, N), random_state=seed)
    Z     = Z * np.sqrt((T_DOF - 2) / T_DOF)
    paths = S0 * np.exp(np.cumsum((mu - 0.5*sigma**2) + sigma*Z, axis=0))
    return paths


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
            "dividend_yield": fin_info.get("dividend_yield"), "beta_nifty": round(beta,3),
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
