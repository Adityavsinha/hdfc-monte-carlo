"""
scripts/run_pipeline.py — QuantEdge Analytics Phase 1 Daily Pipeline

Fixes in this version:
  - pandas Series `or` bug fixed (use explicit None checks)
  - FF3 factor tickers updated to working Yahoo Finance tickers
  - Nifty 100 universe (50 + Next50) for stock selection
  - Backtest robustness improvements
  - Graceful degradation: any step failure → CAPM fallback, never 0 stocks
"""

import json
import logging
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT  = SCRIPT_DIR.parent
DOCS_DIR   = REPO_ROOT / "docs"
LOGS_DIR   = REPO_ROOT / "logs"
DOCS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"{LOGS_DIR}/pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        ),
    ],
)
logger = logging.getLogger(__name__)

from config import (
    TOP_N_STOCKS, NIFTY50_SYMBOLS, NIFTY_NEXT50, SECTOR_MAP,
    USE_FAMA_FRENCH, USE_REGIME_DETECTION, USE_SENTIMENT,
    USE_EARNINGS_RISK, USE_CORRELATED_MC,
)
from data_ingestion import (
    fetch_live_nifty50, fetch_stock_data,
    fetch_financial_info, compute_features,
)
from quant_engine import (
    run_full_pipeline, compute_technical_indicators,
    compute_fundamental_score, detect_market_regime,
    get_earnings_risk_flag, get_sentiment_score,
    run_correlated_mc, generate_signal,
)
from backtest import store_signals, evaluate_signals, generate_report

# ── Nifty 100 = Nifty 50 + Nifty Next 50 ────────────────────
NIFTY100_SYMBOLS = list(dict.fromkeys(NIFTY50_SYMBOLS + NIFTY_NEXT50))


# ════════════════════════════════════════════════════════
#  INDEX DATA FETCHING
# ════════════════════════════════════════════════════════

def fetch_nifty_returns(years: int = 3) -> pd.Series | None:
    """Fetch Nifty 50 index log-returns for beta + regime computation."""
    import yfinance as yf
    logger.info("  Fetching ^NSEI (Nifty 50 index)...")
    try:
        end   = datetime.today()
        start = end - timedelta(days=years * 365 + 90)
        df    = yf.download("^NSEI", start=start, end=end,
                            auto_adjust=True, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close   = pd.to_numeric(df["Close"].squeeze(), errors="coerce").dropna()
        returns = np.log(close / close.shift(1)).dropna()
        logger.info(f"  Nifty 50: {len(returns)} trading days")
        return returns
    except Exception as e:
        logger.warning(f"  Nifty fetch failed: {e}")
        return None


def _series_safe(s) -> bool:
    """True if s is a non-empty pandas Series with data."""
    if s is None:
        return False
    if isinstance(s, pd.Series):
        return len(s) > 10 and not s.empty
    return False


def _try_index(ticker: str, start, end) -> pd.Series | None:
    """Try downloading a Yahoo Finance index, return log-return Series or None."""
    import yfinance as yf
    try:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False, timeout=10)
        if df is None or df.empty or len(df) < 60:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = pd.to_numeric(df["Close"].squeeze(), errors="coerce").dropna()
        ret = np.log(close / close.shift(1)).dropna()
        return ret if len(ret) >= 60 else None
    except Exception:
        return None


def fetch_ff3_factors(nifty_returns: pd.Series | None) -> dict:
    """
    Build Fama-French factor proxies from Indian market data.
    Falls back gracefully — worst case returns CAPM-only.
    """
    logger.info("\n  Building Fama-French factor proxies...")

    rf_daily = 0.065 / 252
    factors  = {"market_excess": None, "smb": None, "hml": None, "status": "capm_only"}

    end   = datetime.today()
    start = end - timedelta(days=3 * 365 + 90)

    # Market excess
    if _series_safe(nifty_returns):
        factors["market_excess"] = nifty_returns - rf_daily
        logger.info("    market_excess: OK (Nifty50 - Rf)")

    # SMB: Midcap - Largecap
    for ticker in ["^CNXMIDCAP", "^NSEMDCP50", "^CNXSC", "^CNXMID150"]:
        mc_ret = _try_index(ticker, start, end)
        if not _series_safe(mc_ret):
            continue
        if _series_safe(nifty_returns):
            common = mc_ret.index.intersection(nifty_returns.index)
            if len(common) < 60:
                continue
            smb = mc_ret.loc[common] - nifty_returns.loc[common]
        else:
            smb = mc_ret - rf_daily
        if len(smb) >= 60:
            factors["smb"] = smb
            logger.info(f"    SMB: OK ({ticker}, {len(smb)}d, avg {smb.mean()*252:.2%}/yr)")
            break

    if factors["smb"] is None:
        logger.info("    SMB: not available — will use CAPM")

    # HML: Value proxy - Growth proxy
    for vtick, gtick in [("^CNXFMCG", "^CNXIT"), ("^CNXPHARMA", "^CNXAUTO")]:
        vr = _try_index(vtick, start, end)
        gr = _try_index(gtick, start, end)
        if not _series_safe(vr) or not _series_safe(gr):
            continue
        common = vr.index.intersection(gr.index)
        if len(common) < 60:
            continue
        hml = vr.loc[common] - gr.loc[common]
        if len(hml) >= 60:
            factors["hml"] = hml
            logger.info(f"    HML: OK ({vtick}-{gtick}, {len(hml)}d, avg {hml.mean()*252:.2%}/yr)")
            break

    if factors["hml"] is None:
        logger.info("    HML: not available — will use CAPM")

    # Status
    if factors["market_excess"] is not None:
        has_s = factors["smb"] is not None
        has_h = factors["hml"] is not None
        factors["status"] = "full_ff3" if (has_s and has_h) else "partial_ff2" if (has_s or has_h) else "capm_only"

    logger.info(f"    Factor status: {factors['status']}")
    return factors


# ════════════════════════════════════════════════════════
#  STOCK SELECTION SCORING
# ════════════════════════════════════════════════════════

def score_stock(features: dict) -> float:
    s  = features.get("mom_3m", 0) * 2.0
    s += features.get("mom_1m", 0) * 1.0
    s += features.get("mom_1y", 0) * 0.5
    if features.get("sigma_annual", 0.3) > 0.6:
        s -= 0.15
    return float(s)


# ════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    logger.info("=" * 65)
    logger.info("QUANTEDGE ANALYTICS — PHASE 1 DAILY PIPELINE")
    logger.info(f"Started: {datetime.now().strftime('%d %b %Y %H:%M IST')}")
    logger.info(f"Universe: Nifty 100 ({len(NIFTY100_SYMBOLS)} symbols)")
    logger.info(f"Features: FF3={USE_FAMA_FRENCH} | Regime={USE_REGIME_DETECTION} | "
                f"Sentiment={USE_SENTIMENT} | Earnings={USE_EARNINGS_RISK} | "
                f"CorrMC={USE_CORRELATED_MC}")
    logger.info("=" * 65)

    # Step 1: Nifty 50 composition (metadata only)
    logger.info("\n[1/8] Live Nifty 50 composition (metadata)...")
    nifty50     = fetch_live_nifty50()
    nifty50_map = {s["symbol"]: s for s in nifty50}
    logger.info(f"  -> {len(nifty50)} stocks from NSE API")
    with open(DOCS_DIR / "nifty50_composition.json", "w") as f:
        json.dump({
            "last_updated": datetime.now().strftime("%d %b %Y %H:%M IST"),
            "count": len(nifty50), "stocks": nifty50,
        }, f)

    # Step 2: Nifty 50 index returns
    logger.info("\n[2/8] Fetching Nifty 50 index returns...")
    nifty_returns = fetch_nifty_returns(years=3)

    # Step 3: Fama-French factors
    logger.info("\n[3/8] Building Fama-French factor proxies...")
    if USE_FAMA_FRENCH:
        ff3_factors = fetch_ff3_factors(nifty_returns)
    else:
        ff3_factors = {"market_excess": nifty_returns, "smb": None, "hml": None, "status": "capm_only"}

    # Step 4: Market regime
    logger.info("\n[4/8] Detecting market regime...")
    if USE_REGIME_DETECTION and _series_safe(nifty_returns):
        regime_info = detect_market_regime(nifty_returns)
        logger.info(f"  Regime: {regime_info['regime']} | Bull prob: {regime_info['bull_prob']:.1%}")
        try:
            with open(DOCS_DIR / "market_regime.json", "w") as f:
                json.dump({
                    "last_updated": datetime.now().strftime("%d %b %Y %H:%M IST"),
                    **{k: v for k, v in regime_info.items() if not isinstance(v, pd.Series)},
                }, f, indent=2)
        except Exception:
            pass
    else:
        regime_info = {"regime": "Unknown", "bull_prob": 0.5, "bear_prob": 0.5,
                       "drift_adjustment": 0.0, "method": "disabled"}
        logger.info("  Regime detection disabled or insufficient Nifty data")

    # Step 5: Quick scan — Nifty 100
    logger.info(f"\n[5/8] Quick scan: Nifty 100 ({len(NIFTY100_SYMBOLS)} symbols) -> top {TOP_N_STOCKS}...")
    quick_feat = {}
    for sym in NIFTY100_SYMBOLS:
        try:
            df = fetch_stock_data(sym, years=1)
            if df is not None and len(df) > 60:
                quick_feat[sym] = compute_features(df)
        except Exception as e:
            logger.debug(f"  {sym}: scan failed — {e}")

    scored   = sorted([(s, score_stock(f)) for s, f in quick_feat.items()], key=lambda x: x[1], reverse=True)
    selected = [s for s, _ in scored[:TOP_N_STOCKS]]
    logger.info(f"  Scanned {len(quick_feat)}/{len(NIFTY100_SYMBOLS)} | Selected {len(selected)}")

    with open(DOCS_DIR / "selected_stocks.json", "w") as f:
        json.dump({"last_updated": datetime.now().strftime("%d %b %Y %H:%M IST"),
                   "selected": selected, "scores": {s: round(sc, 4) for s, sc in scored[:TOP_N_STOCKS]}}, f)

    # Step 6: Full pipeline
    logger.info(f"\n[6/8] Running full pipeline on {len(selected)} stocks...")
    results       = []
    failed        = []
    corr_mc_input = {}

    # CRITICAL: never use `or` to choose between pandas Series — use explicit check
    _mkt = ff3_factors["market_excess"] if _series_safe(ff3_factors.get("market_excess")) else nifty_returns
    _smb = ff3_factors.get("smb") if _series_safe(ff3_factors.get("smb")) else None
    _hml = ff3_factors.get("hml") if _series_safe(ff3_factors.get("hml")) else None

    for i, sym in enumerate(selected, 1):
        logger.info(f"\n[{i:02d}/{len(selected)}] -- {sym}")

        df = fetch_stock_data(sym, years=10)
        if df is None:
            failed.append(sym)
            continue

        feat     = compute_features(df)
        fin_info = fetch_financial_info(sym)

        meta = nifty50_map.get(sym, {})
        fin_info["name"]   = fin_info.get("name") or meta.get("name", sym)
        fin_info["sector"] = fin_info.get("sector") or meta.get("sector") or SECTOR_MAP.get(sym, "Other")

        logger.info("    Technical indicators...")
        tech = compute_technical_indicators(df)

        earnings_info = {}
        if USE_EARNINGS_RISK:
            try:
                earnings_info = get_earnings_risk_flag(sym)
            except Exception as e:
                logger.debug(f"    Earnings flag error: {e}")

        sentiment_info = {}
        if USE_SENTIMENT:
            try:
                sentiment_info = get_sentiment_score(sym, fin_info.get("name", sym))
            except Exception as e:
                logger.debug(f"    Sentiment error: {e}")

        result = run_full_pipeline(
            symbol         = sym,
            features       = feat,
            fin_info       = fin_info,
            nifty_returns  = nifty_returns,
            market_excess  = _mkt,
            smb_returns    = _smb,
            hml_returns    = _hml,
            regime_info    = regime_info,
            earnings_info  = earnings_info,
            sentiment_info = sentiment_info,
        )

        if result:
            result.update(tech)
            sig_updated = generate_signal(
                prob_up           = result["prob_up"],
                var_95            = result["var_95"],
                sharpe            = result["sharpe"],
                expected_ret      = result["expected_return"],
                mispricing        = result["mispricing_pct"] / 100,
                beta              = result["beta_nifty"],
                sentiment_score   = float(result.get("sentiment_score") or 0),
                regime            = result.get("regime", "Unknown"),
                has_earnings      = bool(result.get("has_earnings_soon", False)),
                tech_score        = int(tech.get("tech_score") or 0),
                fundamental_score = int(result.get("fundamental_score") or 0),
            )
            result.update(sig_updated)
            results.append(result)
            corr_mc_input[sym] = {
                "S0": feat["current_price"], "mu": result["mu_annual"] / 252,
                "sigma": feat["sigma_daily"], "log_ret": feat["log_returns"],
            }
        else:
            failed.append(sym)

    logger.info(f"\n  Done: {len(results)} OK, {len(failed)} failed")

    # Step 7: Correlated MC
    portfolio_stats = {}
    if USE_CORRELATED_MC and len(corr_mc_input) >= 5:
        logger.info(f"\n[7/8] Correlated MC ({len(corr_mc_input)} stocks)...")
        try:
            portfolio_stats = run_correlated_mc(corr_mc_input)
            if portfolio_stats:
                port_out = {k: v for k, v in portfolio_stats.items() if k != "correlation_matrix"}
                port_out["last_updated"] = datetime.now().strftime("%d %b %Y %H:%M IST")
                with open(DOCS_DIR / "portfolio_stats.json", "w") as f:
                    json.dump(port_out, f, indent=2)
                if "correlation_matrix" in portfolio_stats:
                    with open(DOCS_DIR / "correlation_matrix.json", "w") as f:
                        json.dump({"last_updated": datetime.now().strftime("%d %b %Y %H:%M IST"),
                                   "symbols": portfolio_stats.get("corr_symbols", []),
                                   "matrix": portfolio_stats["correlation_matrix"]},
                                  f, separators=(",", ":"))
                logger.info(f"  Portfolio VaR 95%: {portfolio_stats.get('portfolio_var_95', 0):.1%}")
        except Exception as e:
            logger.warning(f"  Correlated MC failed: {e}")
    else:
        logger.info(f"\n[7/8] Correlated MC skipped")

    # Step 8: Save outputs
    logger.info(f"\n[8/8] Saving outputs...")

    results.sort(key=lambda x: (x.get("score", 0), x.get("expected_return", 0)), reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i

    charts = {r["symbol"]: {"path_charts": r.pop("path_charts", {}), "histogram": r.pop("histogram", {})} for r in results}

    metrics_out = {
        "last_updated": datetime.now().strftime("%d %b %Y %H:%M IST"),
        "total_stocks": len(results), "failed_stocks": failed,
        "runtime_mins": round((time.time() - t0) / 60, 1),
        "regime": {k: v for k, v in regime_info.items() if not isinstance(v, pd.Series)},
        "ff3_factor_status": ff3_factors.get("status", "unknown"),
        "portfolio_var_95": portfolio_stats.get("portfolio_var_95"),
        "stocks": results,
    }
    with open(DOCS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, separators=(",", ":"))

    with open(DOCS_DIR / "charts.json", "w") as f:
        json.dump(charts, f, separators=(",", ":"))

    summary_fields = [
        "symbol","name","sector","price","expected_return_pct","expected_return",
        "prob_up","signal","signal_color","confidence","score",
        "sharpe","var_95","sigma_annual","mispricing_pct",
        "mom_1m","mom_3m","week52_high","week52_low",
        "market_cap","pe_ratio","beta_nifty","rank","mu_annual",
        "rsi_14","rsi_signal","macd_cross","tech_signal","tech_score",
        "above_sma50","above_sma200","golden_cross",
        "bb_position","bb_signal","stoch_k","vol_ratio",
        "sma_50","sma_200","fundamental_grade","fundamental_score",
        "mean_price","median_price","ci_5","ci_25","ci_75","ci_95",
        "n_simulations","horizon_days",
        "prob_10up","prob_20up","prob_10down",
        "bull_median","base_median","bear_median",
        "max_drawdown","sortino","calmar","cvar_95","var_99",
        "drift_method","model",
        "regime","regime_bull_prob","sentiment_label","sentiment_score",
        "has_earnings_soon","days_to_earnings","earnings_vol_mult",
        "ff3_b_market","ff3_b_smb","ff3_b_hml","ff3_r2",
    ]

    summary = {
        "last_updated": datetime.now().strftime("%d %b %Y %H:%M IST"),
        "total_stocks": len(results),
        "regime": regime_info.get("regime", "Unknown"),
        "regime_bull_prob": regime_info.get("bull_prob", 0.5),
        "ff3_status": ff3_factors.get("status", "unknown"),
        "portfolio_var_95": portfolio_stats.get("portfolio_var_95"),
        "summary": [{k: r.get(k) for k in summary_fields} for r in results],
    }
    with open(DOCS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, separators=(",", ":"))

    if results:
        try:
            store_signals(summary["summary"])
        except Exception as e:
            logger.warning(f"  Signal storage: {e}")
        try:
            evaluate_signals()
            generate_report()
        except Exception as e:
            logger.warning(f"  Backtest: {e}")

    avg_ret = np.mean([r["expected_return_pct"] for r in results]) if results else 0
    buy_cnt = sum(1 for r in results if "BUY" in r.get("signal", ""))
    logger.info("\n" + "=" * 65)
    logger.info(f"  Processed  : {len(results)} / {len(selected)}")
    logger.info(f"  Failed     : {len(failed)}")
    logger.info(f"  BUY signals: {buy_cnt}")
    logger.info(f"  Avg E[ret] : {avg_ret:+.1f}%")
    logger.info(f"  Regime     : {regime_info.get('regime','?')}")
    logger.info(f"  FF3 status : {ff3_factors.get('status','?')}")
    logger.info(f"  Runtime    : {(time.time()-t0)/60:.1f} min")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
