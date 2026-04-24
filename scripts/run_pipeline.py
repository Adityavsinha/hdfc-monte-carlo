"""
scripts/run_pipeline.py — QuantEdge Analytics Phase 1 Daily Pipeline

New in Phase 1:
  - Fama-French 3-Factor drift (market + SMB + HML factors from Nifty data)
  - Market regime detection (HMM bull/bear state passed to each stock)
  - Per-stock: earnings risk flag + news sentiment
  - Correlated Monte Carlo across all stocks → portfolio-level VaR
  - Backtesting: stores today's signals → evaluates past signals → report
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

# ── Local imports ────────────────────────────────────────────
from config import (
    TOP_N_STOCKS, NIFTY50_SYMBOLS, SECTOR_MAP,
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
    run_correlated_mc,
)
from backtest import store_signals, evaluate_signals, generate_report


# ════════════════════════════════════════════════════════
#  PHASE 1: FACTOR DATA FETCHING
# ════════════════════════════════════════════════════════

def fetch_nifty_returns(years: int = 3) -> pd.Series | None:
    """Fetch Nifty 50 index log-returns for beta & regime computation."""
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


def fetch_ff3_factors(nifty_returns: pd.Series | None) -> dict:
    """
    Build Fama-French factor proxies from Indian market data.

    market_excess = Nifty 50 returns - daily Rf
    smb_returns   = Nifty Midcap 150 returns - Nifty 50 returns  (size premium)
    hml_returns   = Nifty Value 20 returns   - Nifty Growth returns (value premium)

    Falls back gracefully if any index fetch fails.
    """
    import yfinance as yf
    logger.info("\n  Building Fama-French factor proxies...")

    factors = {
        "market_excess": None,
        "smb"          : None,
        "hml"          : None,
        "status"       : "partial",
    }

    rf_daily = 0.065 / 252   # risk-free daily rate

    # ── Market excess returns ───────────────────────────
    if nifty_returns is not None:
        factors["market_excess"] = nifty_returns - rf_daily
        logger.info("    market_excess: ✅ (Nifty50 - Rf)")

    # ── SMB: Midcap - Largecap returns (size premium) ──
    try:
        end   = datetime.today()
        start = end - timedelta(days=3 * 365 + 90)
        # Nifty Midcap 150 index
        mc    = yf.download("^CNXMDCP", start=start, end=end,
                            auto_adjust=True, progress=False)
        if mc.empty:
            raise ValueError("Midcap index empty")
        if isinstance(mc.columns, pd.MultiIndex):
            mc.columns = mc.columns.get_level_values(0)
        mc_close   = pd.to_numeric(mc["Close"].squeeze(), errors="coerce").dropna()
        mc_returns = np.log(mc_close / mc_close.shift(1)).dropna()

        if nifty_returns is not None:
            # SMB = Midcap daily return - Nifty50 daily return
            smb = mc_returns.subtract(nifty_returns, fill_value=None).dropna()
            factors["smb"] = smb
            logger.info(f"    SMB: ✅ ({len(smb)} days, avg={smb.mean()*252:.2%}/yr)")
        else:
            factors["smb"] = mc_returns - rf_daily
    except Exception as e:
        logger.info(f"    SMB: ❌ ({e}) — using zero factor")

    # ── HML: Value - Growth returns ────────────────────
    try:
        # Nifty 50 Value 20 as value proxy
        val  = yf.download("^CNXV20",   start=start, end=end,
                           auto_adjust=True, progress=False)
        # Nifty 200 Momentum 30 as growth proxy
        grw  = yf.download("^CNXMOMENTUM", start=start, end=end,
                           auto_adjust=True, progress=False)

        if not val.empty and not grw.empty:
            if isinstance(val.columns, pd.MultiIndex):
                val.columns = val.columns.get_level_values(0)
            if isinstance(grw.columns, pd.MultiIndex):
                grw.columns = grw.columns.get_level_values(0)

            val_ret = np.log(
                pd.to_numeric(val["Close"].squeeze(), errors="coerce").dropna()
                / pd.to_numeric(val["Close"].squeeze(), errors="coerce").dropna().shift(1)
            ).dropna()
            grw_ret = np.log(
                pd.to_numeric(grw["Close"].squeeze(), errors="coerce").dropna()
                / pd.to_numeric(grw["Close"].squeeze(), errors="coerce").dropna().shift(1)
            ).dropna()

            hml = val_ret.subtract(grw_ret, fill_value=None).dropna()
            factors["hml"] = hml
            logger.info(f"    HML: ✅ ({len(hml)} days, avg={hml.mean()*252:.2%}/yr)")
        else:
            raise ValueError("Value/Growth index empty")
    except Exception as e:
        logger.info(f"    HML: ❌ ({e}) — using zero factor")

    if factors["market_excess"] is not None:
        if factors["smb"] is not None and factors["hml"] is not None:
            factors["status"] = "full_ff3"
        elif factors["smb"] is not None or factors["hml"] is not None:
            factors["status"] = "partial_ff2"
        else:
            factors["status"] = "capm_only"

    logger.info(f"    Factor status: {factors['status']}")
    return factors


# ════════════════════════════════════════════════════════
#  STOCK SELECTION SCORING
# ════════════════════════════════════════════════════════

def score_stock(features: dict) -> float:
    """Rank stocks by momentum + liquidity for pipeline selection."""
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
    logger.info(f"Features: FF3={USE_FAMA_FRENCH} | Regime={USE_REGIME_DETECTION} | "
                f"Sentiment={USE_SENTIMENT} | Earnings={USE_EARNINGS_RISK} | "
                f"CorrMC={USE_CORRELATED_MC}")
    logger.info("=" * 65)

    # ── Step 1: Nifty 50 composition ─────────────────────────
    logger.info("\n[1/8] Live Nifty 50 composition...")
    nifty50 = fetch_live_nifty50()
    logger.info(f"  → {len(nifty50)} stocks")
    with open(DOCS_DIR / "nifty50_composition.json", "w") as f:
        json.dump({
            "last_updated": datetime.now().strftime("%d %b %Y %H:%M IST"),
            "count": len(nifty50), "stocks": nifty50,
        }, f)

    # ── Step 2: Nifty 50 index returns ───────────────────────
    logger.info("\n[2/8] Fetching Nifty 50 index returns...")
    nifty_returns = fetch_nifty_returns(years=3)

    # ── Step 3: Phase 1 — Fama-French factors ────────────────
    logger.info("\n[3/8] Building Fama-French factor proxies...")
    ff3_factors = {}
    if USE_FAMA_FRENCH:
        ff3_factors = fetch_ff3_factors(nifty_returns)
    else:
        ff3_factors = {"market_excess": nifty_returns, "smb": None, "hml": None}

    # ── Step 4: Phase 1 — Market regime detection ────────────
    logger.info("\n[4/8] Detecting market regime...")
    regime_info = {}
    if USE_REGIME_DETECTION and nifty_returns is not None:
        regime_info = detect_market_regime(nifty_returns)
        logger.info(
            f"  Regime: {regime_info['regime']} | "
            f"Bull prob: {regime_info['bull_prob']:.1%} | "
            f"Method: {regime_info['method']}"
        )
        # Save regime to docs for frontend
        with open(DOCS_DIR / "market_regime.json", "w") as f:
            json.dump({
                "last_updated": datetime.now().strftime("%d %b %Y %H:%M IST"),
                **regime_info,
            }, f, indent=2)
    else:
        logger.info("  Regime detection disabled or no data")
        regime_info = {"regime": "Unknown", "bull_prob": 0.5,
                       "drift_adjustment": 0.0, "method": "disabled"}

    # ── Step 5: Quick feature scan for stock selection ────────
    logger.info(f"\n[5/8] Quick scan for top-{TOP_N_STOCKS} selection...")
    all_symbols = [s["symbol"] for s in nifty50]
    # Add Nifty Next 50 to expand universe
    from config import NIFTY_NEXT50
    all_symbols = list(dict.fromkeys(all_symbols + NIFTY_NEXT50))

    quick_feat = {}
    for sym in all_symbols:
        try:
            df = fetch_stock_data(sym, years=1)
            if df is not None and len(df) > 60:
                quick_feat[sym] = compute_features(df)
        except Exception as e:
            logger.debug(f"  {sym}: quick scan failed — {e}")

    scored   = sorted(
        [(s, score_stock(f)) for s, f in quick_feat.items()],
        key=lambda x: x[1], reverse=True,
    )
    selected = [s for s, _ in scored[:TOP_N_STOCKS]]
    logger.info(f"  → {len(selected)} stocks selected for full pipeline")

    with open(DOCS_DIR / "selected_stocks.json", "w") as f:
        json.dump({
            "last_updated": datetime.now().strftime("%d %b %Y %H:%M IST"),
            "selected": selected,
            "scores": {s: round(sc, 4) for s, sc in scored[:TOP_N_STOCKS]},
        }, f)

    # ── Step 6: Full quant pipeline ───────────────────────────
    logger.info(f"\n[6/8] Running full pipeline on {len(selected)} stocks...")
    results          = []
    failed           = []
    corr_mc_input    = {}   # feeds correlated MC

    for i, sym in enumerate(selected, 1):
        logger.info(f"\n[{i:02d}/{len(selected)}] ── {sym} ──────────")

        # Data fetch
        df = fetch_stock_data(sym, years=10)
        if df is None:
            failed.append(sym)
            continue

        feat     = compute_features(df)
        fin_info = fetch_financial_info(sym)

        # Sector / name from Nifty50 composition
        meta = next((s for s in nifty50 if s["symbol"] == sym), {})
        fin_info["name"]   = fin_info.get("name") or meta.get("name", sym)
        fin_info["sector"] = (fin_info.get("sector")
                              or meta.get("sector")
                              or SECTOR_MAP.get(sym, "Other"))

        # Technical indicators
        logger.info("    Technical indicators...")
        tech = compute_technical_indicators(df)

        # Phase 1: Earnings risk
        earnings_info = {}
        if USE_EARNINGS_RISK:
            logger.info("    Earnings risk...")
            earnings_info = get_earnings_risk_flag(sym)

        # Phase 1: Sentiment
        sentiment_info = {}
        if USE_SENTIMENT:
            logger.info("    News sentiment...")
            sentiment_info = get_sentiment_score(sym, fin_info.get("name", sym))

        # Run full FF3 + MC pipeline
        result = run_full_pipeline(
            symbol         = sym,
            features       = feat,
            fin_info       = fin_info,
            nifty_returns  = nifty_returns,
            market_excess  = ff3_factors.get("market_excess"),
            smb_returns    = ff3_factors.get("smb"),
            hml_returns    = ff3_factors.get("hml"),
            regime_info    = regime_info,
            earnings_info  = earnings_info,
            sentiment_info = sentiment_info,
        )

        if result:
            # Merge technical indicators
            result.update(tech)

            # Re-run signal with tech_score now available (tech came after pipeline)
            from quant_engine import generate_signal
            sig_updated = generate_signal(
                prob_up           = result["prob_up"],
                var_95            = result["var_95"],
                sharpe            = result["sharpe"],
                expected_ret      = result["expected_return"],
                mispricing        = result["mispricing_pct"] / 100,
                beta              = result["beta_nifty"],
                sentiment_score   = result.get("sentiment_score", 0),
                regime            = result.get("regime", "Unknown"),
                has_earnings      = result.get("has_earnings_soon", False),
                tech_score        = tech.get("tech_score", 0),
                fundamental_score = result.get("fundamental_score", 0),
            )
            result.update(sig_updated)

            results.append(result)

            # Feed to correlated MC
            corr_mc_input[sym] = {
                "S0"     : feat["current_price"],
                "mu"     : result["mu_annual"] / 252,
                "sigma"  : feat["sigma_daily"],
                "log_ret": feat["log_returns"],
            }
        else:
            failed.append(sym)

    logger.info(f"\n  Pipeline done: {len(results)} success, {len(failed)} failed")

    # ── Step 7: Phase 1 — Correlated Monte Carlo ──────────────
    portfolio_stats = {}
    if USE_CORRELATED_MC and len(corr_mc_input) >= 5:
        logger.info(f"\n[7/8] Correlated Monte Carlo ({len(corr_mc_input)} stocks)...")
        portfolio_stats = run_correlated_mc(corr_mc_input)
        if portfolio_stats:
            with open(DOCS_DIR / "portfolio_stats.json", "w") as f:
                # Don't write full correlation matrix to portfolio_stats — only summary
                out = {
                    k: v for k, v in portfolio_stats.items()
                    if k != "correlation_matrix"
                }
                out["last_updated"] = datetime.now().strftime("%d %b %Y %H:%M IST")
                json.dump(out, f, indent=2)

            # Save correlation matrix separately
            if "correlation_matrix" in portfolio_stats:
                with open(DOCS_DIR / "correlation_matrix.json", "w") as f:
                    json.dump({
                        "last_updated": datetime.now().strftime("%d %b %Y %H:%M IST"),
                        "symbols": portfolio_stats.get("corr_symbols", []),
                        "matrix" : portfolio_stats["correlation_matrix"],
                    }, f, separators=(",", ":"))
            logger.info(
                f"  Portfolio VaR 95%: {portfolio_stats.get('portfolio_var_95', 0):.1%}"
            )
    else:
        logger.info("\n[7/8] Correlated MC skipped (disabled or insufficient stocks)")

    # ── Step 8: Save outputs ──────────────────────────────────
    logger.info(f"\n[8/8] Saving outputs...")

    # Sort by score then expected return
    results.sort(
        key=lambda x: (x.get("score", 0), x.get("expected_return", 0)),
        reverse=True,
    )
    for i, r in enumerate(results, 1):
        r["rank"] = i

    # Separate heavy chart data from summary
    charts = {
        r["symbol"]: {
            "path_charts": r.pop("path_charts", {}),
            "histogram"  : r.pop("histogram", {}),
        }
        for r in results
    }

    # ── metrics.json (full per-stock data) ────────────────────
    metrics_output = {
        "last_updated"    : datetime.now().strftime("%d %b %Y %H:%M IST"),
        "total_stocks"    : len(results),
        "failed_stocks"   : failed,
        "runtime_mins"    : round((time.time() - t0) / 60, 1),
        "regime"          : regime_info,
        "ff3_factor_status": ff3_factors.get("status", "unknown"),
        "portfolio_var_95": portfolio_stats.get("portfolio_var_95"),
        "portfolio_expected_ret": portfolio_stats.get("portfolio_expected_ret"),
        "stocks"          : results,
    }
    with open(DOCS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_output, f, separators=(",", ":"))

    # ── charts.json ───────────────────────────────────────────
    with open(DOCS_DIR / "charts.json", "w") as f:
        json.dump(charts, f, separators=(",", ":"))

    # ── summary.json (lightweight — what the frontend loads first) ──
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
        # Phase 1 additions in summary
        "regime","regime_bull_prob","sentiment_label","sentiment_score",
        "has_earnings_soon","days_to_earnings","earnings_vol_mult",
        "ff3_b_market","ff3_b_smb","ff3_b_hml","ff3_r2",
    ]

    summary = {
        "last_updated"    : datetime.now().strftime("%d %b %Y %H:%M IST"),
        "total_stocks"    : len(results),
        "regime"          : regime_info.get("regime", "Unknown"),
        "regime_bull_prob": regime_info.get("bull_prob", 0.5),
        "ff3_status"      : ff3_factors.get("status", "unknown"),
        "portfolio_var_95": portfolio_stats.get("portfolio_var_95"),
        "summary": [
            {k: r.get(k) for k in summary_fields}
            for r in results
        ],
    }
    with open(DOCS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, separators=(",", ":"))

    # ── Backtest: store today's signals ───────────────────────
    logger.info("\n  Storing signals to backtest history...")
    try:
        store_signals(summary["summary"])
    except Exception as e:
        logger.warning(f"  Signal storage failed: {e}")

    # ── Backtest: evaluate matured signals ────────────────────
    logger.info("  Evaluating matured signals...")
    try:
        evaluate_signals()
        generate_report()
    except Exception as e:
        logger.warning(f"  Backtest evaluation failed: {e}")

    # ── Summary log ───────────────────────────────────────────
    avg_ret = np.mean([r["expected_return_pct"] for r in results]) if results else 0
    buy_cnt = sum(1 for r in results if "BUY" in r.get("signal", ""))
    logger.info("\n" + "=" * 65)
    logger.info(f"  Stocks processed : {len(results)} / {len(selected)}")
    logger.info(f"  Failed           : {len(failed)}")
    logger.info(f"  BUY signals      : {buy_cnt}")
    logger.info(f"  Avg expected ret : {avg_ret:+.1f}%")
    logger.info(f"  Market regime    : {regime_info.get('regime','?')}")
    logger.info(f"  FF3 factor model : {ff3_factors.get('status','?')}")
    if portfolio_stats:
        logger.info(f"  Portfolio VaR 95%: {portfolio_stats.get('portfolio_var_95',0):.1%}")
    logger.info(f"  Runtime          : {(time.time()-t0)/60:.1f} min")
    logger.info(f"  summary.json     : {(DOCS_DIR/'summary.json').stat().st_size/1024:.1f} KB")
    logger.info(f"  metrics.json     : {(DOCS_DIR/'metrics.json').stat().st_size/1024:.1f} KB")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
