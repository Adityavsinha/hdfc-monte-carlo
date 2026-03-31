"""
visualization.py
----------------
Publication-quality plots for the Monte Carlo simulation.
  1. Simulated price paths (fanned + percentile bands)
  2. Terminal price histogram with VaR lines
  3. Scenario comparison (Bull / Base / Bear)
  4. Rolling volatility chart
  5. Probability dashboard summary bar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional
from simulation_engine import SimResults

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ──────────────────────────
PALETTE = dict(
    bg        = "#0d1117",
    panel     = "#161b22",
    text      = "#e6edf3",
    accent    = "#58a6ff",
    bull      = "#3fb950",
    bear      = "#f85149",
    neutral   = "#d29922",
    band_fill = "#58a6ff22",
)


def _apply_dark_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor"  : PALETTE["bg"],
        "axes.facecolor"    : PALETTE["panel"],
        "axes.edgecolor"    : "#30363d",
        "axes.labelcolor"   : PALETTE["text"],
        "axes.titlecolor"   : PALETTE["text"],
        "xtick.color"       : PALETTE["text"],
        "ytick.color"       : PALETTE["text"],
        "text.color"        : PALETTE["text"],
        "grid.color"        : "#21262d",
        "grid.linestyle"    : "--",
        "grid.linewidth"    : 0.5,
        "font.family"       : "DejaVu Sans",
        "font.size"         : 10,
    })


# ─────────────────────────────────────────────
# PLOT 1 — Price Path Fan
# ─────────────────────────────────────────────

def plot_price_paths(
    results     : SimResults,
    n_show      : int = 200,
    save        : bool = True,
) -> plt.Figure:
    _apply_dark_theme()
    fig, ax = plt.subplots(figsize=(14, 6))

    T = results.paths.shape[0]
    x = np.arange(1, T + 1)

    # Draw a random subset of paths
    idx = np.random.choice(results.paths.shape[1], size=n_show, replace=False)
    for i in idx:
        ax.plot(x, results.paths[:, i], color=PALETTE["accent"], alpha=0.04, lw=0.6)

    # Percentile bands
    p5  = np.percentile(results.paths, 5,  axis=1)
    p50 = np.percentile(results.paths, 50, axis=1)
    p95 = np.percentile(results.paths, 95, axis=1)

    ax.fill_between(x, p5, p95, color=PALETTE["accent"], alpha=0.12, label="5–95% band")
    ax.plot(x, p95, color=PALETTE["bull"],    lw=1.5, ls="--", label="95th pct")
    ax.plot(x, p50, color=PALETTE["neutral"], lw=2.0,          label="Median")
    ax.plot(x, p5,  color=PALETTE["bear"],    lw=1.5, ls="--", label="5th pct")

    # Current price line
    ax.axhline(results.S0, color="#8b949e", lw=1, ls=":", label=f"Current ₹{results.S0:,.2f}")

    ax.set_title(
        f"HDFC Bank — Monte Carlo Price Paths  "
        f"({results.config.n_simulations:,} sims | {T}-day horizon)",
        pad=14, fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Trading Days Forward")
    ax.set_ylabel("Price (₹)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
    ax.legend(loc="upper left", framealpha=0.3, fontsize=9)
    ax.grid(True)
    fig.tight_layout()

    if save:
        path = OUTPUT_DIR / "01_price_paths.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ─────────────────────────────────────────────
# PLOT 2 — Terminal Price Histogram
# ─────────────────────────────────────────────

def plot_histogram(results: SimResults, save: bool = True) -> plt.Figure:
    _apply_dark_theme()
    fig, ax = plt.subplots(figsize=(12, 5))

    final = results.final_prices
    ax.hist(
        final, bins=120, color=PALETTE["accent"], alpha=0.7,
        edgecolor="#0d1117", linewidth=0.3, density=True
    )

    # VaR lines
    ax.axvline(results.ci_5,  color=PALETTE["bear"],    lw=2, ls="--",
               label=f"5th pct  ₹{results.ci_5:,.0f}")
    ax.axvline(results.ci_50, color=PALETTE["neutral"], lw=2,
               label=f"Median   ₹{results.ci_50:,.0f}")
    ax.axvline(results.ci_95, color=PALETTE["bull"],    lw=2, ls="--",
               label=f"95th pct ₹{results.ci_95:,.0f}")
    ax.axvline(results.S0,    color="white",            lw=1.5, ls=":",
               label=f"Current  ₹{results.S0:,.0f}")

    # VaR annotation
    var_line = results.S0 - results.var_95
    ax.axvline(var_line, color="#ff7b72", lw=1.5, ls="-.",
               label=f"VaR 95%  ₹{results.var_95:,.0f} loss")

    ax.set_title(
        f"Terminal Price Distribution  (Day {results.config.horizon_days})",
        pad=12, fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Terminal Price (₹)")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
    ax.legend(framealpha=0.3, fontsize=9)
    ax.grid(True)
    fig.tight_layout()

    if save:
        path = OUTPUT_DIR / "02_histogram.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ─────────────────────────────────────────────
# PLOT 3 — Scenario Comparison
# ─────────────────────────────────────────────

def plot_scenarios(results: SimResults, save: bool = True) -> plt.Figure:
    _apply_dark_theme()
    fig, ax = plt.subplots(figsize=(14, 6))

    T   = results.config.horizon_days
    x   = np.arange(1, T + 1)
    cfg = {
        "bull": (PALETTE["bull"],   "Bull Case"),
        "base": (PALETTE["accent"], "Base Case"),
        "bear": (PALETTE["bear"],   "Bear Case"),
    }

    for key, (color, label) in cfg.items():
        paths = results.scenarios[key]
        p5    = np.percentile(paths, 5,  axis=1)
        p50   = np.percentile(paths, 50, axis=1)
        p95   = np.percentile(paths, 95, axis=1)
        ax.fill_between(x, p5, p95, color=color, alpha=0.10)
        ax.plot(x, p50, color=color, lw=2, label=f"{label} median ₹{p50[-1]:,.0f}")

    ax.axhline(results.S0, color="#8b949e", lw=1, ls=":", label=f"Entry ₹{results.S0:,.0f}")

    ax.set_title("Scenario Analysis  (Bull / Base / Bear)", pad=12, fontsize=13, fontweight="bold")
    ax.set_xlabel("Trading Days Forward")
    ax.set_ylabel("Price (₹)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
    ax.legend(framealpha=0.3, fontsize=9)
    ax.grid(True)
    fig.tight_layout()

    if save:
        path = OUTPUT_DIR / "03_scenarios.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ─────────────────────────────────────────────
# PLOT 4 — Rolling Volatility
# ─────────────────────────────────────────────

def plot_rolling_vol(log_returns: pd.Series, save: bool = True) -> plt.Figure:
    _apply_dark_theme()
    fig, ax = plt.subplots(figsize=(14, 4))

    rv_21  = log_returns.rolling(21).std()  * np.sqrt(252) * 100
    rv_63  = log_returns.rolling(63).std()  * np.sqrt(252) * 100
    rv_252 = log_returns.rolling(252).std() * np.sqrt(252) * 100

    ax.fill_between(rv_21.index, rv_21, color=PALETTE["accent"], alpha=0.15)
    ax.plot(rv_21.index,  rv_21,  color=PALETTE["accent"],  lw=1,   label="21-day realised vol")
    ax.plot(rv_63.index,  rv_63,  color=PALETTE["neutral"], lw=1.5, label="63-day realised vol")
    ax.plot(rv_252.index, rv_252, color=PALETTE["bear"],    lw=2,   label="252-day realised vol")

    ax.set_title("HDFC Bank — Rolling Annualised Volatility (%)", pad=12, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualised Vol (%)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.legend(framealpha=0.3, fontsize=9)
    ax.grid(True)
    fig.tight_layout()

    if save:
        path = OUTPUT_DIR / "04_rolling_volatility.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ─────────────────────────────────────────────
# PLOT 5 — Summary Dashboard (single figure)
# ─────────────────────────────────────────────

def plot_dashboard(results: SimResults, log_returns: pd.Series, save: bool = True) -> plt.Figure:
    _apply_dark_theme()
    fig = plt.figure(figsize=(18, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax_paths = fig.add_subplot(gs[0, :2])
    ax_hist  = fig.add_subplot(gs[0, 2])
    ax_scen  = fig.add_subplot(gs[1, :2])
    ax_vol   = fig.add_subplot(gs[1, 2])

    T = results.paths.shape[0]
    x = np.arange(1, T + 1)

    # ── Paths ──
    for i in np.random.choice(results.paths.shape[1], 100, replace=False):
        ax_paths.plot(x, results.paths[:, i], color=PALETTE["accent"], alpha=0.04, lw=0.5)
    p5  = np.percentile(results.paths, 5,  axis=1)
    p50 = np.percentile(results.paths, 50, axis=1)
    p95 = np.percentile(results.paths, 95, axis=1)
    ax_paths.fill_between(x, p5, p95, color=PALETTE["accent"], alpha=0.12)
    ax_paths.plot(x, p50, color=PALETTE["neutral"], lw=2, label="Median")
    ax_paths.plot(x, p95, color=PALETTE["bull"],    lw=1.2, ls="--", label="95th")
    ax_paths.plot(x, p5,  color=PALETTE["bear"],    lw=1.2, ls="--", label="5th")
    ax_paths.axhline(results.S0, color="white", lw=0.8, ls=":")
    ax_paths.set_title("Simulated Price Paths", fontsize=11, fontweight="bold")
    ax_paths.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
    ax_paths.legend(fontsize=8, framealpha=0.3)
    ax_paths.grid(True)

    # ── Histogram ──
    ax_hist.hist(results.final_prices, bins=80, color=PALETTE["accent"], alpha=0.75,
                 edgecolor="#0d1117", linewidth=0.2, density=True, orientation="horizontal")
    ax_hist.axhline(results.ci_5,  color=PALETTE["bear"],    lw=1.5, ls="--")
    ax_hist.axhline(results.ci_50, color=PALETTE["neutral"], lw=1.5)
    ax_hist.axhline(results.ci_95, color=PALETTE["bull"],    lw=1.5, ls="--")
    ax_hist.axhline(results.S0,    color="white",            lw=1,   ls=":")
    ax_hist.set_title("Terminal Distribution", fontsize=11, fontweight="bold")
    ax_hist.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
    ax_hist.grid(True)

    # ── Scenarios ──
    for key, (color, label) in [
        ("bull", (PALETTE["bull"],   "Bull")),
        ("base", (PALETTE["accent"], "Base")),
        ("bear", (PALETTE["bear"],   "Bear")),
    ]:
        paths = results.scenarios[key]
        p50s  = np.percentile(paths, 50, axis=1)
        ax_scen.plot(x, p50s, color=color, lw=2, label=f"{label} ₹{p50s[-1]:,.0f}")
    ax_scen.axhline(results.S0, color="white", lw=0.8, ls=":")
    ax_scen.set_title("Scenario Analysis", fontsize=11, fontweight="bold")
    ax_scen.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
    ax_scen.legend(fontsize=8, framealpha=0.3)
    ax_scen.grid(True)

    # ── Rolling Vol ──
    rv = log_returns.rolling(63).std() * np.sqrt(252) * 100
    ax_vol.plot(rv.index, rv, color=PALETTE["neutral"], lw=1.5)
    ax_vol.fill_between(rv.index, rv, color=PALETTE["neutral"], alpha=0.15)
    ax_vol.set_title("63-day Rolling Vol (%)", fontsize=11, fontweight="bold")
    ax_vol.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax_vol.grid(True)

    # ── Suptitle with key metrics ──
    fig.suptitle(
        f"HDFC Bank Monte Carlo Dashboard  |  "
        f"Current ₹{results.S0:,.2f}  →  E[Day {T}] ₹{results.mean_price:,.2f}  |  "
        f"P(↑) {results.prob_increase:.1%}  |  VaR95 ₹{results.var_95:,.0f}",
        fontsize=12, fontweight="bold", y=1.01
    )

    if save:
        path = OUTPUT_DIR / "00_dashboard.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


def render_all(results: SimResults, log_returns: pd.Series) -> None:
    plot_dashboard(results, log_returns)
    plot_price_paths(results)
    plot_histogram(results)
    plot_scenarios(results)
    plot_rolling_vol(log_returns)
    print("\nAll charts saved to ./outputs/")
