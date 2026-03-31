"""
html_dashboard.py
-----------------
Generates a beautiful standalone HTML dashboard from simulation results.
Open the output file in any browser — no server needed.
Also auto-refreshes if you re-run the simulation (just refresh the browser tab).
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from simulation_engine import SimResults

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_html_dashboard(results: SimResults, log_returns, links: dict = None) -> Path:
    """Generate a standalone HTML file with all charts and metrics."""

    S0 = results.S0
    T  = results.config.horizon_days
    now = datetime.now().strftime("%d %b %Y, %H:%M IST")

    # Pre-compute all data for JS
    p5  = np.percentile(results.paths, 5,  axis=1).tolist()
    p25 = np.percentile(results.paths, 25, axis=1).tolist()
    p50 = np.percentile(results.paths, 50, axis=1).tolist()
    p75 = np.percentile(results.paths, 75, axis=1).tolist()
    p95 = np.percentile(results.paths, 95, axis=1).tolist()

    # Histogram data
    counts, edges = np.histogram(results.final_prices, bins=60)
    hist_labels = [f"{(edges[i]+edges[i+1])/2:.0f}" for i in range(len(counts))]
    hist_data   = counts.tolist()

    # Scenario medians
    bull_med = np.percentile(results.scenarios["bull"], 50, axis=1).tolist()
    base_med = np.percentile(results.scenarios["base"], 50, axis=1).tolist()
    bear_med = np.percentile(results.scenarios["bear"], 50, axis=1).tolist()

    # Rolling vol
    rv63 = (log_returns.rolling(63).std() * np.sqrt(252) * 100).dropna()
    rv_dates  = [d.strftime("%b %Y") for d in rv63.index[::20]]
    rv_values = rv63.values[::20].tolist()

    days = list(range(1, T + 1))

    # Shareable links section
    links_html = ""
    if links:
        sheets_url = links.get("google_sheets", "")
        drive_files = links.get("drive_files", {})
        if sheets_url:
            links_html = f"""
            <div class="links-bar">
              <span class="links-label">Live Dashboard:</span>
              <a href="{sheets_url}" target="_blank" class="share-btn">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                Google Sheets
              </a>
              <a href="{'https://drive.google.com/drive/folders/' + list(drive_files.values())[0].split('/d/')[1].split('/')[0] if drive_files else '#'}" target="_blank" class="share-btn">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
                Google Drive
              </a>
            </div>"""

    pct_change = (results.mean_price / S0 - 1) * 100
    pct_sign   = "+" if pct_change >= 0 else ""
    var_pct    = results.var_95 / S0 * 100

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HDFC Bank — Monte Carlo Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --surface2: #21262d;
    --border: #30363d; --text: #e6edf3; --muted: #8b949e;
    --blue: #58a6ff; --green: #3fb950; --red: #f85149;
    --gold: #d29922; --purple: #bc8cff;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; line-height: 1.6; min-height: 100vh; }}
  .header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 20px 32px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; }}
  .header-left h1 {{ font-size: 20px; font-weight: 600; color: var(--text); letter-spacing: -0.3px; }}
  .header-left p {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
  .price-badge {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 8px 16px; text-align: center; }}
  .price-badge .label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }}
  .price-badge .value {{ font-size: 22px; font-weight: 700; color: var(--blue); }}
  .links-bar {{ background: var(--surface2); border-bottom: 1px solid var(--border); padding: 10px 32px; display: flex; align-items: center; gap: 12px; font-size: 12px; color: var(--muted); flex-wrap: wrap; }}
  .share-btn {{ display: inline-flex; align-items: center; gap: 5px; background: var(--surface); border: 1px solid var(--border); color: var(--blue); padding: 5px 12px; border-radius: 6px; text-decoration: none; font-size: 12px; transition: border-color 0.15s; }}
  .share-btn:hover {{ border-color: var(--blue); }}
  .main {{ padding: 24px 32px; max-width: 1400px; margin: 0 auto; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .metric {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
  .metric .m-label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
  .metric .m-value {{ font-size: 20px; font-weight: 700; }}
  .metric .m-sub {{ font-size: 11px; margin-top: 4px; }}
  .green {{ color: var(--green); }} .red {{ color: var(--red); }} .blue {{ color: var(--blue); }} .gold {{ color: var(--gold); }} .purple {{ color: var(--purple); }}
  .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
  .chart-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; }}
  .chart-card.full {{ grid-column: 1 / -1; }}
  .chart-title {{ font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 14px; }}
  .chart-wrap {{ position: relative; height: 220px; }}
  .chart-wrap.tall {{ height: 260px; }}
  .prob-row {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid var(--border); }}
  .prob-row:last-child {{ border-bottom: none; }}
  .prob-label {{ font-size: 13px; color: var(--muted); }}
  .prob-bar-wrap {{ flex: 1; margin: 0 14px; background: var(--surface2); border-radius: 4px; height: 6px; }}
  .prob-bar {{ height: 6px; border-radius: 4px; }}
  .prob-val {{ font-size: 13px; font-weight: 700; min-width: 42px; text-align: right; }}
  .scenario-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .scenario-table th {{ text-align: left; color: var(--muted); font-weight: 500; padding: 8px 12px; border-bottom: 1px solid var(--border); }}
  .scenario-table td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); }}
  .scenario-table tr:last-child td {{ border-bottom: none; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
  .badge-green {{ background: rgba(63,185,80,0.15); color: var(--green); }}
  .badge-red {{ background: rgba(248,81,73,0.15); color: var(--red); }}
  .badge-blue {{ background: rgba(88,166,255,0.15); color: var(--blue); }}
  .footer {{ text-align: center; padding: 24px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--border); margin-top: 8px; }}
  @media (max-width: 768px) {{ .charts-grid {{ grid-template-columns: 1fr; }} .chart-card.full {{ grid-column: 1; }} .main {{ padding: 16px; }} }}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <h1>HDFC Bank Ltd — Monte Carlo Simulation</h1>
    <p>NSE: HDFCBANK &nbsp;·&nbsp; {results.config.n_simulations:,} simulations &nbsp;·&nbsp; {T}-day horizon &nbsp;·&nbsp; Last updated: {now}</p>
  </div>
  <div class="price-badge">
    <div class="label">Current Price</div>
    <div class="value">₹{S0:,.2f}</div>
  </div>
</div>

{links_html}

<div class="main">

  <!-- Metric Cards -->
  <div class="metrics-grid">
    <div class="metric">
      <div class="m-label">Expected Price</div>
      <div class="m-value blue">₹{results.mean_price:,.0f}</div>
      <div class="m-sub {'green' if pct_change >= 0 else 'red'}">{pct_sign}{pct_change:.1f}% expected return</div>
    </div>
    <div class="metric">
      <div class="m-label">Best Case (95th pct)</div>
      <div class="m-value green">₹{results.ci_95:,.0f}</div>
      <div class="m-sub green">+{(results.ci_95/S0-1)*100:.1f}% upside</div>
    </div>
    <div class="metric">
      <div class="m-label">Worst Case (5th pct)</div>
      <div class="m-value red">₹{results.ci_5:,.0f}</div>
      <div class="m-sub red">{(results.ci_5/S0-1)*100:.1f}% downside</div>
    </div>
    <div class="metric">
      <div class="m-label">VaR 95%</div>
      <div class="m-value red">₹{results.var_95:,.0f}</div>
      <div class="m-sub muted" style="color:var(--muted)">{var_pct:.1f}% of investment</div>
    </div>
    <div class="metric">
      <div class="m-label">P(Price Goes Up)</div>
      <div class="m-value {'green' if results.prob_increase > 0.5 else 'red'}">{results.prob_increase:.1%}</div>
      <div class="m-sub" style="color:var(--muted)">probability of profit</div>
    </div>
    <div class="metric">
      <div class="m-label">Annual Volatility</div>
      <div class="m-value gold">{results.sigma*(252**0.5)*100:.1f}%</div>
      <div class="m-sub" style="color:var(--muted)">rolling EWMA</div>
    </div>
  </div>

  <!-- Charts Row 1 -->
  <div class="charts-grid">

    <!-- Price Paths -->
    <div class="chart-card full">
      <div class="chart-title">Simulated Price Paths — Percentile Fan</div>
      <div class="chart-wrap tall">
        <canvas id="pathsChart"></canvas>
      </div>
    </div>

    <!-- Histogram -->
    <div class="chart-card">
      <div class="chart-title">Terminal Price Distribution (Day {T})</div>
      <div class="chart-wrap">
        <canvas id="histChart"></canvas>
      </div>
    </div>

    <!-- Scenarios -->
    <div class="chart-card">
      <div class="chart-title">Scenario Analysis — Bull / Base / Bear</div>
      <div class="chart-wrap">
        <canvas id="scenChart"></canvas>
      </div>
    </div>

    <!-- Probabilities -->
    <div class="chart-card">
      <div class="chart-title">Probability Breakdown</div>
      <div style="margin-top: 8px;">
        <div class="prob-row">
          <span class="prob-label">Price goes up (any gain)</span>
          <div class="prob-bar-wrap"><div class="prob-bar" style="width:{results.prob_increase*100:.1f}%; background:var(--green);"></div></div>
          <span class="prob-val green">{results.prob_increase:.1%}</span>
        </div>
        <div class="prob-row">
          <span class="prob-label">Price gains more than +10%</span>
          <div class="prob-bar-wrap"><div class="prob-bar" style="width:{results.prob_10pct_up*100:.1f}%; background:var(--blue);"></div></div>
          <span class="prob-val blue">{results.prob_10pct_up:.1%}</span>
        </div>
        <div class="prob-row">
          <span class="prob-label">Price stays flat (±5%)</span>
          <div class="prob-bar-wrap"><div class="prob-bar" style="width:{min((1-results.prob_10pct_up-results.prob_10pct_down)*100,100):.1f}%; background:var(--gold);"></div></div>
          <span class="prob-val gold">{(1-results.prob_10pct_up-results.prob_10pct_down):.1%}</span>
        </div>
        <div class="prob-row">
          <span class="prob-label">Price drops more than -10%</span>
          <div class="prob-bar-wrap"><div class="prob-bar" style="width:{results.prob_10pct_down*100:.1f}%; background:var(--red);"></div></div>
          <span class="prob-val red">{results.prob_10pct_down:.1%}</span>
        </div>
      </div>
    </div>

    <!-- Scenario Table -->
    <div class="chart-card">
      <div class="chart-title">Scenario Summary</div>
      <table class="scenario-table">
        <thead>
          <tr><th>Scenario</th><th>Median Price</th><th>Return</th><th>Confidence</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><span class="badge badge-green">Bull</span></td>
            <td class="green">₹{np.median(results.scenarios['bull'][-1,:]):,.0f}</td>
            <td class="green">+{(np.median(results.scenarios['bull'][-1,:])/S0-1)*100:.1f}%</td>
            <td style="color:var(--muted)">Optimistic</td>
          </tr>
          <tr>
            <td><span class="badge badge-blue">Base</span></td>
            <td class="blue">₹{np.median(results.scenarios['base'][-1,:]):,.0f}</td>
            <td class="{'green' if np.median(results.scenarios['base'][-1,:])>S0 else 'red'}">{(np.median(results.scenarios['base'][-1,:])/S0-1)*100:+.1f}%</td>
            <td style="color:var(--muted)">Most likely</td>
          </tr>
          <tr>
            <td><span class="badge badge-red">Bear</span></td>
            <td class="red">₹{np.median(results.scenarios['bear'][-1,:]):,.0f}</td>
            <td class="red">{(np.median(results.scenarios['bear'][-1,:])/S0-1)*100:+.1f}%</td>
            <td style="color:var(--muted)">Pessimistic</td>
          </tr>
        </tbody>
      </table>

      <div style="margin-top: 20px;">
        <div class="chart-title">Rolling Volatility (63-day)</div>
        <div class="chart-wrap" style="height:130px; margin-top:10px;">
          <canvas id="volChart"></canvas>
        </div>
      </div>
    </div>

  </div>
</div>

<div class="footer">
  HDFC Bank Monte Carlo System &nbsp;·&nbsp; GBM + Student-t fat tails + EWMA volatility &nbsp;·&nbsp; Generated {now}
  <br>This is a quantitative model output, not financial advice.
</div>

<script>
const days = {json.dumps(days[::2])};
const p5   = {json.dumps([round(v,2) for v in p5[::2]])};
const p25  = {json.dumps([round(v,2) for v in p25[::2]])};
const p50  = {json.dumps([round(v,2) for v in p50[::2]])};
const p75  = {json.dumps([round(v,2) for v in p75[::2]])};
const p95  = {json.dumps([round(v,2) for v in p95[::2]])};
const bullM = {json.dumps([round(v,2) for v in bull_med[::2]])};
const baseM = {json.dumps([round(v,2) for v in base_med[::2]])};
const bearM = {json.dumps([round(v,2) for v in bear_med[::2]])};
const histL = {json.dumps(hist_labels)};
const histD = {json.dumps(hist_data)};
const rvDates = {json.dumps(rv_dates)};
const rvVals  = {json.dumps([round(v,2) for v in rv_values])};
const S0 = {S0};
const VaR95line = {round(S0 - results.var_95, 2)};

Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";

// ── Price Paths Chart ──
new Chart(document.getElementById('pathsChart'), {{
  type: 'line',
  data: {{
    labels: days,
    datasets: [
      {{ label: '95th pct', data: p95, borderColor: '#3fb950', borderWidth: 1.5, borderDash: [4,3], pointRadius: 0, fill: false }},
      {{ label: '75th pct', data: p75, borderColor: '#58a6ff', borderWidth: 0.5, pointRadius: 0, fill: '+1', backgroundColor: 'rgba(88,166,255,0.06)' }},
      {{ label: 'Median',   data: p50, borderColor: '#d29922', borderWidth: 2.5, pointRadius: 0, fill: false }},
      {{ label: '25th pct', data: p25, borderColor: '#58a6ff', borderWidth: 0.5, pointRadius: 0, fill: false }},
      {{ label: '5th pct',  data: p5,  borderColor: '#f85149', borderWidth: 1.5, borderDash: [4,3], pointRadius: 0, fill: false }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {{ legend: {{ display: true, position: 'top', labels: {{ boxWidth: 10, font: {{ size: 11 }} }} }} }},
    scales: {{
      x: {{ ticks: {{ maxTicksLimit: 10 }}, grid: {{ color: '#21262d' }} }},
      y: {{ ticks: {{ callback: v => '₹' + v.toLocaleString('en-IN') }}, grid: {{ color: '#21262d' }} }}
    }}
  }}
}});

// ── Histogram ──
new Chart(document.getElementById('histChart'), {{
  type: 'bar',
  data: {{
    labels: histL,
    datasets: [{{ label: 'Simulations', data: histD, backgroundColor: 'rgba(88,166,255,0.6)', borderColor: 'rgba(88,166,255,0.9)', borderWidth: 0.5 }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {{
      legend: {{ display: false }},
      annotation: {{}}
    }},
    scales: {{
      x: {{ ticks: {{ maxTicksLimit: 8, callback: (v, i) => '₹' + parseFloat(histL[i]).toLocaleString('en-IN', {{maximumFractionDigits:0}}) }}, grid: {{ display: false }} }},
      y: {{ ticks: {{ maxTicksLimit: 5 }}, grid: {{ color: '#21262d' }} }}
    }}
  }}
}});

// ── Scenarios ──
new Chart(document.getElementById('scenChart'), {{
  type: 'line',
  data: {{
    labels: days,
    datasets: [
      {{ label: 'Bull', data: bullM, borderColor: '#3fb950', borderWidth: 2, pointRadius: 0, fill: false }},
      {{ label: 'Base', data: baseM, borderColor: '#58a6ff', borderWidth: 2, pointRadius: 0, fill: false }},
      {{ label: 'Bear', data: bearM, borderColor: '#f85149', borderWidth: 2, pointRadius: 0, fill: false }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {{ legend: {{ display: true, position: 'top', labels: {{ boxWidth: 10, font: {{ size: 11 }} }} }} }},
    scales: {{
      x: {{ ticks: {{ maxTicksLimit: 8 }}, grid: {{ color: '#21262d' }} }},
      y: {{ ticks: {{ callback: v => '₹' + v.toLocaleString('en-IN') }}, grid: {{ color: '#21262d' }} }}
    }}
  }}
}});

// ── Rolling Vol ──
new Chart(document.getElementById('volChart'), {{
  type: 'line',
  data: {{
    labels: rvDates,
    datasets: [{{ label: '63-day Vol', data: rvVals, borderColor: '#d29922', borderWidth: 1.5, pointRadius: 0, fill: true, backgroundColor: 'rgba(210,153,34,0.1)' }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ maxTicksLimit: 6, font: {{ size: 10 }} }}, grid: {{ color: '#21262d' }} }},
      y: {{ ticks: {{ callback: v => v.toFixed(1) + '%', font: {{ size: 10 }} }}, grid: {{ color: '#21262d' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

    out_path = OUTPUT_DIR / "hdfc_dashboard.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path
