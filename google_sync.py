"""
google_sync.py
--------------
Handles:
  1. Uploading files to Google Drive (public shareable link)
  2. Writing a formatted live dashboard to Google Sheets
  3. Updating charts in Google Sheets

Setup required (run once):
  pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client gspread

Then run:  python setup_google.py
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────
CREDS_FILE  = Path("google_creds.json")    # OAuth token saved after first login
OUTPUT_DIR  = Path("outputs")


# ════════════════════════════════════════════
#  AUTH
# ════════════════════════════════════════════

def get_credentials():
    """
    OAuth2 flow — opens browser on first run, saves token for future runs.
    Scopes: Drive (upload) + Sheets (write dashboard).
    """
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request

    SCOPES = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets",
    ]

    creds = None
    if CREDS_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(CREDS_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open(CREDS_FILE, "w") as f:
            f.write(creds.to_json())

    return creds


# ════════════════════════════════════════════
#  GOOGLE DRIVE UPLOAD
# ════════════════════════════════════════════

def upload_to_drive(
    folder_id : str,
    file_path : Path,
    creds,
    mime_type : str = "application/octet-stream",
) -> str:
    """Upload a file to a Drive folder. Returns the public shareable URL."""
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    service = build("drive", "v3", credentials=creds)

    # Check if file already exists in folder → update instead of duplicate
    query = (
        f"name='{file_path.name}' and '{folder_id}' in parents and trashed=false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    existing = results.get("files", [])

    media = MediaFileUpload(str(file_path), mimetype=mime_type, resumable=True)

    if existing:
        file_id = existing[0]["id"]
        service.files().update(fileId=file_id, media_body=media).execute()
        logger.info(f"  Drive: updated  {file_path.name}")
    else:
        metadata = {"name": file_path.name, "parents": [folder_id]}
        result   = service.files().create(
            body=metadata, media_body=media, fields="id"
        ).execute()
        file_id = result["id"]
        logger.info(f"  Drive: uploaded {file_path.name}")

    # Make public
    service.permissions().create(
        fileId=file_id,
        body={"type": "anyone", "role": "reader"},
    ).execute()

    link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
    return link


def upload_all_outputs(folder_id: str, creds) -> dict:
    """Upload all PNGs + Excel from outputs/ to Drive. Returns {filename: link}."""
    links = {}
    mime_map = {
        ".png" : "image/png",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    for f in sorted(OUTPUT_DIR.glob("*")):
        if f.suffix in mime_map:
            link = upload_to_drive(folder_id, f, creds, mime_map[f.suffix])
            links[f.name] = link

    logger.info(f"  Drive: {len(links)} files uploaded/updated.")
    return links


# ════════════════════════════════════════════
#  GOOGLE SHEETS LIVE DASHBOARD
# ════════════════════════════════════════════

def _col(n: int) -> str:
    """Convert column number to letter (1=A, 2=B …)."""
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def write_sheets_dashboard(
    spreadsheet_id : str,
    results,                    # SimResults object
    drive_links    : dict,
    creds,
) -> str:
    """
    Writes a fully formatted Google Sheets dashboard.
    Returns the public shareable URL.
    """
    import gspread
    from googleapiclient.discovery import build

    gc = gspread.Client(auth=creds)
    sh = gc.open_by_key(spreadsheet_id)

    # ── Make spreadsheet public ──────────────
    drive_svc = build("drive", "v3", credentials=creds)
    drive_svc.permissions().create(
        fileId=spreadsheet_id,
        body={"type": "anyone", "role": "reader"},
    ).execute()

    # ── Summary sheet ────────────────────────
    _write_summary(sh, results, drive_links)

    # ── Percentile paths sheet ───────────────
    _write_paths(sh, results)

    # ── Histogram sheet ──────────────────────
    _write_histogram_data(sh, results)

    url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit?usp=sharing"
    logger.info(f"  Sheets dashboard updated → {url}")
    return url


def _write_summary(sh, results, drive_links: dict):
    import gspread
    import numpy as np

    try:
        ws = sh.worksheet("📊 Dashboard")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet("📊 Dashboard", rows=60, cols=8)

    ws.clear()
    now = datetime.now().strftime("%d %b %Y  %H:%M IST")
    S0  = results.S0
    T   = results.config.horizon_days

    # ── Build data rows ──────────────────────
    rows = [
        ["HDFC BANK LTD — MONTE CARLO SIMULATION DASHBOARD", "", "", "", "", "", "", ""],
        [f"Last Updated: {now}", "", "", f"Simulations: {results.config.n_simulations:,}", "",
         f"Horizon: {T} trading days", "", ""],
        ["", "", "", "", "", "", "", ""],

        # Price projections
        ["📈 PRICE PROJECTIONS", "Price (₹)", "Change vs Today", "", "", "", "", ""],
        ["Current Price (S₀)",       f"₹{S0:,.2f}",               "—",           "", "", "", "", ""],
        ["Expected Price (Mean)",     f"₹{results.mean_price:,.2f}", f"{(results.mean_price/S0-1)*100:+.1f}%", "", "", "", "", ""],
        ["Median Price (50th pct)",   f"₹{results.median_price:,.2f}", f"{(results.median_price/S0-1)*100:+.1f}%", "", "", "", "", ""],
        ["Best Case (95th pct)",      f"₹{results.ci_95:,.2f}",    f"{(results.ci_95/S0-1)*100:+.1f}%",  "", "", "", "", ""],
        ["Worst Case (5th pct)",      f"₹{results.ci_5:,.2f}",     f"{(results.ci_5/S0-1)*100:+.1f}%",   "", "", "", "", ""],
        ["", "", "", "", "", "", "", ""],

        # Risk
        ["⚠️ RISK METRICS", "Value (₹)", "Meaning", "", "", "", "", ""],
        ["VaR 95%",  f"₹{results.var_95:,.2f}",  "Max loss in 95% of scenarios",  "", "", "", "", ""],
        ["VaR 99%",  f"₹{results.var_99:,.2f}",  "Max loss in 99% of scenarios",  "", "", "", "", ""],
        ["CVaR 95%", f"₹{results.cvar_95:,.2f}", "Avg loss in worst 5% scenarios", "", "", "", "", ""],
        ["", "", "", "", "", "", "", ""],

        # Probabilities
        ["🎯 PROBABILITY METRICS", "Probability", "What it means", "", "", "", "", ""],
        ["P(Price goes UP)",    f"{results.prob_increase:.1%}",    "Chance of any profit",         "", "", "", "", ""],
        ["P(Price > +10%)",     f"{results.prob_10pct_up:.1%}",    "Chance of 10%+ gain",          "", "", "", "", ""],
        ["P(Price < -10%)",     f"{results.prob_10pct_down:.1%}",  "Chance of 10%+ loss",          "", "", "", "", ""],
        ["", "", "", "", "", "", "", ""],

        # Scenarios
        ["📊 SCENARIO ANALYSIS", "Median Price", "Return vs Today", "", "", "", "", ""],
    ]

    for key, label in [("bull", "🟢 Bull Case"), ("base", "🔵 Base Case"), ("bear", "🔴 Bear Case")]:
        import numpy as np
        med = float(np.median(results.scenarios[key][-1, :]))
        rows.append([label, f"₹{med:,.2f}", f"{(med/S0-1)*100:+.1f}%", "", "", "", "", ""])

    rows += [
        ["", "", "", "", "", "", "", ""],
        ["⚙️ MODEL PARAMETERS", "Value", "", "", "", "", "", ""],
        ["Annual Volatility",    f"{results.sigma*(252**0.5)*100:.2f}%", "", "", "", "", "", ""],
        ["Daily Drift (μ)",      f"{results.mu:.6f}",                    "", "", "", "", "", ""],
        ["Distribution",         "Student-t (fat tails)" if results.config.use_fat_tails else "Normal", "", "", "", "", "", ""],
        ["Data Window",          "10 years (rolling)",                   "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", ""],
        ["📁 CHART LINKS (click to view)", "", "", "", "", "", "", ""],
    ]

    label_map = {
        "00_dashboard.png"         : "📊 Full Dashboard",
        "01_price_paths.png"       : "📈 Price Paths Fan Chart",
        "02_histogram.png"         : "📉 Terminal Price Distribution",
        "03_scenarios.png"         : "🎭 Scenario Comparison",
        "04_rolling_volatility.png": "📊 Rolling Volatility",
        "hdfc_monte_carlo.xlsx"    : "📗 Excel Workbook",
    }
    for fname, link in drive_links.items():
        rows.append([label_map.get(fname, fname), link, "", "", "", "", "", ""])

    ws.update("A1", rows)

    # ── Formatting via Sheets API ────────────
    _format_summary_sheet(sh.id, ws, creds=None)


def _format_summary_sheet(spreadsheet_id, ws, creds):
    """Apply colors, bold, font sizes via batchUpdate."""
    # We skip heavy formatting here to keep it simple — gspread basic formatting
    try:
        ws.format("A1", {"textFormat": {"bold": True, "fontSize": 16},
                         "backgroundColor": {"red": 0.06, "green": 0.15, "blue": 0.28}})
        ws.format("A1:H1", {"textFormat": {"foregroundColor": {"red":1,"green":1,"blue":1}},
                             "backgroundColor": {"red": 0.06, "green": 0.15, "blue": 0.28}})
        for row in ["A4", "A11", "A16", "A22", "A27"]:
            ws.format(row, {"textFormat": {"bold": True, "fontSize": 11},
                            "backgroundColor": {"red": 0.12, "green": 0.25, "blue": 0.44},
                            "textFormat": {"foregroundColor": {"red":1,"green":1,"blue":1}, "bold": True}})
    except Exception:
        pass   # formatting is cosmetic — don't crash if it fails


def _write_paths(sh, results):
    import gspread, numpy as np
    try:
        ws = sh.worksheet("📈 Percentile Paths")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet("📈 Percentile Paths", rows=270, cols=7)

    ws.clear()
    T   = results.config.horizon_days
    p5  = np.percentile(results.paths, 5,  axis=1)
    p25 = np.percentile(results.paths, 25, axis=1)
    p50 = np.percentile(results.paths, 50, axis=1)
    p75 = np.percentile(results.paths, 75, axis=1)
    p95 = np.percentile(results.paths, 95, axis=1)

    rows = [["Day", "P5 (Worst)", "P25", "P50 (Median)", "P75", "P95 (Best)", "Current"]]
    for i in range(T):
        rows.append([
            i + 1,
            round(p5[i], 2), round(p25[i], 2), round(p50[i], 2),
            round(p75[i], 2), round(p95[i], 2), round(results.S0, 2)
        ])
    ws.update("A1", rows)
    logger.info("  Sheets: Percentile Paths sheet updated.")


def _write_histogram_data(sh, results):
    import gspread, numpy as np
    try:
        ws = sh.worksheet("📉 Distribution")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet("📉 Distribution", rows=70, cols=3)

    ws.clear()
    final = results.final_prices
    counts, edges = np.histogram(final, bins=50)
    centers = (edges[:-1] + edges[1:]) / 2

    rows = [["Price Bin (₹)", "Frequency", "% of Simulations"]]
    total = len(final)
    for c, v in zip(centers, counts):
        rows.append([round(c, 2), int(v), round(v / total * 100, 2)])
    ws.update("A1", rows)
    logger.info("  Sheets: Distribution sheet updated.")


# ════════════════════════════════════════════
#  SAVE LINKS TO FILE
# ════════════════════════════════════════════

def save_links(drive_links: dict, sheets_url: str):
    data = {
        "last_updated"  : datetime.now().isoformat(),
        "google_sheets" : sheets_url,
        "drive_files"   : drive_links,
    }
    with open("outputs/shareable_links.json", "w") as f:
        json.dump(data, f, indent=2)

    # Also write a simple HTML file with clickable links
    html = f"""<!DOCTYPE html>
<html>
<head><title>HDFC Monte Carlo — Links</title>
<style>
  body {{ font-family: monospace; background: #0d1117; color: #e6edf3; padding: 40px; }}
  a {{ color: #58a6ff; }}
  h2 {{ color: #58a6ff; }}
  li {{ margin: 10px 0; }}
</style>
</head>
<body>
<h2>HDFC Bank Monte Carlo — Shareable Links</h2>
<p>Last updated: {datetime.now().strftime('%d %b %Y %H:%M IST')}</p>
<h3>📊 Live Dashboard</h3>
<ul><li><a href="{sheets_url}" target="_blank">Google Sheets Dashboard (anyone with link can view)</a></li></ul>
<h3>📁 Charts & Files on Google Drive</h3>
<ul>
{"".join(f'<li><a href="{link}" target="_blank">{name}</a></li>' for name, link in drive_links.items())}
</ul>
</body>
</html>"""
    with open("outputs/links.html", "w") as f:
        f.write(html)
    logger.info("  Saved: outputs/links.html  (open in browser for all links)")
