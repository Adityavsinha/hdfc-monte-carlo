# QuantEdge Analytics — Vercel Deployment Guide

## Complete Migration from Firebase to Vercel

This guide covers deploying your QuantEdge Analytics platform to Vercel, which offers:

- **Free tier:** 100GB bandwidth/month, unlimited deployments
- **Automatic SSL:** Free HTTPS for custom domains
- **Edge functions:** Run Python/Node at the edge
- **Git integration:** Automatic deployments from GitHub

---

## Step 1: Prepare Your Repository

### 1.1 Create vercel.json (root directory)

```json
// vercel.json
{
  "builds": [
    {
      "src": "docs/**",
      "use": "@vercel/static"
    },
    {
      "src": "scripts/**/*.py",
      "use": "@vercel/python",
      "config": {
        "pythonVersion": "3.11"
      }
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/scripts/$1"
    },
    {
      "handle": "filesystem"
    },
    {
      "src": "/(.*)",
      "dest": "/docs/index.html"
    }
  ]
}
```

### 1.2 Create API endpoint for signals

```python
# scripts/api/signals.py
from quant_engine import run_full_pipeline
import json

def handler(request, context):
    # Your existing pipeline logic
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"message": "QuantEdge API", "status": "active"})
    }
```

---

## Step 2: Deploy to Vercel

### Option A: Via Vercel CLI (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy from project root
cd c:\Projects\quantedge
vercel

# Follow prompts:
# - Set up and deploy? Yes
# - Which scope? Your username
# - Link to existing project? No (create new)
# - Project name: quantedge-analytics
# - Directory? ./
# - Want to override settings? No
```

### Option B: Via GitHub (Automatic Deployments)

1. **Push code to GitHub:**

   ```bash
   git init
   git add .
   git commit -m "Initial QuantEdge deployment"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/quantedge.git
   git push -u origin main
   ```

2. **Connect to Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Click "Add New..." → "Project"
   - Import your GitHub repository
   - Configure:
     - Framework Preset: Other
     - Build Command: (leave empty)
     - Output Directory: docs
   - Click "Deploy"

---

## Step 3: Configure Custom Domain

### 3.1 Add Domain in Vercel

1. Go to your project → Settings → Domains
2. Enter `quantedgeanalytics.co.in`
3. Add DNS records as instructed by Vercel

### 3.2 Update DNS (at your domain provider)

| Type  | Name | Value                |
| ----- | ---- | -------------------- |
| CNAME | @    | cname.vercel-dns.com |
| CNAME | www  | cname.vercel-dns.com |

---

## Step 4: Environment Variables

Set these in Vercel dashboard → Settings → Environment Variables:

```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHANNEL_ID=@quantedge_signals
NIFTY_API_KEY=your_nse_api_key
```

---

## Step 5: Schedule Daily Updates (GitHub Actions)

Create `.github/workflows/daily.yml`:

```yaml
name: Daily Pipeline Run

on:
  schedule:
    - cron: '0 6 * * *' # 6 AM IST daily
  workflow_dispatch:

jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run pipeline
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
        run: python run_pipeline.py

      - name: Commit and push results
        run: |
          git config --local user.email "github-actions@quantedge.co.in"
          git config --local user.name "GitHub Actions"
          git add docs/
          git commit -m "Update: $(date '+%Y-%m-%d %H:%M')" || exit 0
          git push
```

---

## Step 6: Compare Firebase vs Vercel

| Feature                  | Firebase                      | Vercel                               |
| ------------------------ | ----------------------------- | ------------------------------------ |
| **Static Hosting**       | ✅                            | ✅                                   |
| **Serverless Functions** | ✅ (Cloud Functions)          | ✅ (Edge Functions)                  |
| **Free Tier**            | 1GB storage, 100K invocations | 100GB bandwidth                      |
| **SSL**                  | ✅ (auto)                     | ✅ (auto)                            |
| **Custom Domain**        | ✅                            | ✅                                   |
| **Python Support**       | ❌ (needs Cloud Run)          | ✅ (via Python runtime)              |
| **CI/CD**                | Limited                       | Excellent (GitHub integration)       |
| **Analytics**            | ✅ (Firebase Analytics)       | ✅ (Vercel Analytics)                |
| **Database**             | Firestore (built-in)          | Need external (Supabase/PlanetScale) |

---

## Step 7: Optional — Add Database (Supabase)

For user authentication and signal history:

1. **Sign up at [supabase.com](https://supabase.com)** (free tier)
2. **Create new project:** `quantedge-db`
3. **Get connection string:**
   - Settings → Database → Connection String
4. **Add to Vercel env vars:**

   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_anon_key
   ```

5. **Update your code to use Supabase:**
   ```python
   import supabase
   client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
   ```

---

## Step 8: Verify Deployment

After deployment, verify:

1. **Main page:** `https://quantedgeanalytics.co.in`
2. **Screener:** `https://quantedgeanalytics.co.in/docs/screener_data.json`
3. **API (if added):** `https://quantedgeanalytics.co.in/api/signals`
4. **Sitemap:** `https://quantedgeanalytics.co.in/sitemap.xml`

---

## Troubleshooting

### Issue: Python scripts not running

**Solution:** Add `vercel.json` with Python runtime config

### Issue: Static files not loading

**Solution:** Ensure `docs/` is the output directory

### Issue: Environment variables not working

**Solution:** Redeploy after adding env vars (Vercel requires redeploy for env changes)

### Issue: Build failing

**Solution:** Check Build Logs in Vercel dashboard

---

## Cost Summary

| Component             | Monthly Cost             |
| --------------------- | ------------------------ |
| Vercel Pro (optional) | $20/month                |
| Supabase (optional)   | $0 (free tier)           |
| GitHub Actions        | $0 (2000 min/month free) |
| **Total**             | **$0** (with free tiers) |

---

## Next Steps

1. ✅ Deploy to Vercel
2. ✅ Configure custom domain
3. ✅ Set up GitHub Actions for daily runs
4. ✅ (Optional) Add Supabase for user auth
5. ✅ Submit sitemap to Google Search Console

---

_Last updated: April 2026_
