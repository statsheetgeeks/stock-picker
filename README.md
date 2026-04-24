# 📈 Stock Predictor

XGBoost-powered ML pipeline that scores 82 stocks daily across three prediction horizons — **next day, 1 week (5 days), and 1 month (20 days)** — and publishes results to a GitHub Pages dashboard.

---

## How It Works

```
Daily (GitHub Actions, 6 AM CT, Mon–Fri)
  │
  ├─ 1. cache_manager.py   Fetch OHLCV for 82 tickers via yfinance → Parquet files
  ├─ 2. features.py        Build technical indicators + market-relative features
  ├─ 3. train.py           Retrain 3 XGBoost models (one per horizon)
  ├─ 4. predict.py         Score every ticker → docs/predictions.json
  ├─ 5. score_ledger.py    Log today's predictions; grade matured past predictions
  └─ 6. GitHub Pages       Dashboard auto-reads predictions.json & accuracy.json
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/stock-predictor.git
cd stock-predictor
pip install -r requirements.txt
```

### 2. First run (fetches 2 years of data + trains)

```bash
python run_pipeline.py
```

This will take 3–5 minutes on first run (downloading 2 years × 82 tickers).
Subsequent daily runs take ~1–2 minutes.

### 3. Enable GitHub Pages

In your repo settings → **Pages** → Source: **Deploy from branch** → Branch: `main` → Folder: `/docs`

Your dashboard will be live at: `https://YOUR_USERNAME.github.io/stock-predictor/`

### 4. Enable GitHub Actions

The workflow is already at `.github/workflows/daily_run.yml`.
Just push to GitHub — Actions will pick it up automatically and run every weekday at 6 AM CT.

---

## File Structure

```
stock_predictor/
├── .github/
│   └── workflows/
│       └── daily_run.yml        ← Scheduled GitHub Actions runner
├── data/
│   ├── cache/                   ← One .parquet file per ticker (grows daily)
│   │   ├── NVDA.parquet
│   │   ├── MSTR.parquet
│   │   └── ...
│   ├── models/                  ← Trained XGBoost models (.json + .pkl meta)
│   │   ├── xgb_1d.json
│   │   ├── xgb_5d.json
│   │   └── xgb_20d.json
│   └── predictions_ledger.csv   ← Running log of all predictions + actuals
├── docs/                        ← GitHub Pages static site
│   ├── index.html               ← Dashboard
│   ├── predictions.json         ← Today's scores (written by pipeline)
│   └── accuracy.json            ← Accuracy stats (written by pipeline)
├── config.py                    ← Tickers, paths, hyperparameters
├── cache_manager.py             ← yfinance data fetcher + Parquet cache
├── features.py                  ← Technical indicators + feature engineering
├── train.py                     ← XGBoost training + walk-forward validation
├── predict.py                   ← Scoring engine
├── score_ledger.py              ← Predictions log + accuracy grading
├── run_pipeline.py              ← Master orchestration script
└── requirements.txt
```

---

## Features Used

| Category | Features |
|---|---|
| Momentum | RSI-14, MACD, Stochastic %K/%D, Williams %R |
| Trend | EMA (9/21/50), SMA-200, ADX |
| Volatility | Bollinger Bands (width, %B), ATR |
| Volume | OBV, Volume vs 20-day avg |
| Price | Daily return, gap %, intraday range, rolling 5d/20d returns |
| Lags | Prior 5 days of returns |
| Market-relative | Return vs QQQ and SPY (1d, 5d, 20d) |
| Context | QQQ/SPY RSI, volume ratio |

---

## Dashboard

The dashboard at `docs/index.html` reads two JSON files updated daily:

- **predictions.json** — today's probability scores and signals per ticker
- **accuracy.json** — rolling 90-day accuracy stats + graded prediction history

Features:
- Ranked predictions table with probability bars
- Filter by signal strength or search by ticker
- High-confidence filter (P > 60%)
- 90-day accuracy scorecard for all 3 horizons
- Color-coded signals (🟢 Strong Up → 🔴 Strong Down)

---

## Modifying the Universe

Edit the `TICKERS` list in `config.py`. If a ticker maps to a non-standard yfinance symbol (like `VIX` → `^VIX`), add it to `TICKER_MAP`.

---

## Caveats

- Signals are **probabilistic**, not financial advice.
- Walk-forward validation prevents look-ahead bias, but live performance may differ.
- Volatile tickers (MSTR, SOUN, DJT, etc.) will have noisier predictions.
- The model is retrained from scratch daily on the full 2-year (growing) history.
