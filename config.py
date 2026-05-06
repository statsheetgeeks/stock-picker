"""
Central configuration for the stock predictor pipeline.
Edit TICKERS, paths, and model hyperparameters here.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
CACHE_DIR  = DATA_DIR / "cache"
DOCS_DIR   = BASE_DIR / "docs"
MODELS_DIR = DATA_DIR / "models"

LEDGER_PATH = DATA_DIR / "predictions_ledger.csv"

for _d in [CACHE_DIR, DOCS_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Universe ──────────────────────────────────────────────────────────────────
# Tickers that require a non-standard yfinance symbol (display → fetch).
TICKER_MAP = {"VIX": "^VIX"}

# QQQ and SPY are intentionally NOT in TICKERS — they are benchmark reference
# tickers only (see BENCHMARK_TICKERS below).  They appear pinned at the top of
# the dashboard as market-context rows, but no standalone prediction is scored
# for them because relative-to-self features would be identically zero.
TICKERS = [
    "ABTC", "ACHR", "AFRM", "AMD",  "APLD", "APP",  "ASTS", "AUR",
    "AVAH", "BBAI", "BE",   "BFLY", "BTBT", "BYND", "CCL",  "CCO",
    "CG",   "CIFR", "CLNE", "CLOV", "CLSK", "COIN", "COMP", "CORZ",
    "CRWV", "DHC",  "DJT",  "ENVX", "EOSE", "FFAI", "FUBO", "GAP",
    "GETY", "GOSS", "GPUS", "HIMS", "HIVE", "HOOD", "HTZ",  "INDI",
    "IONQ", "JOBY", "KKR",  "MAC",  "MARA", "MSTR", "MVST", "NAKA",
    "NCLH", "NET",  "NVDA", "NVTS", "OCGN", "OLPX", "ONDS", "OPEN",
    "PACB", "PLUG", "PTON", "QS",   "QUBT", "QXO",  "RIOT",
    "RKLB", "RKT",  "RUN",  "SAIL", "SMR",  "SOFI", "SOUN",
    "TNYA", "U",    "ULCC", "UP",   "VG",   "VISN", "VIX",  "VRT",
    "WULF", "XYZ",
]

# Benchmark tickers — cached and used for market-relative feature columns.
# Their closing prices also appear in predictions.json for dashboard display.
BENCHMARK_TICKERS = ["QQQ", "SPY"]

# ── Data fetch settings ───────────────────────────────────────────────────────
HISTORY_YEARS = 2   # How many years of history to fetch on first run

# ── Prediction horizons ───────────────────────────────────────────────────────
HORIZONS = {
    "1d":  1,
    "5d":  5,
    "20d": 20,
}

# ── Feature settings ──────────────────────────────────────────────────────────
LAG_DAYS       = 5    # Number of prior-day return lags to include
RSI_PERIOD     = 14
MACD_FAST      = 12
MACD_SLOW      = 26
MACD_SIGNAL    = 9
BB_PERIOD      = 20
ATR_PERIOD     = 14
EMA_PERIODS    = [9, 21, 50]
SMA_PERIODS    = [200]
VOL_AVG_PERIOD = 20   # Volume vs N-day average

# ── Model settings ────────────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators":      500,
    "max_depth":         4,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  5,
    "gamma":             0.1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "eval_metric":       "logloss",
    "random_state":      42,
    "n_jobs":            -1,
}

# Walk-forward: train on first TRAIN_FRAC of history, test on remainder
TRAIN_FRAC           = 0.75
CONFIDENCE_THRESHOLD = 0.60   # Min probability to flag as a strong signal

# ── Ledger columns ────────────────────────────────────────────────────────────
LEDGER_COLS = [
    "date_made", "ticker", "as_of",
    "prob_1d",  "signal_1d",
    "prob_5d",  "signal_5d",
    "prob_20d", "signal_20d",
    "actual_1d",  "correct_1d",
    "actual_5d",  "correct_5d",
    "actual_20d", "correct_20d",
]
