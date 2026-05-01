"""
Central configuration for the stock predictor pipeline.
Edit TICKERS, paths, and model hyperparameters here.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
CACHE_DIR  = DATA_DIR / "cache"
DOCS_DIR   = BASE_DIR / "docs"
MODELS_DIR = DATA_DIR / "models"

LEDGER_PATH      = DATA_DIR / "predictions_ledger.csv"
PREDICTIONS_PATH = DOCS_DIR / "predictions.json"

for _d in [CACHE_DIR, DOCS_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Universe ─────────────────────────────────────────────────────────────────
# VIX must be fetched as ^VIX from yfinance; map display → fetch symbol here
TICKER_MAP = {"VIX": "^VIX"}

TICKERS = [
    "ABTC", "ACHR", "AFRM", "AMD",  "APLD", "APP",  "ASTS", "AUR",
    "AVAH", "BBAI", "BE",   "BFLY", "BTBT", "BYND", "CCL",  "CCO",
    "CG",   "CIFR", "CLNE", "CLOV", "CLSK", "COIN", "COMP", "CORZ",
    "CRWV", "DHC",  "DJT",  "ENVX", "EOSE", "FFAI", "FUBO", "GAP",
    "GETY", "GOSS", "GPUS", "HIMS", "HIVE", "HOOD", "HTZ",  "INDI",
    "IONQ", "JOBY", "KKR",  "MAC",  "MARA", "MSTR", "MVST", "NAKA",
    "NCLH", "NET",  "NVDA", "NVTS", "OCGN", "OLPX", "ONDS", "OPEN",
    "PACB", "PLUG", "PTON", "QQQ",  "QS",   "QUBT", "QXO",  "RIOT",
    "RKLB", "RKT",  "RUN",  "SAIL", "SMR",  "SOFI", "SOUN", "SPY",
    "TNYA", "U",    "ULCC", "UP",   "VG",   "VISN", "VIX",  "VRT",
    "WULF", "XYZ",
]

# Tickers used as market-relative benchmarks (added to every stock's features)
BENCHMARK_TICKERS = ["QQQ", "SPY"]

# ── Data fetch settings ───────────────────────────────────────────────────────
HISTORY_YEARS  = 2          # How many years of history to fetch on first run
CACHE_STALENESS_HOURS = 6   # Re-fetch if cache is older than this

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
TRAIN_FRAC         = 0.75
CONFIDENCE_THRESHOLD = 0.60   # Min probability to show as a strong signal

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
