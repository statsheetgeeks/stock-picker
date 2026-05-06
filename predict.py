"""
predict.py
Loads trained models and generates today's probability scores for all tickers.

Outputs:
  - docs/predictions.json   → read by the GitHub Pages dashboard
  - Returns a DataFrame for use by score_ledger.py

New in this version:
  - last_close column included per ticker for dashboard display
  - QQQ and SPY benchmark rows included in predictions.json (dashboard pins them
    to the top of the table as market-reference rows)
  - Unseen-ticker encoding falls back to the median encoded value (not -1)
"""

import json
import logging
from datetime import date

import numpy as np
import pandas as pd

from config import TICKERS, BENCHMARK_TICKERS, HORIZONS, DOCS_DIR, CONFIDENCE_THRESHOLD
from features import build_feature_matrix
from train import load_model

log = logging.getLogger(__name__)


def classify_signal(prob: float) -> str:
    if prob >= CONFIDENCE_THRESHOLD:
        return "strong_up"
    if prob >= 0.53:
        return "up"
    if prob <= (1 - CONFIDENCE_THRESHOLD):
        return "strong_down"
    if prob <= 0.47:
        return "down"
    return "neutral"


def get_latest_row(feature_matrix: pd.DataFrame, ticker: str) -> pd.Series | None:
    """Return the most recent feature row for a ticker (today or last trading day)."""
    sub = feature_matrix[feature_matrix["ticker"] == ticker]
    if sub.empty:
        return None
    return sub.iloc[-1]


def get_last_close(ticker: str, all_data: dict[str, pd.DataFrame]) -> float | None:
    """Return the most recent closing price for ticker from the OHLCV cache."""
    df = all_data.get(ticker)
    if df is None or df.empty:
        return None
    return float(df["close"].iloc[-1])


def run_predictions(
    all_data: dict[str, pd.DataFrame],
    feature_matrix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build feature matrix, load models, and score every ticker in TICKERS.
    Pass a pre-built feature_matrix to skip rebuilding it (saves time in pipeline).
    Returns a DataFrame with one row per ticker, sorted by prob_1d descending.
    """
    if feature_matrix is None:
        log.info("Building feature matrix for predictions...")
        feature_matrix = build_feature_matrix(all_data)
    matrix = feature_matrix

    # ── Load all three models ──────────────────────────────────────────────────
    models: dict[str, tuple] = {}
    for horizon in HORIZONS:
        try:
            model, le, feat_cols = load_model(horizon)
            models[horizon] = (model, le, feat_cols)
        except Exception as exc:
            log.error("Could not load model for %s: %s", horizon, exc)

    if not models:
        raise RuntimeError("No models loaded — run train.py first.")

    # Pre-compute median ticker encoding as a safe fallback for unseen tickers.
    # Using -1 (old approach) feeds a value outside the training distribution.
    first_le    = next(iter(models.values()))[1]
    median_enc  = int(np.median(first_le.transform(first_le.classes_)))

    # ── Score each ticker ─────────────────────────────────────────────────────
    rows = []
    for ticker in TICKERS:
        row = get_latest_row(matrix, ticker)
        if row is None:
            log.warning("No feature data for %s — skipping.", ticker)
            continue

        ticker_row: dict = {
            "ticker":     ticker,
            "as_of":      str(row.name.date()) if hasattr(row.name, "date") else str(row.name),
            "last_close": get_last_close(ticker, all_data),
        }

        for horizon, (model, le, feat_cols) in models.items():
            # Encode ticker — fall back to median encoding if unseen
            try:
                ticker_enc = le.transform([ticker])[0]
            except ValueError:
                ticker_enc = median_enc
                log.warning(
                    "Ticker %s was not seen during training; using median encoding "
                    "(%d) as fallback. Predictions for this ticker may be less reliable.",
                    ticker, median_enc,
                )

            feat_vals = []
            for col in feat_cols:
                if col == "ticker_enc":
                    feat_vals.append(ticker_enc)
                else:
                    feat_vals.append(row.get(col, np.nan))

            X = pd.DataFrame([feat_vals], columns=feat_cols)
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

            prob = float(model.predict_proba(X)[0, 1])
            ticker_row[f"prob_{horizon}"]   = round(prob, 4)
            ticker_row[f"signal_{horizon}"] = classify_signal(prob)

        rows.append(ticker_row)

    predictions = pd.DataFrame(rows)
    predictions = predictions.sort_values("prob_1d", ascending=False).reset_index(drop=True)
    log.info("Predictions generated for %d tickers.", len(predictions))
    return predictions


def build_benchmark_rows(all_data: dict[str, pd.DataFrame]) -> list[dict]:
    """
    Build minimal reference rows for QQQ and SPY to be pinned at the top of
    the dashboard.  These rows carry only last_close — no model scores — so the
    dashboard can display current benchmark prices alongside the predictions.
    """
    rows = []
    for bm in BENCHMARK_TICKERS:
        close = get_last_close(bm, all_data)
        rows.append({
            "ticker":      bm,
            "is_benchmark": True,
            "last_close":  close,
            "as_of":       None,
        })
    return rows


def save_predictions_json(
    predictions: pd.DataFrame,
    all_data: dict[str, pd.DataFrame],
) -> None:
    """Write predictions.json for the GitHub Pages dashboard."""
    benchmark_rows = build_benchmark_rows(all_data)
    ticker_records = predictions.to_dict(orient="records")

    output = {
        "generated":   str(date.today()),
        "benchmarks":  benchmark_rows,
        "tickers":     ticker_records,
    }
    path = DOCS_DIR / "predictions.json"
    with open(path, "w") as fh:
        json.dump(output, fh, indent=2)
    log.info("Saved predictions → %s  (%d tickers)", path, len(ticker_records))


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    from cache_manager import refresh_all

    all_data    = refresh_all()
    predictions = run_predictions(all_data)
    save_predictions_json(predictions, all_data)

    print("\n" + "═" * 75)
    print(f"{'TICKER':<8} {'CLOSE':>8} {'P(1d)':>7} {'SIG-1d':<14} "
          f"{'P(5d)':>7} {'SIG-5d':<14} {'P(20d)':>7}")
    print("─" * 75)
    for _, r in predictions.iterrows():
        close_str = f"${r['last_close']:.2f}" if r.get("last_close") else "—"
        print(
            f"{r['ticker']:<8} {close_str:>8} "
            f"{r.get('prob_1d',  0):>7.3f} {r.get('signal_1d',  ''):.<14} "
            f"{r.get('prob_5d',  0):>7.3f} {r.get('signal_5d',  ''):.<14} "
            f"{r.get('prob_20d', 0):>7.3f}"
        )
