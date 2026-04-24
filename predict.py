"""
predict.py
Loads trained models and generates today's probability scores for all tickers.

Outputs:
  - docs/predictions.json   → read by the GitHub Pages dashboard
  - Returns a DataFrame for use by score_ledger.py
"""

import json
import logging
from datetime import date

import numpy as np
import pandas as pd

from config import TICKERS, HORIZONS, DOCS_DIR, CONFIDENCE_THRESHOLD
from features import build_feature_matrix, get_feature_cols
from train import load_model

log = logging.getLogger(__name__)

SIGNAL_LABELS = {
    "strong_up":   "🟢 Strong Up",
    "up":          "🟡 Up",
    "neutral":     "⬜ Neutral",
    "down":        "🟡 Down",
    "strong_down": "🔴 Strong Down",
}


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


def run_predictions(all_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build feature matrix, load models, and score every ticker.
    Returns a DataFrame with one row per ticker.
    """
    log.info("Building feature matrix for predictions...")
    matrix = build_feature_matrix(all_data)

    # Load all three models
    models = {}
    for horizon in HORIZONS:
        try:
            model, le, feat_cols = load_model(horizon)
            models[horizon] = (model, le, feat_cols)
        except Exception as exc:
            log.error("Could not load model for %s: %s", horizon, exc)

    if not models:
        raise RuntimeError("No models loaded — run train.py first.")

    rows = []
    for ticker in TICKERS:
        row = get_latest_row(matrix, ticker)
        if row is None:
            log.warning("No feature data for %s — skipping.", ticker)
            continue

        ticker_row: dict = {
            "ticker":    ticker,
            "as_of":     str(row.name.date()) if hasattr(row.name, "date") else str(row.name),
        }

        for horizon, (model, le, feat_cols) in models.items():
            # Encode ticker
            try:
                ticker_enc = le.transform([ticker])[0]
            except ValueError:
                ticker_enc = -1   # unseen ticker fallback

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


def save_predictions_json(predictions: pd.DataFrame) -> None:
    """Write predictions.json for the GitHub Pages dashboard."""
    output = {
        "generated": str(date.today()),
        "tickers":   predictions.to_dict(orient="records"),
    }
    path = DOCS_DIR / "predictions.json"
    with open(path, "w") as fh:
        json.dump(output, fh, indent=2)
    log.info("Saved predictions → %s", path)


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    from cache_manager import refresh_all

    all_data    = refresh_all()
    predictions = run_predictions(all_data)
    save_predictions_json(predictions)

    print("\n" + "═" * 70)
    print(f"{'TICKER':<8} {'P(1d)':>7} {'SIG-1d':<15} {'P(5d)':>7} {'SIG-5d':<15} {'P(20d)':>7}")
    print("─" * 70)
    for _, r in predictions.iterrows():
        print(f"{r['ticker']:<8} "
              f"{r.get('prob_1d',0):>7.3f} {r.get('signal_1d',''):.<15} "
              f"{r.get('prob_5d',0):>7.3f} {r.get('signal_5d',''):.<15} "
              f"{r.get('prob_20d',0):>7.3f}")
