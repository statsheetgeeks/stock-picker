"""
run_pipeline.py
Master orchestration script — run this daily (via GitHub Actions or manually).

Steps:
  1. Refresh OHLCV cache (yfinance)
  2. Build feature matrix
  3. Retrain models on full history
  4. Generate today's predictions → docs/predictions.json
  5. Append today's predictions to the ledger
  6. Grade any matured past predictions
  7. Write accuracy stats to docs/accuracy.json
  8. Print summary report
"""

import json
import logging
import sys
from datetime import date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def main() -> None:
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║          Stock Predictor — Daily Pipeline            ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    # ── 1. Cache refresh ────────────────────────────────────────────────────
    log.info("[1/7] Refreshing OHLCV cache...")
    from cache_manager import refresh_all
    all_data = refresh_all()
    if not all_data:
        log.error("Cache refresh returned no data. Aborting.")
        sys.exit(1)

    # ── 2. Feature matrix ───────────────────────────────────────────────────
    log.info("[2/7] Building feature matrix...")
    from features import build_feature_matrix
    matrix = build_feature_matrix(all_data)

    # ── 3. Retrain models ───────────────────────────────────────────────────
    log.info("[3/7] Training models (all 3 horizons)...")
    from train import train_all
    train_all(matrix)

    # ── 4. Generate predictions ─────────────────────────────────────────────
    log.info("[4/7] Generating today's predictions...")
    from predict import run_predictions, save_predictions_json
    predictions = run_predictions(all_data, feature_matrix=matrix)
    save_predictions_json(predictions)

    # ── 5. Append to ledger ─────────────────────────────────────────────────
    log.info("[5/7] Appending predictions to ledger...")
    from score_ledger import append_predictions
    ledger = append_predictions(predictions, run_date=date.today())

    # ── 6. Grade matured predictions ────────────────────────────────────────
    log.info("[6/7] Grading matured predictions...")
    from score_ledger import grade_predictions
    ledger = grade_predictions(all_data, today=date.today())

    # ── 7. Write accuracy JSON for dashboard ────────────────────────────────
    log.info("[7/7] Writing accuracy stats for dashboard...")
    from score_ledger import accuracy_report, ledger_to_json, ticker_accuracy
    from config import DOCS_DIR

    stats = accuracy_report(ledger)

    accuracy_payload = {
        "generated":       str(date.today()),
        "rolling_90d":     stats,
        "recent_ledger":   ledger_to_json(lookback_days=90),
        "ticker_accuracy": ticker_accuracy(ledger).to_dict(orient="records"),
    }
    acc_path = DOCS_DIR / "accuracy.json"
    with open(acc_path, "w") as fh:
        json.dump(accuracy_payload, fh, indent=2, default=str)
    log.info("Accuracy JSON saved → %s", acc_path)

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║                  Pipeline complete ✓                 ║")
    log.info("╚══════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
