"""
score_ledger.py
Manages the predictions ledger (data/predictions_ledger.csv).

Daily workflow:
  1. append_predictions()  → log today's new signals as pending
  2. grade_predictions()   → fill in actuals for rows that have matured
  3. accuracy_report()     → print rolling accuracy stats
"""

import logging
from datetime import date

import numpy as np
import pandas as pd

from config import CONFIDENCE_THRESHOLD, HORIZONS, LEDGER_PATH, LEDGER_COLS

log = logging.getLogger(__name__)


# ── Ledger I/O ─────────────────────────────────────────────────────────────────

def load_ledger() -> pd.DataFrame:
    if LEDGER_PATH.exists():
        df = pd.read_csv(LEDGER_PATH, parse_dates=["date_made"])
        # Ensure all expected columns exist
        for col in LEDGER_COLS:
            if col not in df.columns:
                df[col] = np.nan
        return df[LEDGER_COLS]
    return pd.DataFrame(columns=LEDGER_COLS)


def save_ledger(df: pd.DataFrame) -> None:
    df[LEDGER_COLS].to_csv(LEDGER_PATH, index=False)
    log.info("Ledger saved → %s (%d rows)", LEDGER_PATH, len(df))


# ── Append today's predictions ─────────────────────────────────────────────────

def append_predictions(predictions: pd.DataFrame,
                        run_date: date | None = None) -> pd.DataFrame:
    """
    Add today's predictions to the ledger as pending rows.
    Skips tickers already logged for today.
    """
    run_date = run_date or date.today()
    ledger   = load_ledger()

    already_logged = set()
    if not ledger.empty:
        today_rows = ledger[ledger["date_made"].dt.date == run_date]
        already_logged = set(today_rows["ticker"])

    new_rows = []
    for _, pred in predictions.iterrows():
        ticker = pred["ticker"]
        if ticker in already_logged:
            continue

        row = {
            "date_made":  run_date,
            "ticker":     ticker,
            "prob_1d":    pred.get("prob_1d",  np.nan),
            "signal_1d":  pred.get("signal_1d", ""),
            "prob_5d":    pred.get("prob_5d",  np.nan),
            "signal_5d":  pred.get("signal_5d", ""),
            "prob_20d":   pred.get("prob_20d", np.nan),
            "signal_20d": pred.get("signal_20d", ""),
            # actuals filled in later
            "actual_1d":  np.nan, "correct_1d":  np.nan,
            "actual_5d":  np.nan, "correct_5d":  np.nan,
            "actual_20d": np.nan, "correct_20d": np.nan,
        }
        new_rows.append(row)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        ledger = pd.concat([ledger, new_df], ignore_index=True)
        log.info("Appended %d new predictions for %s.", len(new_rows), run_date)
    else:
        log.info("No new predictions to append for %s.", run_date)

    save_ledger(ledger)
    return ledger


# ── Grade matured predictions ──────────────────────────────────────────────────

def _trading_days_later(start: date, n: int,
                         price_data: dict[str, pd.DataFrame]) -> date | None:
    """
    Approximate N trading days after start by using any liquid ticker's index.
    Returns None if not enough history exists.
    """
    for ticker, df in price_data.items():
        idx = pd.DatetimeIndex(df.index)
        future = idx[idx.date > start]  # type: ignore[attr-defined]
        if len(future) >= n:
            return future[n - 1].date()
    return None


def _get_close(ticker: str, target_date: date,
               price_data: dict[str, pd.DataFrame]) -> float | None:
    """Return the closing price for ticker on or nearest after target_date."""
    if ticker not in price_data:
        return None
    df  = price_data[ticker]
    sub = df[df.index.date >= target_date]  # type: ignore[attr-defined]
    if sub.empty:
        return None
    return float(sub.iloc[0]["close"])


def grade_predictions(price_data: dict[str, pd.DataFrame],
                       today: date | None = None) -> pd.DataFrame:
    """
    For every ledger row whose horizon has matured, fetch the actual close
    and compute whether the prediction was correct.
    """
    today  = today or date.today()
    ledger = load_ledger()

    if ledger.empty:
        log.info("Ledger is empty — nothing to grade.")
        return ledger

    graded = 0
    for idx, row in ledger.iterrows():
        made_date = row["date_made"].date() if hasattr(row["date_made"], "date") \
                    else row["date_made"]
        ticker    = row["ticker"]
        base_close = _get_close(ticker, made_date, price_data)
        if base_close is None:
            continue

        for label, n_days in HORIZONS.items():
            actual_col  = f"actual_{label}"
            correct_col = f"correct_{label}"
            if pd.notna(row[actual_col]):
                continue   # already graded

            target_date = _trading_days_later(made_date, n_days, price_data)
            if target_date is None or target_date > today:
                continue   # not matured yet

            future_close = _get_close(ticker, target_date, price_data)
            if future_close is None:
                continue

            went_up = int(future_close > base_close)
            ledger.at[idx, actual_col]  = went_up
            ledger.at[idx, correct_col] = int(
                (row[f"prob_{label}"] >= 0.5) == bool(went_up)
            )
            graded += 1

    if graded:
        log.info("Graded %d prediction outcomes.", graded)
        save_ledger(ledger)
    else:
        log.info("No new outcomes to grade today.")

    return ledger


# ── Accuracy report ────────────────────────────────────────────────────────────

def accuracy_report(ledger: pd.DataFrame | None = None,
                    lookback_days: int = 90) -> dict:
    """
    Compute rolling accuracy metrics and print a summary.
    Returns a dict with per-horizon accuracy and counts.
    """
    if ledger is None:
        ledger = load_ledger()

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
    recent = ledger[ledger["date_made"] >= cutoff]

    report = {}
    print(f"\n{'═'*55}")
    print(f"  Model Accuracy Report (last {lookback_days} days)")
    print(f"{'═'*55}")

    for label in HORIZONS:
        correct_col = f"correct_{label}"
        graded = recent[recent[correct_col].notna()]
        if graded.empty:
            print(f"  {label:<5}  No graded predictions yet")
            continue

        overall_acc = graded[correct_col].mean()
        n           = len(graded)

        # High-confidence only
        prob_col = f"prob_{label}"
        hc = graded[
            (graded[prob_col] >= CONFIDENCE_THRESHOLD) |
            (graded[prob_col] <= (1 - CONFIDENCE_THRESHOLD))
        ]
        hc_acc = hc[correct_col].mean() if not hc.empty else float("nan")

        print(f"  {label:<5}  Overall: {overall_acc:.1%} (n={n:,})  |  "
              f"High-conf: {hc_acc:.1%} (n={len(hc):,})" if not np.isnan(hc_acc)
              else f"  {label:<5}  Overall: {overall_acc:.1%} (n={n:,})  |  "
                   f"High-conf: — (n=0)")

        report[label] = {
            "accuracy":      round(overall_acc, 4),
            "n":             n,
            "hc_accuracy":   round(hc_acc, 4) if not np.isnan(hc_acc) else None,
            "hc_n":          len(hc),
        }

    print(f"{'═'*55}\n")
    return report


# ── Persistence helper for the dashboard ──────────────────────────────────────

def ledger_to_json(lookback_days: int = 90) -> list[dict]:
    """Return the recent ledger as a list of dicts for the dashboard."""
    ledger = load_ledger()
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
    recent = ledger[ledger["date_made"] >= cutoff].copy()
    recent["date_made"] = recent["date_made"].dt.strftime("%Y-%m-%d")
    return recent.to_dict(orient="records")


# ── Accuracy by ticker ─────────────────────────────────────────────────────────

def ticker_accuracy(ledger: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return per-ticker accuracy for the 1d horizon (most reliable signal)."""
    if ledger is None:
        ledger = load_ledger()
    graded = ledger[ledger["correct_1d"].notna()]
    if graded.empty:
        return pd.DataFrame()
    return (
        graded.groupby("ticker")
        .agg(
            n=("correct_1d", "count"),
            accuracy_1d=("correct_1d", "mean"),
            accuracy_5d=("correct_5d", "mean"),
        )
        .sort_values("accuracy_1d", ascending=False)
        .reset_index()
    )


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    from config import CONFIDENCE_THRESHOLD
    from cache_manager import load_all_cached

    price_data = load_all_cached()
    ledger     = grade_predictions(price_data)
    accuracy_report(ledger)

    ta = ticker_accuracy(ledger)
    if not ta.empty:
        print(ta.to_string(index=False))
