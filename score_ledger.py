"""
score_ledger.py
Manages the predictions ledger (data/predictions_ledger.csv).

Daily workflow:
  1. append_predictions()  → log today's new signals as pending
  2. grade_predictions()   → fill in actuals for rows that have matured
  3. accuracy_report()     → compute rolling accuracy stats (overall + Strong Up)
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
        df = pd.read_csv(LEDGER_PATH)
        df["date_made"] = pd.to_datetime(df["date_made"]).dt.tz_localize(None)
        for col in LEDGER_COLS:
            if col not in df.columns:
                df[col] = np.nan
        return df[LEDGER_COLS]
    return pd.DataFrame(columns=LEDGER_COLS)


def save_ledger(df: pd.DataFrame) -> None:
    df[LEDGER_COLS].to_csv(LEDGER_PATH, index=False)
    log.info("Ledger saved → %s (%d rows)", LEDGER_PATH, len(df))


# ── Append today's predictions ─────────────────────────────────────────────────

def append_predictions(
    predictions: pd.DataFrame, run_date: date | None = None
) -> pd.DataFrame:
    """
    Add today's predictions to the ledger as pending rows.
    Tickers already logged for today are skipped (idempotent).
    """
    run_date = run_date or date.today()
    ledger   = load_ledger()

    already_logged: set = set()
    if not ledger.empty:
        today_rows     = ledger[ledger["date_made"].dt.date == run_date]
        already_logged = set(today_rows["ticker"])

    new_rows = []
    for _, pred in predictions.iterrows():
        ticker = pred["ticker"]
        if ticker in already_logged:
            continue
        new_rows.append({
            "date_made":  pd.Timestamp(run_date),
            "ticker":     ticker,
            "as_of":      pred.get("as_of", ""),
            "prob_1d":    pred.get("prob_1d",  np.nan),
            "signal_1d":  pred.get("signal_1d", ""),
            "prob_5d":    pred.get("prob_5d",  np.nan),
            "signal_5d":  pred.get("signal_5d", ""),
            "prob_20d":   pred.get("prob_20d", np.nan),
            "signal_20d": pred.get("signal_20d", ""),
            # actuals filled in later by grade_predictions()
            "actual_1d":  np.nan, "correct_1d":  np.nan,
            "actual_5d":  np.nan, "correct_5d":  np.nan,
            "actual_20d": np.nan, "correct_20d": np.nan,
        })

    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)
        log.info("Appended %d new predictions for %s.", len(new_rows), run_date)
    else:
        log.info("No new predictions to append for %s.", run_date)

    save_ledger(ledger)
    return ledger


# ── Grade matured predictions ──────────────────────────────────────────────────

def _build_trading_calendar(price_data: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """
    Build a sorted index of all known trading dates from any available ticker.
    Used to advance N trading days without relying on calendar arithmetic.
    """
    all_dates: set = set()
    for df in price_data.values():
        idx = pd.to_datetime(df.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        all_dates.update(idx.normalize())
    return pd.DatetimeIndex(sorted(all_dates))


def _build_close_lookup(
    price_data: dict[str, pd.DataFrame],
) -> dict[str, pd.Series]:
    """
    Build a dict of {ticker → pd.Series(close, index=DatetimeIndex)} for O(1)
    close-price lookup during grading.  Replaces repeated DataFrame scans.
    """
    lookup: dict[str, pd.Series] = {}
    for ticker, df in price_data.items():
        idx = pd.to_datetime(df.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        s = pd.Series(df["close"].values, index=idx.normalize(), dtype=float)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        lookup[ticker] = s
    return lookup


def _get_close_from_lookup(
    series: pd.Series, target_date: pd.Timestamp
) -> float | None:
    """Return the close on or after target_date from a pre-built lookup Series."""
    sub = series[series.index >= target_date]
    return float(sub.iloc[0]) if not sub.empty else None


def grade_predictions(
    price_data: dict[str, pd.DataFrame], today: date | None = None
) -> pd.DataFrame:
    """
    For every ledger row whose horizon has matured, fetch the actual close
    and compute whether the price went up.

    Performance notes vs the old implementation:
      - Trading calendar and close-price lookups are built once (not per row).
      - Per-row work is purely in-memory O(log n) lookups.
    """
    today  = today or date.today()
    ledger = load_ledger()

    if ledger.empty:
        log.info("Ledger is empty — nothing to grade.")
        return ledger

    trading_cal  = _build_trading_calendar(price_data)
    close_lookup = _build_close_lookup(price_data)

    graded = 0
    for idx, row in ledger.iterrows():
        made_date = (
            row["date_made"].date()
            if hasattr(row["date_made"], "date")
            else row["date_made"]
        )
        ticker = row["ticker"]

        if ticker not in close_lookup:
            continue

        # Use as_of as grading base so evaluation matches the training objective:
        # target = close[as_of + N] > close[as_of].
        # Falls back to made_date for legacy rows that lack as_of.
        as_of_val = row.get("as_of", "")
        if as_of_val and not (isinstance(as_of_val, float) and np.isnan(as_of_val)):
            base_date = pd.Timestamp(as_of_val).normalize()
        else:
            base_date = pd.Timestamp(made_date).normalize()

        base_close = _get_close_from_lookup(close_lookup[ticker], base_date)
        if base_close is None:
            continue

        for label, n_days in HORIZONS.items():
            actual_col  = f"actual_{label}"
            correct_col = f"correct_{label}"

            if pd.notna(row[actual_col]):
                continue   # already graded

            # Find the nth trading day after base_date using the calendar
            future_dates = trading_cal[trading_cal > base_date]
            if len(future_dates) < n_days:
                continue   # not enough history yet
            target_ts = future_dates[n_days - 1]

            if target_ts.date() > today:
                continue   # horizon hasn't matured yet

            future_close = _get_close_from_lookup(close_lookup[ticker], target_ts)
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

def accuracy_report(
    ledger: pd.DataFrame | None = None, lookback_days: int = 90
) -> dict:
    """
    Compute rolling accuracy metrics for the dashboard.

    Returns a dict keyed by horizon label, each containing:
      - accuracy      : overall accuracy (prob >= 0.5 correct)
      - n             : graded count
      - hc_accuracy   : high-confidence accuracy (prob >= CONFIDENCE_THRESHOLD or <= 1-threshold)
      - hc_n          : high-confidence count
      - strong_up_accuracy : of all 'strong_up' 1d calls, % where price actually went up
                             at this horizon (1d, 5d, 20d) — uses actual_Xd directly
      - strong_up_n        : number of strong_up calls with a graded outcome at this horizon
    """
    if ledger is None:
        ledger = load_ledger()

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
    recent = ledger[ledger["date_made"] >= cutoff]

    # Rows flagged as Strong Up on the 1d signal
    strong_up_rows = recent[recent["signal_1d"] == "strong_up"]

    report: dict = {}

    print(f"\n{'═' * 65}")
    print(f"  Model Accuracy Report — last {lookback_days} days")
    print(f"{'═' * 65}")

    for label in HORIZONS:
        correct_col = f"correct_{label}"
        actual_col  = f"actual_{label}"
        prob_col    = f"prob_{label}"

        graded = recent[recent[correct_col].notna()]

        # Overall accuracy
        if graded.empty:
            print(f"  {label:<5}  No graded predictions yet")
            report[label] = {
                "accuracy":           None, "n":            0,
                "hc_accuracy":        None, "hc_n":         0,
                "strong_up_accuracy": None, "strong_up_n":  0,
            }
            continue

        overall_acc = graded[correct_col].mean()
        n           = len(graded)

        # High-confidence accuracy (prob pushed strongly in either direction)
        hc_mask = (
            (graded[prob_col] >= CONFIDENCE_THRESHOLD) |
            (graded[prob_col] <= (1 - CONFIDENCE_THRESHOLD))
        )
        hc      = graded[hc_mask]
        hc_acc  = hc[correct_col].mean() if not hc.empty else float("nan")
        hc_n    = len(hc)

        # Strong Up accuracy — using actual_Xd (did price go up?) not correct_Xd
        # (was model right?), because the user wants to know whether a Strong Up
        # 1d call translated to real upward movement at each horizon.
        su_graded     = strong_up_rows[strong_up_rows[actual_col].notna()]
        su_acc        = su_graded[actual_col].mean() if not su_graded.empty else float("nan")
        su_n          = len(su_graded)

        hc_str = f"{hc_acc:.1%} (n={hc_n:,})" if not np.isnan(hc_acc) else "— (n=0)"
        su_str = f"{su_acc:.1%} (n={su_n:,})" if su_n > 0 else "— (n=0)"

        print(
            f"  {label:<5}  Overall: {overall_acc:.1%} (n={n:,})  |  "
            f"High-conf: {hc_str}  |  Strong Up: {su_str}"
        )

        report[label] = {
            "accuracy":           round(overall_acc, 4),
            "n":                  n,
            "hc_accuracy":        round(hc_acc, 4) if not np.isnan(hc_acc) else None,
            "hc_n":               hc_n,
            "strong_up_accuracy": round(su_acc, 4) if not np.isnan(su_acc) else None,
            "strong_up_n":        su_n,
        }

    print(f"{'═' * 65}\n")
    return report


# ── Persistence helper for the dashboard ──────────────────────────────────────

def ledger_to_json(lookback_days: int = 90) -> list[dict]:
    """Return the recent ledger as a list of dicts (NaN → null) for the dashboard."""
    ledger  = load_ledger()
    cutoff  = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
    recent  = ledger[ledger["date_made"] >= cutoff].copy()
    recent["date_made"] = recent["date_made"].dt.strftime("%Y-%m-%d")
    records = recent.to_dict(orient="records")
    for record in records:
        for key, val in record.items():
            if isinstance(val, float) and np.isnan(val):
                record[key] = None
    return records


# ── Accuracy by ticker ─────────────────────────────────────────────────────────

def ticker_accuracy(ledger: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Return per-ticker accuracy with per-horizon sample sizes.

    Previously, all accuracy figures used the 1d grading count as 'n', which
    was misleading for 5d/20d figures (those mature later).  Now each horizon
    has its own graded count so the sample size is accurate.
    """
    if ledger is None:
        ledger = load_ledger()

    rows = []
    for ticker, group in ledger.groupby("ticker"):
        record: dict = {"ticker": ticker}
        for label in HORIZONS:
            actual_col  = f"actual_{label}"
            correct_col = f"correct_{label}"
            graded = group[group[correct_col].notna()]
            record[f"n_{label}"]        = len(graded)
            record[f"accuracy_{label}"] = graded[correct_col].mean() if not graded.empty else None
        rows.append(record)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    # Sort by 1d accuracy descending, tickers with no data at bottom
    result = result.sort_values("accuracy_1d", ascending=False, na_position="last").reset_index(drop=True)

    # Replace NaN with None for clean JSON serialisation downstream
    return result.where(result.notna(), other=None)


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    from cache_manager import load_all_cached

    price_data = load_all_cached()
    ledger     = grade_predictions(price_data)
    accuracy_report(ledger)

    ta_df = ticker_accuracy(ledger)
    if not ta_df.empty:
        print(ta_df.to_string(index=False))
