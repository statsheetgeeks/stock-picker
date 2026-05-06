"""
cache_manager.py
Fetches OHLCV data from yfinance and stores one Parquet file per ticker.
Skips re-fetch if the cached data is already current (see is_stale()).
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import (
    TICKERS, BENCHMARK_TICKERS, TICKER_MAP, CACHE_DIR,
    HISTORY_YEARS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# All tickers that need a cache file = scored tickers + benchmarks
ALL_CACHE_TICKERS = list(dict.fromkeys(TICKERS + BENCHMARK_TICKERS))


def cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.parquet"


def is_stale(ticker: str) -> bool:
    """
    Return True if the cache file is missing or its data is out of date.

    Uses the MAX DATE inside the parquet data rather than file mtime.
    File mtime is unreliable: git checkout resets all mtimes to now, which
    would treat stale committed files as fresh.

    Logic:
      - Missing file                             → stale
      - Empty file                               → stale
      - Latest data date >= today               → fresh (already have today)
      - Latest data date is yesterday (weekday) → fresh
      - Latest data date is Friday, today is Mon/Tue/Wed (≤4 days) → fresh (weekend/holiday gap)
      - Anything older                           → stale
    """
    p = cache_path(ticker)
    if not p.exists():
        return True
    try:
        df = pd.read_parquet(p, columns=["close"])
        if df.empty:
            return True
        latest = pd.to_datetime(df.index).max()
        if latest.tzinfo is not None:
            latest = latest.tz_localize(None)
        latest_date = latest.date()
        today       = datetime.now(timezone.utc).date()
        gap_days    = (today - latest_date).days

        if gap_days <= 0:
            return False   # already have today's data

        import calendar
        weekday_latest = calendar.weekday(
            latest_date.year, latest_date.month, latest_date.day
        )  # 0=Mon … 6=Sun

        if weekday_latest == 4:   # latest date was a Friday
            # Allow up to 4 calendar days to cover long weekends (Mon holiday)
            return gap_days > 4
        return gap_days > 1       # weekday data: stale if older than yesterday

    except Exception:
        return True   # unreadable parquet → force re-fetch


def fetch_ticker(ticker: str) -> pd.DataFrame | None:
    """Download full history for one ticker and return a clean DataFrame."""
    yf_symbol = TICKER_MAP.get(ticker, ticker)
    try:
        raw = yf.download(
            yf_symbol,
            period=f"{HISTORY_YEARS}y",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            log.warning("yfinance returned no data for %s (%s).", ticker, yf_symbol)
            return None

        # Flatten MultiIndex columns (yfinance sometimes returns them)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
        raw.index   = pd.to_datetime(raw.index)
        if raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)

        # Keep only the columns we need
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
        raw  = raw[keep].dropna(subset=["close"])

        log.info("Fetched %d rows for %s.", len(raw), ticker)
        return raw

    except Exception as exc:
        log.error("Failed to fetch %s: %s", ticker, exc)
        return None


def load_cache(ticker: str) -> pd.DataFrame | None:
    """Load cached Parquet data for one ticker, or return None if missing."""
    p = cache_path(ticker)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as exc:
        log.warning("Could not read cache for %s: %s", ticker, exc)
        return None


def update_cache(ticker: str, force: bool = False) -> pd.DataFrame | None:
    """
    Fetch fresh data for ticker if the cache is stale (or force=True),
    otherwise load from disk.  Returns the up-to-date DataFrame or None.
    """
    if not force and not is_stale(ticker):
        return load_cache(ticker)

    df = fetch_ticker(ticker)
    if df is not None:
        df.to_parquet(cache_path(ticker))
        log.info("Cache updated → %s", cache_path(ticker))
    return df


def refresh_all(force: bool = False) -> dict[str, pd.DataFrame]:
    """
    Refresh cache for every ticker (scored + benchmarks).
    Returns a dict mapping ticker → DataFrame for all successfully loaded tickers.
    """
    results: dict[str, pd.DataFrame] = {}
    failed:  list[str] = []

    for i, ticker in enumerate(ALL_CACHE_TICKERS):
        df = update_cache(ticker, force=force)
        if df is not None:
            results[ticker] = df
        else:
            failed.append(ticker)
        # Gentle rate-limit buffer every 10 tickers
        if i > 0 and i % 10 == 0:
            time.sleep(1)

    if failed:
        log.warning("Failed to load data for: %s", ", ".join(failed))
    log.info(
        "Cache refresh complete. %d/%d tickers loaded.",
        len(results), len(ALL_CACHE_TICKERS),
    )
    return results


def load_all_cached() -> dict[str, pd.DataFrame]:
    """Load all cached Parquet files from disk without network calls."""
    results = {}
    for ticker in ALL_CACHE_TICKERS:
        df = load_cache(ticker)
        if df is not None:
            results[ticker] = df
        else:
            log.warning("No cache found for %s — run refresh_all() first.", ticker)
    return results


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Refresh OHLCV cache")
    parser.add_argument("--force", action="store_true", help="Force re-fetch all tickers")
    args = parser.parse_args()
    refresh_all(force=args.force)
