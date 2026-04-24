"""
cache_manager.py
Fetches OHLCV data from yfinance and stores one Parquet file per ticker.
Skips re-fetch if the cache is recent enough (CACHE_STALENESS_HOURS).
"""

import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import (
    TICKERS, TICKER_MAP, CACHE_DIR,
    HISTORY_YEARS, CACHE_STALENESS_HOURS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.parquet"


def is_stale(ticker: str) -> bool:
    """Return True if the cache file is missing or older than CACHE_STALENESS_HOURS."""
    p = cache_path(ticker)
    if not p.exists():
        return True
    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    age   = datetime.now(tz=timezone.utc) - mtime
    return age > timedelta(hours=CACHE_STALENESS_HOURS)


def fetch_ticker(ticker: str) -> pd.DataFrame | None:
    """Download full history for one ticker and return a clean DataFrame."""
    fetch_symbol = TICKER_MAP.get(ticker, ticker)
    end   = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=HISTORY_YEARS * 365 + 30)  # small buffer

    try:
        raw = yf.download(
            fetch_symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        log.warning("yfinance download failed for %s: %s", ticker, exc)
        return None

    if raw.empty:
        log.warning("No data returned for %s", ticker)
        return None

    # Flatten MultiIndex columns if yfinance returns them
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(columns=str.lower)
    raw.index.name = "date"
    raw = raw[["open", "high", "low", "close", "volume"]].copy()
    raw = raw.dropna(subset=["close"])
    raw["ticker"] = ticker
    return raw


def load_cache(ticker: str) -> pd.DataFrame | None:
    """Load existing Parquet cache for a ticker."""
    p = cache_path(ticker)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def update_cache(ticker: str, force: bool = False) -> pd.DataFrame | None:
    """
    Update the cache for one ticker.
    - If stale (or force=True): fetch full 2-year history and overwrite.
    - If fresh: load from disk and return.
    Returns the DataFrame or None on failure.
    """
    if not force and not is_stale(ticker):
        return load_cache(ticker)

    log.info("Fetching %s ...", ticker)
    df = fetch_ticker(ticker)
    if df is None or df.empty:
        # Fall back to existing cache if available
        existing = load_cache(ticker)
        if existing is not None:
            log.warning("Using stale cache for %s", ticker)
        return existing

    df.to_parquet(cache_path(ticker))
    log.info("  Cached %s: %d rows (%s → %s)",
             ticker, len(df),
             df.index.min().date(), df.index.max().date())
    return df


def refresh_all(force: bool = False) -> dict[str, pd.DataFrame]:
    """
    Refresh the cache for every ticker in the universe.
    Returns a dict of {ticker: DataFrame}.
    Adds a small sleep between requests to be polite to yfinance rate limits.
    """
    results: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for i, ticker in enumerate(TICKERS):
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
    log.info("Cache refresh complete. %d/%d tickers loaded.", len(results), len(TICKERS))
    return results


def load_all_cached() -> dict[str, pd.DataFrame]:
    """Load all cached Parquet files from disk without network calls."""
    results = {}
    for ticker in TICKERS:
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
