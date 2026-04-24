"""
features.py
Builds the ML feature matrix from raw OHLCV caches.

Features per stock:
  - Technical indicators (RSI, MACD, Bollinger, ATR, EMA, SMA, OBV, etc.)
  - Lag returns (prior N days)
  - Volatility and volume-relative metrics
  - Relative-to-benchmark (vs QQQ and SPY)
  - Binary target labels for T+1, T+5, T+20

Call build_feature_matrix() to get a single pooled DataFrame across all tickers.
"""

import logging

import numpy as np
import pandas as pd
import ta

from config import (
    TICKERS, BENCHMARK_TICKERS, HORIZONS,
    LAG_DAYS, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, ATR_PERIOD, EMA_PERIODS, SMA_PERIODS, VOL_AVG_PERIOD,
)

log = logging.getLogger(__name__)


# ── Per-ticker feature builder ─────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add ta-lib indicators to a single-ticker OHLCV DataFrame."""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(c, window=RSI_PERIOD).rsi()

    # MACD
    macd_obj = ta.trend.MACD(c, window_fast=MACD_FAST,
                               window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_diff"]   = macd_obj.macd_diff()

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(h, l, c)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Williams %R
    df["williams_r"] = ta.momentum.WilliamsRIndicator(h, l, c).williams_r()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(c, window=BB_PERIOD)
    df["bb_width"]  = bb.bollinger_wband()
    df["bb_pct"]    = bb.bollinger_pband()
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_lower"]  = bb.bollinger_lband()

    # ATR (Average True Range)
    df["atr"] = ta.volatility.AverageTrueRange(h, l, c, window=ATR_PERIOD).average_true_range()
    df["atr_pct"] = df["atr"] / c  # normalise by price

    # EMAs
    for p in EMA_PERIODS:
        df[f"ema_{p}"] = ta.trend.EMAIndicator(c, window=p).ema_indicator()
        df[f"ema_{p}_dist"] = (c - df[f"ema_{p}"]) / df[f"ema_{p}"]  # % distance

    # SMAs
    for p in SMA_PERIODS:
        df[f"sma_{p}"] = ta.trend.SMAIndicator(c, window=p).sma_indicator()
        df[f"sma_{p}_dist"] = (c - df[f"sma_{p}"]) / df[f"sma_{p}"]

    # ADX
    adx = ta.trend.ADXIndicator(h, l, c)
    df["adx"]    = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # OBV (On-Balance Volume)
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["obv_change"] = df["obv"].pct_change()

    # Volume vs N-day average
    df["vol_ratio"] = v / v.rolling(VOL_AVG_PERIOD).mean()

    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Daily returns, gap, range, and lag features."""
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]

    df["daily_return"]  = c.pct_change()
    df["gap_pct"]       = (o - c.shift(1)) / c.shift(1)        # overnight gap
    df["daily_range"]   = (h - l) / l                           # intraday range
    df["close_vs_open"] = (c - o) / o                           # close vs open

    # Rolling returns
    df["ret_5d"]  = c.pct_change(5)
    df["ret_20d"] = c.pct_change(20)

    # Lag returns
    for lag in range(1, LAG_DAYS + 1):
        df[f"ret_lag_{lag}"] = df["daily_return"].shift(lag)

    return df


def add_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Binary classification targets: 1 if future close > today's close."""
    for label, n in HORIZONS.items():
        future_close = df["close"].shift(-n)
        df[f"target_{label}"] = (future_close > df["close"]).astype(float)
    return df


def build_ticker_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline for a single-ticker DataFrame."""
    df = df.sort_index()
    df = add_technical_indicators(df)
    df = add_price_features(df)
    df = add_target_labels(df)
    return df


# ── Benchmark (market-relative) features ──────────────────────────────────────

def build_benchmark_features(all_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute market-relative features from QQQ and SPY.
    Returns a DataFrame indexed by date with columns like:
      qqq_ret_1d, qqq_rsi, spy_ret_1d, spy_rsi, ...
    """
    frames = []
    for bm in BENCHMARK_TICKERS:
        if bm not in all_data:
            log.warning("Benchmark %s not found in cache — skipping.", bm)
            continue
        bdf = all_data[bm].copy()
        prefix = bm.lower()
        bdf[f"{prefix}_ret_1d"]  = bdf["close"].pct_change()
        bdf[f"{prefix}_ret_5d"]  = bdf["close"].pct_change(5)
        bdf[f"{prefix}_ret_20d"] = bdf["close"].pct_change(20)
        bdf[f"{prefix}_rsi"]     = ta.momentum.RSIIndicator(
                                        bdf["close"], window=RSI_PERIOD).rsi()
        bdf[f"{prefix}_vol_ratio"] = (bdf["volume"]
                                      / bdf["volume"].rolling(VOL_AVG_PERIOD).mean())
        cols = [c for c in bdf.columns if c.startswith(prefix)]
        frames.append(bdf[cols])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


# ── Pooled feature matrix ──────────────────────────────────────────────────────

def build_feature_matrix(all_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a single pooled feature DataFrame across all tickers.

    Steps:
      1. Build per-ticker technical + price features.
      2. Build benchmark (QQQ/SPY) features.
      3. Join benchmark features onto each ticker's rows.
      4. Compute relative-to-benchmark return columns.
      5. Pool all tickers into one DataFrame with a 'ticker' column.
    """
    benchmark_df = build_benchmark_features(all_data)

    ticker_frames = []
    for ticker, raw_df in all_data.items():
        if raw_df is None or raw_df.empty:
            continue
        try:
            feat = build_ticker_features(raw_df.copy())
        except Exception as exc:
            log.warning("Feature build failed for %s: %s", ticker, exc)
            continue

        # Join benchmark columns
        if not benchmark_df.empty:
            feat = feat.join(benchmark_df, how="left")
            # Relative return = stock return minus benchmark return
            for bm in BENCHMARK_TICKERS:
                prefix = bm.lower()
                col = f"{prefix}_ret_1d"
                if col in feat.columns:
                    feat[f"rel_{prefix}_1d"]  = feat["daily_return"] - feat[col]
                    feat[f"rel_{prefix}_5d"]  = feat["ret_5d"]  - feat[f"{prefix}_ret_5d"]
                    feat[f"rel_{prefix}_20d"] = feat["ret_20d"] - feat[f"{prefix}_ret_20d"]

        feat["ticker"] = ticker
        ticker_frames.append(feat)

    if not ticker_frames:
        raise RuntimeError("No feature data was built — check cache.")

    pooled = pd.concat(ticker_frames, axis=0)
    pooled = pooled.sort_index()
    log.info("Feature matrix: %d rows × %d cols", *pooled.shape)
    return pooled


# ── Feature / target column helpers ───────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all feature column names (exclude targets, OHLCV, ticker)."""
    exclude = {"open", "high", "low", "close", "volume", "ticker"}
    exclude |= {f"target_{h}" for h in HORIZONS}
    return [c for c in df.columns if c not in exclude]


def get_target_col(horizon: str) -> str:
    return f"target_{horizon}"
