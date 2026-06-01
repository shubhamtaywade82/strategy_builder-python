"""
Market Regime Detection
========================
Classifies market state: trending, ranging, volatile, quiet.
Used for regime-adjusted position sizing and strategy selection.
"""
import numpy as np
import pandas as pd
from typing import Literal

RegimeType = Literal["trending", "ranging", "volatile", "quiet"]


def detect_regime(df: pd.DataFrame, lookback: int = 100) -> RegimeType:
    """Detect current market regime from 1m OHLCV.

    Uses: ADX-like trend strength, ATR percentile, Bollinger width.
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # Trend strength: EMA alignment
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()

    trend_aligned = (
        (ema20 > ema50).iloc[-lookback:].mean() > 0.6
        and (ema50 > ema200).iloc[-lookback:].mean() > 0.6
    )
    trend_opposed = (
        (ema20 < ema50).iloc[-lookback:].mean() > 0.6
        and (ema50 < ema200).iloc[-lookback:].mean() > 0.6
    )
    trend_strength = max(
        (ema20 > ema50).iloc[-lookback:].mean(),
        (ema20 < ema50).iloc[-lookback:].mean(),
    )

    # Volatility
    ret = c.pct_change()
    atr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1).rolling(14).mean()
    atr_pct = (atr / c).iloc[-lookback:]
    atr_percentile = (atr_pct > atr_pct.rolling(lookback * 2).quantile(0.5).iloc[-1]).mean()

    # Bollinger width
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bb_width = ((std20 * 2) / sma20).iloc[-lookback:]
    is_wide = bb_width.mean() > bb_width.rolling(lookback * 2).mean().iloc[-1]

    # Range-bound detection: price oscillating around EMAs
    around_ema = ((c - ema20).abs() / c < 0.005).iloc[-lookback:].mean()

    # Classify
    if trend_strength > 0.65 and (trend_aligned or trend_opposed):
        if atr_percentile > 0.7:
            return "volatile"
        return "trending"
    elif around_ema > 0.4 or not is_wide:
        return "ranging"
    elif atr_percentile < 0.3:
        return "quiet"
    else:
        return "volatile"


def regime_features(df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    """Compute regime-indicator features."""
    out = pd.DataFrame(index=df.index)
    c = df["close"]

    # Trend features
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    out["ema20_above_50"] = (ema20 > ema50).astype(int)
    out["ema50_above_200"] = (ema50 > c.ewm(span=200, adjust=False).mean()).astype(int)

    # Volatility regime
    ret = c.pct_change()
    out["rvol_20"] = ret.rolling(20).std()
    out["rvol_percentile"] = out["rvol_20"].rolling(100).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5,
        raw=False,
    )

    # Bollinger squeeze
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    out["bb_width"] = (std20 * 2) / sma20
    out["bb_squeeze"] = (out["bb_width"] < out["bb_width"].rolling(50).quantile(0.2)).astype(int)

    # Price range position
    range_high = df["high"].rolling(lookback).max()
    range_low = df["low"].rolling(lookback).min()
    out["range_position"] = (c - range_low) / (range_high - range_low).replace(0, np.nan)

    return out
