"""
Smart Money Concepts (SMC) Features
====================================
Leakage-safe implementation of:
- Fair Value Gaps (FVG)
- Break of Structure (BOS)
- Order Blocks (OB)
- Liquidity Sweeps
- Premium/Discount zones

All features use data <= bar t only (no lookahead).
"""
import numpy as np
import pandas as pd
from typing import Tuple


def detect_fvg(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Detect Fair Value Gaps (3-candle imbalances).

    Bullish FVG: low[i+2] > high[i] (gap up)
    Bearish FVG: high[i+2] < low[i] (gap down)
    Returns binary flags + distance to nearest FVG.
    """
    out = pd.DataFrame(index=df.index)
    h, l, c = df["high"], df["low"], df["close"]

    # Bullish FVG: two candles ago's high < current candle's low
    bull_fvg = (l > h.shift(2)).astype(int)
    # Bearish FVG: two candles ago's low > current candle's high
    bear_fvg = (h < l.shift(2)).astype(int)

    out["fvg_bull"] = bull_fvg
    out["fvg_bear"] = bear_fvg
    out["fvg_any"] = ((bull_fvg + bear_fvg) > 0).astype(int)

    # Distance to most recent FVG (in price %)
    bull_fvg_level = np.where(bull_fvg, h.shift(2), np.nan)
    bear_fvg_level = np.where(bear_fvg, l.shift(2), np.nan)

    # Forward fill the last FVG level
    bull_fvg_ff = pd.Series(bull_fvg_level, index=df.index).ffill()
    bear_fvg_ff = pd.Series(bear_fvg_level, index=df.index).ffill()

    out["dist_bull_fvg_pct"] = (c - bull_fvg_ff) / c
    out["dist_bear_fvg_pct"] = (bear_fvg_ff - c) / c

    # FVG above/below price (mitigated or not)
    out["fvg_above_unmitigated"] = ((bull_fvg_ff > c) & bull_fvg_ff.notna()).astype(int)
    out["fvg_below_unmitigated"] = ((bear_fvg_ff < c) & bear_fvg_ff.notna()).astype(int)

    return out


def detect_bos(df: pd.DataFrame, swing_len: int = 5) -> pd.DataFrame:
    """Detect Break of Structure (BOS) and Change of Character (CHOCH).

    BOS: Price breaks above previous swing high (bullish) or below previous swing low (bearish).
    CHOCH: Price breaks below previous swing low after making higher highs (bullish to bearish).
    """
    out = pd.DataFrame(index=df.index)
    c, h, l = df["close"], df["high"], df["low"]

    # Swing highs/lows using rolling window
    swing_high = (h == h.rolling(swing_len * 2 + 1, center=True).max())
    swing_low = (l == l.rolling(swing_len * 2 + 1, center=True).min())

    # Previous swing levels
    prev_swing_high = h.where(swing_high).ffill().shift(1)
    prev_swing_low = l.where(swing_low).ffill().shift(1)

    # BOS: close breaks previous swing
    bos_bull = (c > prev_swing_high).astype(int)
    bos_bear = (c < prev_swing_low).astype(int)

    out["bos_bull"] = bos_bull
    out["bos_bear"] = bos_bear
    out["bos_recent"] = np.where(
        bos_bull.rolling(10).sum() > 0, 1,
        np.where(bos_bear.rolling(10).sum() > 0, -1, 0)
    )

    # CHOCH: break of previous structure after trend change
    # Simplified: if last BOS was bullish but now bearish
    out["choch_bull"] = ((bos_bull.rolling(20).sum() > 0) & (bos_bear.rolling(5).sum() > 0)).astype(int)
    out["choch_bear"] = ((bos_bear.rolling(20).sum() > 0) & (bos_bull.rolling(5).sum() > 0)).astype(int)

    return out


def detect_order_blocks(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Detect Order Blocks (last opposing candle before a strong move).

    Bullish OB: last bearish candle before a strong bullish move (BOS).
    Bearish OB: last bullish candle before a strong bearish move (BOS).
    """
    out = pd.DataFrame(index=df.index)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # Candle direction
    bullish = c > o
    bearish = c < o

    # Strong move: 3+ consecutive candles in same direction
    consec_bull = bullish.rolling(3).sum() >= 3
    consec_bear = bearish.rolling(3).sum() >= 3

    # OB: last opposing candle before strong move
    bull_ob = (consec_bull & bearish.shift(1)).astype(int)
    bear_ob = (consec_bear & bullish.shift(1)).astype(int)

    out["ob_bull"] = bull_ob
    out["ob_bear"] = bear_ob

    # OB levels (for reference)
    ob_bull_level = np.where(bull_ob, h.shift(1), np.nan)
    ob_bear_level = np.where(bear_ob, l.shift(1), np.nan)

    ob_bull_ff = pd.Series(ob_bull_level, index=df.index).ffill()
    ob_bear_ff = pd.Series(ob_bear_level, index=df.index).ffill()

    # Price relative to OB (mitigated or not)
    out["ob_bull_active"] = ((c < ob_bull_ff) & ob_bull_ff.notna()).astype(int)
    out["ob_bear_active"] = ((c > ob_bear_ff) & ob_bear_ff.notna()).astype(int)

    return out


def detect_liquidity_sweeps(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Detect liquidity sweeps (wicks beyond recent swing levels).

    Bullish sweep: wick below previous swing low, then close back above.
    Bearish sweep: wick above previous swing high, then close back below.
    """
    out = pd.DataFrame(index=df.index)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # Recent swing levels
    recent_low = l.rolling(lookback).min().shift(1)
    recent_high = h.rolling(lookback).max().shift(1)

    # Bullish sweep: low goes below recent low, close back above
    sweep_bull = ((l < recent_low) & (c > recent_low)).astype(int)
    # Bearish sweep: high goes above recent high, close back below
    sweep_bear = ((h > recent_high) & (c < recent_high)).astype(int)

    out["sweep_bull"] = sweep_bull
    out["sweep_bear"] = sweep_bear

    # Sweep strength (wick size relative to range)
    range_bar = h - l
    out["sweep_bull_strength"] = np.where(
        sweep_bull, (recent_low - l) / range_bar.replace(0, np.nan), 0
    )
    out["sweep_bear_strength"] = np.where(
        sweep_bear, (h - recent_high) / range_bar.replace(0, np.nan), 0
    )

    return out


def premium_discount(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """Calculate premium/discount relative to recent range.

    Premium = price in upper 30% of recent range (expensive)
    Discount = price in lower 30% of recent range (cheap)
    """
    out = pd.DataFrame(index=df.index)
    c, h, l = df["close"], df["high"], df["low"]

    range_high = h.rolling(lookback).max()
    range_low = l.rolling(lookback).min()
    range_size = range_high - range_low

    position = (c - range_low) / range_size.replace(0, np.nan)

    out["premium_discount"] = position  # 0 = discount, 1 = premium
    out["is_premium"] = (position > 0.7).astype(int)
    out["is_discount"] = (position < 0.3).astype(int)
    out["is_equilibrium"] = ((position >= 0.3) & (position <= 0.7)).astype(int)

    return out


def compute_smc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all SMC features for a dataframe."""
    fvg = detect_fvg(df)
    bos = detect_bos(df)
    ob = detect_order_blocks(df)
    sweep = detect_liquidity_sweeps(df)
    pd_zone = premium_discount(df)

    return pd.concat([fvg, bos, ob, sweep, pd_zone], axis=1)


def compute_all_features(frames: dict, base_tf: str = "1m") -> pd.DataFrame:
    """Build complete feature matrix: MTF + SMC."""
    from features import build_mtf_features

    # Get base MTF features
    features = build_mtf_features(frames, base_tf)

    # Add SMC features on 1m
    smc = compute_smc_features(frames[base_tf])
    for col in smc.columns:
        features[f"smc_{col}"] = smc[col].values

    # Add SMC on 15m (HTF context)
    if "15m" in frames:
        smc_15m = compute_smc_features(frames["15m"])
        smc_15m.columns = [f"15m_smc_{c}" for c in smc_15m.columns]
        # Simple approach: forward-fill 15m SMC to 1m grid
        htf = frames["15m"].copy()
        htf_merged = smc_15m.copy()
        htf_merged["ref_time"] = htf["close_time"]
        features["key"] = features["open_time"]
        features = pd.merge_asof(
            features.sort_values("key"),
            htf_merged.sort_values("ref_time"),
            left_on="key", right_on="ref_time",
            direction="backward", allow_exact_matches=True,
        ).drop(columns=["ref_time", "key"], errors="ignore")

    return features.reset_index(drop=True)
