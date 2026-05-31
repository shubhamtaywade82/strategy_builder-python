"""
Leakage-Safe Multi-Timeframe Feature Engineering
=================================================
Builds features from LTF (1m) through HTF (1d) with NO lookahead bias.

Invariant: Each base bar at open_time t is joined to the most recent FULLY-CLOSED
higher-TF bar whose close_time <= t. Uses merge_asof(direction="backward").

Based on: mtf_research.py (verified against Binance kline contract)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategy_research.utils.config import FeatureConfig, TimeframeConfig

log = logging.getLogger("mtf_features")


def _compute_ltf_features(base: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Compute LTF (1m) features using only data up to and including current bar.

    All features are strictly backward-looking - no future information.
    """
    df = base[["open_time", "close_time", "open", "high", "low", "close",
               "volume", "n_trades", "taker_buy_base", "quote_volume"]].copy()

    # Returns
    df["ltf_ret"] = df["close"].pct_change()
    df["ltf_ret_5m"] = df["close"].pct_change(5)
    df["ltf_ret_15m"] = df["close"].pct_change(15)

    # Realized volatility
    df["ltf_rvol_20"] = df["ltf_ret"].rolling(cfg.vol_lookback).std()
    df["ltf_rvol_10"] = df["ltf_ret"].rolling(10).std()

    # EMA and trend
    df["ltf_ema_fast"] = df["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    df["ltf_ema_slow"] = df["close"].ewm(span=cfg.ema_slow, adjust=False).mean()
    df["ltf_trend"] = np.sign(df["ltf_ema_fast"] - df["ltf_ema_slow"])
    df["ltf_ema_ratio"] = df["ltf_ema_fast"] / df["ltf_ema_slow"] - 1

    # Price position within bar range
    df["ltf_bar_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)

    # Volume features
    df["ltf_vol_sma"] = df["volume"].rolling(cfg.vol_lookback).mean()
    df["ltf_vol_ratio"] = df["volume"] / df["ltf_vol_sma"].replace(0, np.nan)
    df["ltf_taker_imb"] = df["taker_buy_base"] / df["volume"].replace(0, np.nan)

    # Trade intensity
    df["ltf_trade_intensity"] = df["n_trades"] / df["volume"].replace(0, np.nan)

    # Momentum
    df["ltf_momentum"] = df["close"] / df["close"].shift(cfg.momentum_lookback) - 1

    # ATR (simplified)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    df["ltf_atr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(cfg.atr_period).mean()
    df["ltf_atr_pct"] = df["ltf_atr"] / df["close"]

    # Bar size features
    df["ltf_range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["ltf_body_pct"] = (df["close"] - df["open"]).abs() / df["close"]
    df["ltf_body_dir"] = np.sign(df["close"] - df["open"])

    # Candlestick patterns (1-bar)
    df["ltf_is_doji"] = df["ltf_body_pct"] < 0.001
    upper_wick = (df["high"] - df[["close", "open"]].max(axis=1)) / df["close"]
    lower_wick = (df[["close", "open"]].min(axis=1) - df["low"]) / df["close"]
    df["ltf_upper_wick_pct"] = upper_wick
    df["ltf_lower_wick_pct"] = lower_wick

    return df


def _compute_htf_feature_block(
    htf: pd.DataFrame,
    prefix: str,
    cfg: FeatureConfig,
) -> pd.DataFrame:
    """Compute HTF features using ONLY data available at each HTF bar's close_time.

    The ref_time column marks when this feature row becomes known to the LTF bar.
    """
    h = htf.copy()
    ret = h["close"].pct_change()

    out = pd.DataFrame({
        "ref_time": h["close_time"],
        f"{prefix}_ret": ret,
        f"{prefix}_ret_5": h["close"].pct_change(5),
        f"{prefix}_ema_fast": h["close"].ewm(span=cfg.ema_fast, adjust=False).mean(),
        f"{prefix}_ema_slow": h["close"].ewm(span=cfg.ema_slow, adjust=False).mean(),
        f"{prefix}_rvol": ret.rolling(cfg.vol_lookback).std(),
        f"{prefix}_range_pct": (h["high"] - h["low"]) / h["close"],
        f"{prefix}_taker_imb": h["taker_buy_base"] / h["volume"].replace(0, np.nan),
        f"{prefix}_vol_sma": h["volume"].rolling(cfg.vol_lookback).mean(),
        f"{prefix}_momentum": h["close"] / h["close"].shift(cfg.momentum_lookback) - 1,
    })

    out[f"{prefix}_trend"] = np.sign(
        out[f"{prefix}_ema_fast"] - out[f"{prefix}_ema_slow"]
    )
    out[f"{prefix}_ema_ratio"] = (
        out[f"{prefix}_ema_fast"] / out[f"{prefix}_ema_slow"] - 1
    )

    # Bar position and body
    out[f"{prefix}_bar_position"] = (
        (h["close"] - h["low"]) / (h["high"] - h["low"]).replace(0, np.nan)
    )
    out[f"{prefix}_body_pct"] = (
        (h["close"] - h["open"]).abs() / h["close"]
    )

    # Number of trades intensity
    out[f"{prefix}_trade_intensity"] = (
        h["n_trades"] / h["volume"].replace(0, np.nan)
    )

    return out.dropna().reset_index(drop=True)


def build_mtf_features(
    frames: Dict[str, pd.DataFrame],
    cfg: Optional[FeatureConfig] = None,
    base_tf: str = "1m",
) -> pd.DataFrame:
    """Build leakage-safe multi-timeframe features aligned to base_tf grid.

    INVARIANT: Each base bar at open_time t joins only to HTF bars whose
    close_time <= t (fully closed). This prevents lookahead bias.

    Args:
        frames: Dict of timeframe -> OHLCV DataFrame
        cfg: Feature configuration
        base_tf: Base timeframe (default "1m")

    Returns:
        DataFrame with MTF features aligned to base_tf index
    """
    cfg = cfg or FeatureConfig()

    if base_tf not in frames:
        raise ValueError(f"Base timeframe '{base_tf}' not in frames. Got: {list(frames)}")

    base = frames[base_tf].copy()
    base["key"] = base["open_time"]

    # Start with base columns + key
    merged = base[[
        "open_time", "close_time", "open", "high", "low", "close",
        "volume", "key"
    ]].copy()

    # LTF features
    ltf_feats = _compute_ltf_features(base, cfg)
    ltf_cols = [c for c in ltf_feats.columns if c not in merged.columns or c == "key"]
    for c in ltf_cols:
        if c not in merged.columns:
            merged[c] = ltf_feats[c].values

    log.info(f"Built LTF features: {len([c for c in merged.columns if c.startswith('ltf_')])} columns")

    # Merge each HTF
    htf_count = 0
    for tf, df in frames.items():
        if tf == base_tf:
            continue

        block = _compute_htf_feature_block(df, prefix=tf, cfg=cfg)

        merged = pd.merge_asof(
            merged.sort_values("key"),
            block.sort_values("ref_time"),
            left_on="key",
            right_on="ref_time",
            direction="backward",
            allow_exact_matches=True,
        ).drop(columns=["ref_time"], errors="ignore")

        htf_count += 1
        htf_cols = [c for c in merged.columns if c.startswith(f"{tf}_")]
        log.info(f"  Merged {tf}: {len(htf_cols)} HTF columns")

    result = merged.drop(columns=["key"], errors="ignore").reset_index(drop=True)

    # Report NaN count
    nan_pct = result.isna().mean().mean() * 100
    log.info(f"MTF feature matrix: {result.shape}, NaN: {nan_pct:.1f}%")

    return result


def add_regime_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add regime-detection features derived from MTF data.

    These help the model adapt to different market conditions.
    """
    df = features.copy()

    # Trend strength: alignment of trends across timeframes
    trend_cols = [c for c in df.columns if c.endswith("_trend")]
    if len(trend_cols) >= 2:
        df["regime_trend_alignment"] = df[trend_cols].sum(axis=1) / len(trend_cols)

    # Volatility regime: ratio of short-term to longer-term vol
    if "ltf_rvol_10" in df.columns and "ltf_rvol_20" in df.columns:
        df["regime_vol_expanding"] = df["ltf_rvol_10"] / df["ltf_rvol_20"].replace(0, np.nan)

    # Volume regime
    if "ltf_vol_ratio" in df.columns:
        df["regime_high_volume"] = (df["ltf_vol_ratio"] > 1.5).astype(int)

    # Momentum alignment
    mom_cols = [c for c in df.columns if c.endswith("_momentum")]
    if len(mom_cols) >= 2:
        df["regime_mom_alignment"] = np.sign(df[mom_cols]).sum(axis=1) / len(mom_cols)

    return df
