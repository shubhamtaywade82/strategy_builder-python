"""
Leakage-Safe Multi-Timeframe Feature Engineering
INVARIANT: HTF features joined via merge_asof(backward) on close_time.
"""
import numpy as np
import pandas as pd
from typing import Dict


def compute_ltf_features(df: pd.DataFrame) -> pd.DataFrame:
    """1m features - strictly backward-looking."""
    out = pd.DataFrame(index=df.index)
    c = df["close"]
    v = df["volume"]

    out["ltf_ret"] = c.pct_change()
    out["ltf_ret_5m"] = c.pct_change(5)
    out["ltf_ret_15m"] = c.pct_change(15)
    out["ltf_ret_60m"] = c.pct_change(60)

    out["ltf_rvol_20"] = out["ltf_ret"].rolling(20).std()
    out["ltf_rvol_10"] = out["ltf_ret"].rolling(10).std()
    out["ltf_rvol_5"] = out["ltf_ret"].rolling(5).std()

    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    out["ltf_ema9_ratio"] = c / ema9 - 1
    out["ltf_ema21_ratio"] = c / ema21 - 1
    out["ltf_ema50_ratio"] = c / ema50 - 1
    out["ltf_trend"] = np.sign(ema9 - ema21)

    out["ltf_bar_pos"] = (c - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    out["ltf_body_pct"] = (c - df["open"]).abs() / c
    out["ltf_body_dir"] = np.sign(c - df["open"])

    vol_sma20 = v.rolling(20).mean()
    out["ltf_vol_ratio"] = v / vol_sma20.replace(0, np.nan)
    out["ltf_taker_imb"] = df["taker_buy_base"] / v.replace(0, np.nan)
    out["ltf_trade_intensity"] = df["n_trades"] / v.replace(0, np.nan)

    out["ltf_mom_10"] = c / c.shift(10) - 1
    out["ltf_mom_20"] = c / c.shift(20) - 1

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - c.shift()).abs()
    tr3 = (df["low"] - c.shift()).abs()
    atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    out["ltf_atr_pct"] = atr / c

    out["ltf_range_pct"] = (df["high"] - df["low"]) / c
    out["ltf_hh"] = (c >= c.rolling(20).max().shift(1)).astype(int)
    out["ltf_ll"] = (c <= c.rolling(20).min().shift(1)).astype(int)

    return out


def compute_htf_block(htf: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """HTF features - only data available at close_time."""
    h = htf.copy()
    c = h["close"]
    ret = c.pct_change()

    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()

    out = pd.DataFrame({
        "ref_time": h["close_time"],
        f"{prefix}_ret": ret,
        f"{prefix}_ret_5": c.pct_change(5),
        f"{prefix}_ema9_ratio": c / ema9 - 1,
        f"{prefix}_ema21_ratio": c / ema21 - 1,
        f"{prefix}_trend": np.sign(ema9 - ema21),
        f"{prefix}_rvol": ret.rolling(20).std(),
        f"{prefix}_range_pct": (h["high"] - h["low"]) / c,
        f"{prefix}_taker_imb": h["taker_buy_base"] / h["volume"].replace(0, np.nan),
        f"{prefix}_vol_ratio": h["volume"] / h["volume"].rolling(20).mean().replace(0, np.nan),
        f"{prefix}_mom": c / c.shift(5) - 1,
        f"{prefix}_bar_pos": (c - h["low"]) / (h["high"] - h["low"]).replace(0, np.nan),
        f"{prefix}_body_pct": (c - h["open"]).abs() / c,
    })
    return out.dropna().reset_index(drop=True)


def build_mtf_features(frames: Dict[str, pd.DataFrame], base_tf: str = "1m") -> pd.DataFrame:
    """Build MTF feature matrix with NO lookahead."""
    base = frames[base_tf].copy()
    base["key"] = base["open_time"]

    merged = base[["open_time", "close_time", "open", "high", "low", "close", "volume", "key"]].copy()

    ltf = compute_ltf_features(base)
    for c in ltf.columns:
        merged[c] = ltf[c].values

    for tf, df in frames.items():
        if tf == base_tf:
            continue
        block = compute_htf_block(df, tf)
        merged = pd.merge_asof(
            merged.sort_values("key"),
            block.sort_values("ref_time"),
            left_on="key", right_on="ref_time",
            direction="backward", allow_exact_matches=True,
        ).drop(columns=["ref_time"], errors="ignore")

    result = merged.drop(columns=["key"]).reset_index(drop=True)
    return result
