"""
Five-Condition Rule Strategy — the honest, transparent counterpart
==================================================================
A fixed, interpretable rule (no ML confidence filter) backtested out-of-sample
across time folds with a bootstrap significance test and a random-entry baseline.

Conditions (LONG; SHORT mirrors with bearish direction):
  1. 4H EMA50 > EMA200          (higher-timeframe trend)
  2. 1H Break of Structure up   (close breaks prior SWING-bar high)
  3. 15m bullish FVG present    (fair-value gap within recent window)
  4. 1m ATR(14) > 60th pctile   (volatility filter, strictly-past window)
  5. 1m volume > 70th pctile    (participation filter, strictly-past window)

All five must be true on a 1m close to fire. Every column is leakage-safe: HTF
conditions are joined with merge_asof(backward) so a higher-timeframe bar only
influences 1m bars strictly after its close, and the percentile thresholds use a
trailing window shifted by one bar.

This single module is the source of truth for the rule. ``run_strategies.py``
(multi-symbol research) and ``trading_bot.py`` (live) both import from here so the
backtested rule and the executed rule can never drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .backtest import backtest

SEED = 42


@dataclass
class RuleConfig:
    swing: int = 5            # 1H BOS swing lookback
    fvg_lookback: int = 10    # 15m FVG recency window
    atr_win: int = 1440       # trailing window for ATR percentile (1 day of 1m)
    vol_win: int = 1440       # trailing window for volume percentile
    atr_q: float = 0.60       # condition 4 threshold
    vol_q: float = 0.70       # condition 5 threshold
    cooldown_bars: int = 30   # min spacing between entries (dedupe overlap)
    n_folds: int = 5
    n_boot: int = 2000
    min_trades: int = 30
    baseline_sample: int = 3000


# Human-readable condition table surfaced to the UI (labels only; the actual
# masks are computed in build_signal_mask).
CONDITION_LABELS = {
    "long": [
        {"id": "trend", "label": "4H EMA50 > EMA200 (bull trend)", "feature": "4h_ema50_200"},
        {"id": "bos", "label": "1H Break of Structure Up", "feature": "1h_bos"},
        {"id": "fvg", "label": "15m Bullish FVG (recent)", "feature": "15m_fvg_above"},
        {"id": "atr", "label": "ATR(14) > 60th percentile", "feature": "ltf_atr_pctile"},
        {"id": "vol", "label": "Volume > 70th percentile", "feature": "ltf_vol_pctile"},
    ],
    "short": [
        {"id": "trend", "label": "4H EMA50 < EMA200 (bear trend)", "feature": "4h_ema50_200"},
        {"id": "bos", "label": "1H Break of Structure Down", "feature": "1h_bos"},
        {"id": "fvg", "label": "15m Bearish FVG (recent)", "feature": "15m_fvg_below"},
        {"id": "atr", "label": "ATR(14) > 60th percentile", "feature": "ltf_atr_pctile"},
        {"id": "vol", "label": "Volume > 70th percentile", "feature": "ltf_vol_pctile"},
    ],
}


def _htf_cond(frame: pd.DataFrame, cond_long: pd.Series, cond_short: pd.Series) -> pd.DataFrame:
    """Wrap an HTF boolean condition keyed by close_time for a backward merge."""
    df = pd.DataFrame({"ref_time": frame["close_time"].reset_index(drop=True)})
    df["c_long"] = cond_long.reset_index(drop=True).fillna(False).astype(bool)
    df["c_short"] = cond_short.reset_index(drop=True).fillna(False).astype(bool)
    return df.dropna(subset=["ref_time"]).sort_values("ref_time")


def build_signal_mask(frames: Dict[str, pd.DataFrame], cfg: RuleConfig = RuleConfig()) -> pd.DataFrame:
    """Return the 1m frame with per-bar long/short signal flags. No lookahead."""
    d1 = frames["1m"].copy().sort_values("open_time").reset_index(drop=True)
    out = d1[["open_time", "close_time", "open", "high", "low", "close", "volume"]].copy()
    out["key"] = out["open_time"]

    # C1: 4H EMA50 vs EMA200 (trend)
    f4 = frames["4h"].sort_values("open_time")
    e50 = f4["close"].ewm(span=50, adjust=False).mean()
    e200 = f4["close"].ewm(span=200, adjust=False).mean()
    c1 = _htf_cond(f4, e50 > e200, e50 < e200).rename(
        columns={"c_long": "c1_long", "c_short": "c1_short"})

    # C2: 1H Break of Structure (close breaks prior `swing`-bar extreme)
    f1h = frames["1h"].sort_values("open_time")
    prev_high = f1h["high"].rolling(cfg.swing).max().shift(1)
    prev_low = f1h["low"].rolling(cfg.swing).min().shift(1)
    c2 = _htf_cond(f1h, f1h["close"] > prev_high, f1h["close"] < prev_low).rename(
        columns={"c_long": "c2_long", "c_short": "c2_short"})

    # C3: 15m FVG present within last `fvg_lookback` bars
    f15 = frames["15m"].sort_values("open_time")
    bull_fvg = (f15["low"] > f15["high"].shift(2))
    bear_fvg = (f15["high"] < f15["low"].shift(2))
    c3_long = bull_fvg.rolling(cfg.fvg_lookback, min_periods=1).max().fillna(0).astype(bool)
    c3_short = bear_fvg.rolling(cfg.fvg_lookback, min_periods=1).max().fillna(0).astype(bool)
    c3 = _htf_cond(f15, c3_long, c3_short).rename(
        columns={"c_long": "c3_long", "c_short": "c3_short"})

    for blk in (c1, c2, c3):
        out = pd.merge_asof(
            out.sort_values("key"), blk,
            left_on="key", right_on="ref_time",
            direction="backward", allow_exact_matches=True,
        ).drop(columns=["ref_time"], errors="ignore")

    # C4: 1m ATR(14) percentile (trailing window, strictly past)
    c = out["close"]
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - c.shift()).abs(),
        (out["low"] - c.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_thresh = atr.rolling(cfg.atr_win, min_periods=cfg.atr_win // 4).quantile(cfg.atr_q).shift(1)
    out["c4"] = (atr > atr_thresh)

    # C5: 1m volume percentile (trailing window, strictly past)
    v = out["volume"]
    vol_thresh = v.rolling(cfg.vol_win, min_periods=cfg.vol_win // 4).quantile(cfg.vol_q).shift(1)
    out["c5"] = (v > vol_thresh)

    for col in ["c1_long", "c1_short", "c2_long", "c2_short",
                "c3_long", "c3_short", "c4", "c5"]:
        out[col] = out[col].fillna(False).astype(bool)

    out["sig_long"] = out["c1_long"] & out["c2_long"] & out["c3_long"] & out["c4"] & out["c5"]
    out["sig_short"] = out["c1_short"] & out["c2_short"] & out["c3_short"] & out["c4"] & out["c5"]
    return out.reset_index(drop=True)


def _apply_cooldown(idxs: List[int], bars: int) -> List[int]:
    kept, last = [], -10**9
    for i in idxs:
        if i - last >= bars:
            kept.append(i)
            last = i
    return kept


def _walk_forward_folds(trades: list, n_bars: int, n_folds: int) -> dict:
    """Per-fold expectancy/WR by entry-bar time slice (out-of-time consistency)."""
    if not trades:
        return {"folds": [], "stability": 0.0}
    edges = np.linspace(0, n_bars, n_folds + 1).astype(int)
    folds = []
    for k in range(n_folds):
        lo, hi = edges[k], edges[k + 1]
        ft = [t for t in trades if lo <= t["entry_idx"] < hi]
        if len(ft) < 5:
            folds.append({"fold": k + 1, "trades": len(ft), "expectancy": None,
                          "win_rate": None, "profit_factor": None})
            continue
        pnl = np.array([t["net_pnl"] for t in ft])
        gp = float(pnl[pnl > 0].sum())
        gl = float(-pnl[pnl <= 0].sum())
        folds.append({
            "fold": k + 1, "trades": len(ft),
            "expectancy": float(pnl.mean()),
            "win_rate": float((pnl > 0).mean()),
            "profit_factor": (gp / gl) if gl > 0 else None,
        })
    scored = [f for f in folds if f["expectancy"] is not None]
    stability = (sum(1 for f in scored if f["expectancy"] > 0) / len(scored)) if scored else 0.0
    return {"folds": folds, "stability": stability}


def _bootstrap_pvalue(pnl: np.ndarray, n_boot: int) -> float:
    """One-sided: bootstrap probability the true mean per-trade pnl is <= 0."""
    if len(pnl) < 5:
        return 1.0
    rng = np.random.default_rng(SEED)
    means = rng.choice(pnl, size=(n_boot, len(pnl)), replace=True).mean(axis=1)
    return float((means <= 0).mean())


def _baseline_metrics(d1: pd.DataFrame, side_val: int, up_pct: float, dn_pct: float,
                      leverage: float, cost: float, horizon: int, cfg: RuleConfig) -> dict:
    """Null hypothesis: WR/expectancy of RANDOM entries at this RR (geometry+fees)."""
    n = len(d1)
    upper = n - horizon - 1
    if upper <= 1:
        return {"win_rate": 0.0, "expectancy": 0.0}
    rng = np.random.default_rng(SEED + 1)
    size = min(cfg.baseline_sample, upper)
    idx = rng.choice(np.arange(0, upper), size=size, replace=False)
    sigs = [{"bar_idx": int(i), "side": side_val, "confidence": 1.0} for i in idx]
    m = backtest(d1, sigs, up_pct, dn_pct, leverage, cost, horizon).metrics or {}
    return {"win_rate": m.get("win_rate", 0.0), "expectancy": m.get("expectancy", 0.0)}


def backtest_rule_side(
    mask: pd.DataFrame,
    d1: pd.DataFrame,
    side: str,
    up_pct: float,
    dn_pct: float,
    leverage: float = 10.0,
    cost: float = 0.0009,
    horizon: int = 120,
    cfg: RuleConfig = RuleConfig(),
) -> dict:
    """Backtest one side of the 5-condition rule. Returns real metrics or {error}."""
    col = "sig_long" if side == "long" else "sig_short"
    side_val = 1 if side == "long" else -1
    sig_idx = _apply_cooldown(mask.index[mask[col]].tolist(), cfg.cooldown_bars)
    signals = [{"bar_idx": int(i), "side": side_val, "confidence": 1.0} for i in sig_idx]

    conditions = CONDITION_LABELS[side]
    if len(signals) < cfg.min_trades:
        return {"side": side, "conditions": conditions, "n_signals": len(signals),
                "error": f"insufficient signals (n={len(signals)} < {cfg.min_trades})"}

    bt = backtest(d1, signals, up_pct, dn_pct, leverage, cost, horizon)
    m = bt.metrics
    if not m or m.get("trade_count", 0) < cfg.min_trades:
        return {"side": side, "conditions": conditions, "n_signals": len(signals),
                "error": "insufficient trades after backtest"}

    pnl = np.array([t["net_pnl"] for t in bt.trades])
    wf = _walk_forward_folds(bt.trades, len(d1), cfg.n_folds)
    p_value = _bootstrap_pvalue(pnl, cfg.n_boot)
    sharpe = float(pnl.mean() / pnl.std()) if pnl.std() > 0 else 0.0

    base = _baseline_metrics(d1, side_val, up_pct, dn_pct, leverage, cost, horizon, cfg)
    edge_wr = m["win_rate"] - base["win_rate"]
    edge_exp = m["expectancy"] - base["expectancy"]
    beats_baseline = edge_wr > 0 and edge_exp > 0

    is_robust = bool(m["expectancy"] > 0 and wf["stability"] >= 0.6
                     and p_value < 0.05 and m["trade_count"] >= cfg.min_trades
                     and beats_baseline)

    warnings: List[str] = []
    if m["expectancy"] <= 0:
        warnings.append("negative expectancy")
    if p_value >= 0.05:
        warnings.append(f"not significant (p={p_value:.3f})")
    if wf["stability"] < 0.6:
        warnings.append(f"inconsistent across folds (stability={wf['stability']:.2f})")
    if not beats_baseline:
        warnings.append("no edge vs random-entry baseline")

    pf = m["profit_factor"]
    return {
        "side": side,
        "name": f"{side.upper()} {_rr_label(up_pct, dn_pct)}",
        "conditions": conditions,
        "n_signals": len(signals),
        "metrics": {
            "trade_count": m["trade_count"],
            "win_rate": m["win_rate"],
            "profit_factor": pf if np.isfinite(pf) else None,
            "expectancy": m["expectancy"],
            "avg_win": m["avg_win"],
            "avg_loss": m["avg_loss"],
            "sharpe": sharpe,
            "target_hit_rate": m["target_hit_rate"],
            "avg_bars_held": m["avg_bars_held"],
            "p_value": p_value,
        },
        "baseline": {"win_rate": base["win_rate"], "expectancy": base["expectancy"]},
        "edge_win_rate": edge_wr,
        "edge_expectancy": edge_exp,
        "folds": wf["folds"],
        "stability": wf["stability"],
        "is_robust": is_robust,
        "warnings": warnings,
        "exits": {"target_pct": up_pct, "stop_pct": dn_pct, "time_stop_bars": horizon},
    }


def _rr_label(up_pct: float, dn_pct: float) -> str:
    ratio = up_pct / dn_pct if dn_pct else 0
    return f"{ratio:.0f}:1" if ratio == int(ratio) else f"{ratio:.1f}:1"


def run_player(
    frames: Dict[str, pd.DataFrame],
    up_pct: float = 0.010,
    dn_pct: float = 0.005,
    leverage: float = 10.0,
    cost: float = 0.0009,
    horizon: int = 120,
    cfg: RuleConfig = RuleConfig(),
) -> Dict[str, dict]:
    """Backtest the 5-condition rule both directions. Powers the UI Strategy Player."""
    required = {"1m", "15m", "1h", "4h"}
    missing = required - set(frames)
    if missing:
        return {"error": f"missing timeframes for rule strategy: {sorted(missing)}"}

    mask = build_signal_mask(frames, cfg)
    d1 = frames["1m"].sort_values("open_time").reset_index(drop=True)
    return {
        "rr": _rr_label(up_pct, dn_pct),
        "long": backtest_rule_side(mask, d1, "long", up_pct, dn_pct, leverage, cost, horizon, cfg),
        "short": backtest_rule_side(mask, d1, "short", up_pct, dn_pct, leverage, cost, horizon, cfg),
    }
