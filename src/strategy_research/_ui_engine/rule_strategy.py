"""
Rule-Strategy Backtest (honest counterpart to grid_search)
==========================================================
Backtests the report's transparent 5-condition rule with LEAKAGE-SAFE features,
out-of-time walk-forward folds, a bootstrap significance test, and a random-entry
baseline. Tests BOTH directions in BOTH modes:

  momentum  — trade the signal direction (chase the breakout)
  reversion — fade the signal direction (mean-revert the breakout)

Empirically (see docs/strategy_analysis_report/RULE_VALIDATION_FINDINGS.md) the
momentum mode is anti-predictive; the reversion mode (fade the bearish breakout =
go long) carries a thin but consistent cross-symbol edge. This module surfaces the
REAL number for each, gated by robustness — no fabricated metrics.

Single source of truth shared by the dashboard (ui_research) and run_strategies.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import backtest

# rule knobs (match run_strategies.py / trading_bot.py)
SWING = 5
FVG_LOOKBACK = 10
ATR_WIN = 1440
VOL_WIN = 1440
ATR_Q = 0.60
VOL_Q = 0.70
COOLDOWN_BARS = 30
N_FOLDS = 5
N_BOOT = 2000
MIN_TRADES = 30
BASELINE_SAMPLE = 3000

CONDITIONS = [
    {"id": "trend", "label": "4H EMA50 vs EMA200"},
    {"id": "bos", "label": "1H Break of Structure"},
    {"id": "fvg", "label": "15m FVG (recent)"},
    {"id": "atr", "label": f"ATR(14) > {int(ATR_Q*100)}th pct"},
    {"id": "vol", "label": f"Volume > {int(VOL_Q*100)}th pct"},
]


# --------------------------------------------------------------------------
# leakage-safe signal construction
# --------------------------------------------------------------------------
def _htf_cond(frame: pd.DataFrame, cond_long: pd.Series, cond_short: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"ref_time": frame["close_time"].reset_index(drop=True)})
    df["c_long"] = cond_long.reset_index(drop=True).fillna(False).astype(bool)
    df["c_short"] = cond_short.reset_index(drop=True).fillna(False).astype(bool)
    return df.dropna(subset=["ref_time"]).sort_values("ref_time")


def build_signal_mask(frames: dict) -> pd.DataFrame:
    """1m frame with per-bar long/short signal flags. No lookahead."""
    d1 = frames["1m"].sort_values("open_time").reset_index(drop=True)
    out = d1[["open_time", "close_time", "open", "high", "low", "close", "volume"]].copy()
    out["key"] = out["open_time"]

    f4 = frames["4h"].sort_values("open_time")
    e50 = f4["close"].ewm(span=50, adjust=False).mean()
    e200 = f4["close"].ewm(span=200, adjust=False).mean()
    c1 = _htf_cond(f4, e50 > e200, e50 < e200).rename(
        columns={"c_long": "c1_long", "c_short": "c1_short"})

    f1h = frames["1h"].sort_values("open_time")
    prev_high = f1h["high"].rolling(SWING).max().shift(1)
    prev_low = f1h["low"].rolling(SWING).min().shift(1)
    c2 = _htf_cond(f1h, f1h["close"] > prev_high, f1h["close"] < prev_low).rename(
        columns={"c_long": "c2_long", "c_short": "c2_short"})

    f15 = frames["15m"].sort_values("open_time")
    bull_fvg = (f15["low"] > f15["high"].shift(2))
    bear_fvg = (f15["high"] < f15["low"].shift(2))
    c3l = bull_fvg.rolling(FVG_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    c3s = bear_fvg.rolling(FVG_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    c3 = _htf_cond(f15, c3l, c3s).rename(columns={"c_long": "c3_long", "c_short": "c3_short"})

    for blk in (c1, c2, c3):
        out = pd.merge_asof(
            out.sort_values("key"), blk, left_on="key", right_on="ref_time",
            direction="backward", allow_exact_matches=True,
        ).drop(columns=["ref_time"], errors="ignore")

    c = out["close"]
    tr = pd.concat([out["high"] - out["low"],
                    (out["high"] - c.shift()).abs(),
                    (out["low"] - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    out["c4"] = (atr > atr.rolling(ATR_WIN, min_periods=ATR_WIN // 4).quantile(ATR_Q).shift(1))
    out["c5"] = (out["volume"] >
                 out["volume"].rolling(VOL_WIN, min_periods=VOL_WIN // 4).quantile(VOL_Q).shift(1))

    for col in ["c1_long", "c1_short", "c2_long", "c2_short",
                "c3_long", "c3_short", "c4", "c5"]:
        out[col] = out[col].fillna(False).astype(bool)

    out["sig_long"] = out["c1_long"] & out["c2_long"] & out["c3_long"] & out["c4"] & out["c5"]
    out["sig_short"] = out["c1_short"] & out["c2_short"] & out["c3_short"] & out["c4"] & out["c5"]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# validation primitives
# --------------------------------------------------------------------------
def apply_cooldown(idxs: list, bars: int = COOLDOWN_BARS) -> list:
    kept, last = [], -10**9
    for i in idxs:
        if i - last >= bars:
            kept.append(i)
            last = i
    return kept


def walk_forward_folds(trades: list, n_bars: int, n_folds: int = N_FOLDS) -> dict:
    if not trades:
        return {"folds": [], "stability": 0.0}
    edges = np.linspace(0, n_bars, n_folds + 1).astype(int)
    folds = []
    for k in range(n_folds):
        lo, hi = edges[k], edges[k + 1]
        ft = [t for t in trades if lo <= t["entry_idx"] < hi]
        if len(ft) < 5:
            folds.append({"fold": k + 1, "trades": len(ft), "expectancy": None, "win_rate": None})
            continue
        pnl = np.array([t["net_pnl"] for t in ft])
        folds.append({"fold": k + 1, "trades": len(ft),
                      "expectancy": float(pnl.mean()), "win_rate": float((pnl > 0).mean())})
    scored = [f for f in folds if f["expectancy"] is not None]
    stability = (sum(1 for f in scored if f["expectancy"] > 0) / len(scored)) if scored else 0.0
    return {"folds": folds, "stability": stability}


def bootstrap_pvalue(pnl: np.ndarray, n_boot: int = N_BOOT) -> float:
    if len(pnl) < 5:
        return 1.0
    rng = np.random.default_rng(42)
    means = rng.choice(pnl, size=(n_boot, len(pnl)), replace=True).mean(axis=1)
    return float((means <= 0).mean())


def _robustness(metrics: dict, wf: dict, p_value: float, beats_baseline: bool) -> dict:
    n = metrics.get("trade_count", 0)
    exp = metrics.get("expectancy", 0.0)
    stab = wf["stability"]
    is_robust = bool(exp > 0 and stab >= 0.6 and p_value < 0.05
                     and n >= MIN_TRADES and beats_baseline)
    sig_pts = (1 - min(p_value / 0.05, 1.0)) * 35
    stab_pts = stab * 30
    exp_pts = float(np.clip(exp / 0.001, 0, 1)) * 20
    vol_pts = float(np.clip(n / 100, 0, 1)) * 15
    score = round(sig_pts + stab_pts + exp_pts + vol_pts, 1)
    warnings = []
    if n < MIN_TRADES:
        warnings.append(f"low sample (n={n})")
    if p_value >= 0.05:
        warnings.append(f"not significant (p={p_value:.3f})")
    if stab < 0.6:
        warnings.append(f"inconsistent across folds ({stab:.2f})")
    if exp <= 0:
        warnings.append("negative expectancy")
    if not beats_baseline:
        warnings.append("no edge vs random baseline")
    return {"robustness_score": score, "is_robust": is_robust, "warnings": warnings}


def _baseline(d1: pd.DataFrame, side_val: int, rr: dict, cost: float, lev: float, horizon: int) -> dict:
    n = len(d1)
    rng = np.random.default_rng(7)
    hi = max(n - horizon - 1, 1)
    idx = rng.choice(np.arange(0, hi), size=min(BASELINE_SAMPLE, hi), replace=False)
    sigs = [{"bar_idx": int(i), "side": side_val, "confidence": 1.0} for i in idx]
    m = (backtest(d1, sigs, rr["up_pct"], rr["dn_pct"], lev, cost, horizon).metrics or {})
    return {"win_rate": m.get("win_rate", 0.0), "expectancy": m.get("expectancy", 0.0)}


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def _evaluate(mask, d1, side, mode, rr, cost, lev, horizon) -> dict:
    invert = (mode == "reversion")
    col = "sig_long" if side == "long" else "sig_short"
    side_val = (1 if side == "long" else -1) * (-1 if invert else 1)
    sig_idx = apply_cooldown(mask.index[mask[col]].tolist())
    signals = [{"bar_idx": int(i), "side": side_val, "confidence": 1.0} for i in sig_idx]

    bt = backtest(d1, signals, rr["up_pct"], rr["dn_pct"], lev, cost, horizon)
    m = bt.metrics
    base = {"win_rate": 0.0, "expectancy": 0.0}
    if not m or m.get("trade_count", 0) == 0:
        return {"side": side, "mode": mode, "trade_direction": "long" if side_val > 0 else "short",
                "n_signals": len(signals), "metrics": {}, "is_robust": False,
                "robustness_score": 0, "warnings": ["no trades"],
                "walk_forward": {"folds": [], "stability": 0.0}, "baseline": base}

    pnl = np.array([t["net_pnl"] for t in bt.trades])
    wf = walk_forward_folds(bt.trades, len(d1))
    p_value = bootstrap_pvalue(pnl)
    sharpe = float(pnl.mean() / pnl.std()) if pnl.std() > 0 else 0.0
    base = _baseline(d1, side_val, rr, cost, lev, horizon)
    edge_wr = m["win_rate"] - base["win_rate"]
    edge_exp = m["expectancy"] - base["expectancy"]
    beats = edge_wr > 0 and edge_exp > 0
    rob = _robustness(m, wf, p_value, beats)

    return {
        "side": side, "mode": mode,
        "trade_direction": "long" if side_val > 0 else "short",
        "n_signals": len(signals),
        "baseline": {"win_rate": round(base["win_rate"], 4),
                     "expectancy": round(base["expectancy"], 6)},
        "edge_win_rate": round(edge_wr, 4), "edge_expectancy": round(edge_exp, 6),
        "metrics": {
            "trade_count": m["trade_count"], "win_rate": round(m["win_rate"], 4),
            "profit_factor": round(m["profit_factor"], 3) if np.isfinite(m["profit_factor"]) else None,
            "expectancy_net": round(m["expectancy"], 6),
            "avg_win": round(m["avg_win"], 6), "avg_loss": round(m["avg_loss"], 6),
            "sharpe": round(sharpe, 3), "target_hit_rate": round(m["target_hit_rate"], 4),
            "avg_bars_held": round(m["avg_bars_held"], 1), "p_value": round(p_value, 4),
        },
        "walk_forward": wf, **rob,
    }


def backtest_rule(frames: dict, rr: dict, cost: float = 0.0009,
                  leverage: float = 10.0, horizon: int = 120) -> dict:
    """Full rule-strategy result for the dashboard 'player' block.

    Returns both directions in both modes, plus the single best (robust-first,
    then highest expectancy) for headline display.
    """
    mask = build_signal_mask(frames)
    d1 = frames["1m"].sort_values("open_time").reset_index(drop=True)

    cells, flat = {}, []
    for side in ("long", "short"):
        cells[side] = {}
        for mode in ("momentum", "reversion"):
            res = _evaluate(mask, d1, side, mode, rr, cost, leverage, horizon)
            cells[side][mode] = res
            if res.get("metrics"):
                flat.append(res)

    best = max(flat, key=lambda r: (r["is_robust"], r["metrics"]["expectancy_net"])) if flat else None
    return {
        "rr": rr.get("label", ""), "cost": cost, "leverage": leverage,
        "horizon_bars": horizon, "conditions": CONDITIONS,
        "long": cells["long"], "short": cells["short"], "best": best,
        "has_edge": bool(best and best["is_robust"]),
    }
