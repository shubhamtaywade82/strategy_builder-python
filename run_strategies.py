"""
Multi-Symbol Rule-Strategy Runner
==================================
Backtests the report's 5-condition rule (LONG + SHORT) per symbol with
LEAKAGE-SAFE features, out-of-sample walk-forward folds, and a bootstrap
significance test. Emits a bot-usable strategies.json — only strategies that
pass the robustness gate are marked `deploy: true`.

This is the HONEST counterpart to grid_search's model-confidence strategies
(which overfit — see docs/strategy_analysis_report). The 5-condition rule is
what trading_bot.py actually executes.

Run:  python run_strategies.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from strategy_research._ui_engine.data_fetcher import fetch_symbol_mtf
from strategy_research._ui_engine.backtest import backtest
# Shared 5-condition SMC mask — single source of truth (also used by the UI
# Strategy Player and trading_bot.py) so the rule can't drift between copies.
from strategy_research._ui_engine.rule_strategy import build_signal_mask as _shared_mask, RuleConfig

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
DAYS = 60
LEVERAGE = 10.0
COST = 0.0009          # round-trip fee (taker) used by backtest
HORIZON = 120          # 2h time-stop on 1m
RR = {"up_pct": 0.010, "dn_pct": 0.005, "label": "2:1 (1.0%/0.5%)"}

# RR grid swept per side; signals (5-cond mask) are RR-independent, only barriers change
RR_GRID = {
    "2:1":   {"up_pct": 0.010, "dn_pct": 0.005, "label": "2:1 (1.0%/0.5%)"},
    "1.5:1": {"up_pct": 0.0075, "dn_pct": 0.005, "label": "1.5:1 (0.75%/0.5%)"},
}
# fee scenarios (round-trip): taker≈0.09%, maker≈0.04%, zero=pure edge
FEE_SCENARIOS = {"taker": 0.0009, "maker": 0.0004, "zero": 0.0}
COOLDOWN_BARS = 30      # skip new signals within N bars of last entry (dedupe overlap)
BASELINE_SAMPLE = 3000  # random bars to estimate null WR/expectancy per RR/side

# rule knobs
SWING = 5              # 1H BOS swing lookback
FVG_LOOKBACK = 10      # 15m FVG recency window
ATR_WIN = 1440         # trailing window for ATR percentile (1 day of 1m)
VOL_WIN = 1440         # trailing window for volume percentile
ATR_Q = 0.60           # condition 4 threshold
VOL_Q = 0.70           # condition 5 threshold
N_FOLDS = 5
N_BOOT = 2000
MIN_TRADES = 30


# --------------------------------------------------------------------------
# leakage-safe condition columns (delegates to the shared rule module)
# --------------------------------------------------------------------------
_RULE_CFG = RuleConfig(
    swing=SWING, fvg_lookback=FVG_LOOKBACK, atr_win=ATR_WIN, vol_win=VOL_WIN,
    atr_q=ATR_Q, vol_q=VOL_Q, cooldown_bars=COOLDOWN_BARS,
    n_folds=N_FOLDS, n_boot=N_BOOT, min_trades=MIN_TRADES, baseline_sample=BASELINE_SAMPLE,
)


def build_signal_mask(frames: dict) -> pd.DataFrame:
    """Return 1m frame with per-bar long/short signal flags. No lookahead.

    Thin wrapper around the shared engine mask so this multi-symbol runner and
    the dashboard's Strategy Player evaluate byte-identical conditions.
    """
    return _shared_mask(frames, _RULE_CFG)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def walk_forward_folds(trades: list, n_bars: int, n_folds: int = N_FOLDS) -> dict:
    """Per-fold expectancy/WR by entry-bar time slice (out-of-time consistency)."""
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
        folds.append({
            "fold": k + 1, "trades": len(ft),
            "expectancy": float(pnl.mean()),
            "win_rate": float((pnl > 0).mean()),
        })
    scored = [f for f in folds if f["expectancy"] is not None]
    stability = (sum(1 for f in scored if f["expectancy"] > 0) / len(scored)) if scored else 0.0
    return {"folds": folds, "stability": stability}


def bootstrap_pvalue(pnl: np.ndarray, n_boot: int = N_BOOT) -> float:
    """One-sided: probability the true mean per-trade pnl is <= 0."""
    if len(pnl) < 5:
        return 1.0
    rng = np.random.default_rng(42)
    means = rng.choice(pnl, size=(n_boot, len(pnl)), replace=True).mean(axis=1)
    return float((means <= 0).mean())


def robustness(metrics: dict, wf: dict, p_value: float, beats_baseline: bool) -> dict:
    n = metrics.get("trade_count", 0)
    exp = metrics.get("expectancy", 0.0)
    stab = wf["stability"]
    is_robust = bool(exp > 0 and stab >= 0.6 and p_value < 0.05
                     and n >= MIN_TRADES and beats_baseline)

    # 0-100 score
    sig_pts = (1 - min(p_value / 0.05, 1.0)) * 35
    stab_pts = stab * 30
    exp_pts = float(np.clip(exp / 0.001, 0, 1)) * 20   # 0.10% net/trade -> full
    vol_pts = float(np.clip(n / 100, 0, 1)) * 15       # sample size
    score = round(sig_pts + stab_pts + exp_pts + vol_pts, 1)

    warnings = []
    if n < MIN_TRADES:
        warnings.append(f"low sample (n={n})")
    if p_value >= 0.05:
        warnings.append(f"not significant (p={p_value:.3f})")
    if stab < 0.6:
        warnings.append(f"inconsistent across folds (stability={stab:.2f})")
    if exp <= 0:
        warnings.append("negative expectancy")
    if not beats_baseline:
        warnings.append("no edge vs random baseline")
    return {"robustness_score": score, "is_robust": is_robust, "warnings": warnings}


def apply_cooldown(idxs: list, bars: int = COOLDOWN_BARS) -> list:
    """Keep only signals at least `bars` apart (dedupe overlapping entries)."""
    kept, last = [], -10**9
    for i in idxs:
        if i - last >= bars:
            kept.append(i)
            last = i
    return kept


def baseline_metrics(d1: pd.DataFrame, side: str, rr: dict, cost: float) -> dict:
    """Null hypothesis: WR/expectancy of RANDOM entries at this RR (geometry+fees)."""
    side_val = 1 if side == "long" else -1
    n = len(d1)
    rng = np.random.default_rng(7)
    idx = rng.choice(np.arange(0, n - HORIZON - 1), size=min(BASELINE_SAMPLE, n - HORIZON - 1),
                     replace=False)
    sigs = [{"bar_idx": int(i), "side": side_val, "confidence": 1.0} for i in idx]
    bt = backtest(d1, sigs, rr["up_pct"], rr["dn_pct"], LEVERAGE, cost, HORIZON)
    m = bt.metrics or {}
    return {"win_rate": m.get("win_rate", 0.0), "expectancy": m.get("expectancy", 0.0)}


# --------------------------------------------------------------------------
# per symbol / side
# --------------------------------------------------------------------------
def evaluate_side(mask: pd.DataFrame, d1: pd.DataFrame, side: str, rr: dict,
                  base: dict, cost: float, fee_label: str) -> dict:
    col = "sig_long" if side == "long" else "sig_short"
    side_val = 1 if side == "long" else -1
    sig_idx = apply_cooldown(mask.index[mask[col]].tolist())
    signals = [{"bar_idx": int(i), "side": side_val, "confidence": 1.0} for i in sig_idx]

    bt = backtest(d1, signals, rr["up_pct"], rr["dn_pct"], LEVERAGE, cost, HORIZON)
    m = bt.metrics
    if not m or m.get("trade_count", 0) == 0:
        return {"side": side, "rr": rr["label"], "fee": fee_label, "n_signals": len(signals),
                "metrics": {}, "error": "no trades", "is_robust": False,
                "robustness_score": 0, "warnings": ["no trades"]}

    pnl = np.array([t["net_pnl"] for t in bt.trades])
    wf = walk_forward_folds(bt.trades, len(d1))
    p_value = bootstrap_pvalue(pnl)
    sharpe = float(pnl.mean() / pnl.std()) if pnl.std() > 0 else 0.0
    # edge: must beat random-entry baseline on BOTH win-rate and expectancy
    edge_wr = m["win_rate"] - base["win_rate"]
    edge_exp = m["expectancy"] - base["expectancy"]
    beats_baseline = edge_wr > 0 and edge_exp > 0
    rob = robustness(m, wf, p_value, beats_baseline)

    return {
        "side": side,
        "rr": rr["label"],
        "fee": fee_label,
        "n_signals": len(signals),
        "baseline": {"win_rate": round(base["win_rate"], 4),
                     "expectancy": round(base["expectancy"], 6)},
        "edge_win_rate": round(edge_wr, 4),
        "edge_expectancy": round(edge_exp, 6),
        "metrics": {
            "trade_count": m["trade_count"],
            "win_rate": round(m["win_rate"], 4),
            "profit_factor": round(m["profit_factor"], 3) if np.isfinite(m["profit_factor"]) else None,
            "expectancy_net": round(m["expectancy"], 6),
            "avg_win": round(m["avg_win"], 6),
            "avg_loss": round(m["avg_loss"], 6),
            "sharpe": round(sharpe, 3),
            "target_hit_rate": round(m["target_hit_rate"], 4),
            "avg_bars_held": round(m["avg_bars_held"], 1),
            "p_value": round(p_value, 4),
        },
        "walk_forward": wf,
        **rob,
    }


def run_symbol(symbol: str) -> dict:
    print(f"\n{'='*64}\n{symbol}\n{'='*64}")
    frames = fetch_symbol_mtf(symbol, days=DAYS, timeframes=["1m", "15m", "1h"])
    # 4H EMA200 needs ~33d warmup; fetch extra history so trend filter is valid
    # from day 1 of the 1m window (early 4H bars are context, predate 1m range).
    warm = fetch_symbol_mtf(symbol, days=DAYS + 40, timeframes=["4h"])
    frames["4h"] = warm["4h"]
    if "1m" not in frames or len(frames["1m"]) < 5000:
        return {"symbol": symbol, "error": "insufficient 1m data"}

    mask = build_signal_mask(frames)
    d1 = frames["1m"].sort_values("open_time").reset_index(drop=True)

    out = {"symbol": symbol, "bars_1m": len(d1)}
    for side in ("long", "short"):
        candidates = []
        for fee_label, cost in FEE_SCENARIOS.items():
            for rr in RR_GRID.values():
                base = baseline_metrics(d1, side, rr, cost)
                res = evaluate_side(mask, d1, side, rr, base, cost, fee_label)
                candidates.append(res)
                m = res.get("metrics", {})
                if m:
                    flag = "DEPLOY" if res["is_robust"] else "reject"
                    print(f"  {side.upper():5s} {fee_label:5s} {rr['label']:18s} n={m['trade_count']:4d} "
                          f"WR={m['win_rate']*100:5.1f}% (base {res['baseline']['win_rate']*100:4.1f}%) "
                          f"netE={m['expectancy_net']*100:+.3f}% edgeWR={res['edge_win_rate']*100:+.1f}% "
                          f"p={m['p_value']:.3f} -> {flag}")
        # best = robust first, then highest expectancy
        best = max(candidates, key=lambda r: (r.get("is_robust", False),
                                              r.get("metrics", {}).get("expectancy_net", -9)))
        out[side] = {"best": best, "all_rr": candidates}
    return out


def to_bot_config(all_results: list) -> dict:
    """Distill into deployable per-symbol/side configs for trading bots."""
    cfg = {"generated_utc": None, "leverage": LEVERAGE,
           "horizon_bars": HORIZON, "rule": "5-condition (4H trend, 1H BOS, 15m FVG, ATR>60p, Vol>70p)",
           "strategies": []}
    for r in all_results:
        for side in ("long", "short"):
            res = r.get(side, {}).get("best", {})
            m = res.get("metrics", {})
            if not m:
                continue
            cfg["strategies"].append({
                "symbol": r["symbol"], "side": side,
                "deploy": res["is_robust"],
                "rr": res["rr"], "fee_scenario": res.get("fee"),
                "robustness_score": res["robustness_score"],
                "win_rate": m["win_rate"], "profit_factor": m["profit_factor"],
                "expectancy_net": m["expectancy_net"], "sharpe": m["sharpe"],
                "p_value": m["p_value"], "trades": m["trade_count"],
                "edge_win_rate": res.get("edge_win_rate"),
                "edge_expectancy": res.get("edge_expectancy"),
                "warnings": res["warnings"],
                "exits": {"time_stop_bars": HORIZON},
            })
    cfg["strategies"].sort(key=lambda s: (s["deploy"], s["robustness_score"]), reverse=True)
    return cfg


def main():
    t0 = time.time()
    all_results = [run_symbol(s) for s in SYMBOLS]

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    raw_path = out_dir / f"rule_research_{stamp}.json"
    raw_path.write_text(json.dumps(all_results, indent=2, default=str))

    cfg = to_bot_config(all_results)
    cfg["generated_utc"] = stamp
    cfg_path = out_dir / "strategies.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, default=str))

    print(f"\n{'#'*64}\nDEPLOYABLE STRATEGIES (passed robustness gate)\n{'#'*64}")
    deploy = [s for s in cfg["strategies"] if s["deploy"]]
    if not deploy:
        print("  NONE passed. All rule variants are in-sample noise on this window.")
    for s in deploy:
        print(f"  {s['symbol']:8s} {s['side'].upper():5s} {s['fee_scenario']:5s} {s['rr']:18s} score={s['robustness_score']:.0f} "
              f"WR={s['win_rate']*100:.1f}% PF={s['profit_factor']} "
              f"netE={s['expectancy_net']*100:+.4f}% p={s['p_value']:.3f} n={s['trades']}")

    print(f"\nRaw: {raw_path}\nBot config: {cfg_path}\nElapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
