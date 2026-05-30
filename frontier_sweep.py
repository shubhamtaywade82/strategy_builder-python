"""
Win/RR Frontier Sweep — find the real best operating point (not a decreed 50%/1:2).
====================================================================================
For the two entries that showed positive out-of-sample edge (Liquidity_Sweep_MSS,
MTF-alignment), sweep the take-profit from 1:1 → 1:3 plus partial/trail variants
across SOL/XRP/ETH/BTC, anchored OOS holdout, and report win% + net expectancy +
PF + trades at each. The max-expectancy variant per (symbol, entry) is the actual
best operating point — chosen by data, not by demand. All metrics NET of fees.

Usage:  ./venv/bin/python frontier_sweep.py --days 90
"""
import argparse
import logging
from datetime import datetime, timedelta

from strategy_builder.market_data.candle_loader import CandleLoader
from strategy_builder.backtest.engine import BacktestEngine
from strategy_builder.backtest.signal_evaluator import SignalEvaluator
from strategy_builder.backtest.walk_forward import WalkForward

# Reuse the cached MTF-alignment condition + precompute from the edge build.
from build_mtf_edge import precompute as mtf_precompute, _register as mtf_register, MACRO_TFS

logging.basicConfig(level=logging.WARNING)
SYMBOLS = ["SOLUSDT", "XRPUSDT", "ETHUSDT", "BTCUSDT"]

# Each variant: (label, exit-dict). Exit R-multiples are multiples of the entry's stop.
VARIANTS = [
    ("R=1.0", {"targets": [1.0]}),
    ("R=1.5", {"targets": [1.5]}),
    ("R=2.0", {"targets": [2.0]}),
    ("R=2.5", {"targets": [2.5]}),
    ("part 1R+2R", {"targets": [1.0, 2.0], "partial_exits": [0.5, 0.5]}),
]

# (label, entry condition, base timeframe)
ALL_ENTRIES = [
    ("Liquidity_Sweep_MSS", "liquidity_sweep_mss_entry", "15m"),
    ("MTF_Alignment",       "mtf_edge_entry",            "5m"),
]


def run():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--symbols", default=",".join(SYMBOLS))
    p.add_argument("--entry", default="all", choices=["all", "sweep", "mtf"])
    a = p.parse_args()
    mtf_register()
    if a.entry == "sweep":
        ENTRIES = [e for e in ALL_ENTRIES if e[1] == "liquidity_sweep_mss_entry"]
    elif a.entry == "mtf":
        ENTRIES = [e for e in ALL_ENTRIES if e[1] == "mtf_edge_entry"]
    else:
        ENTRIES = ALL_ENTRIES
    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    loader = CandleLoader(market_data_source="binance")
    wf = WalkForward(engine=BacktestEngine())
    end = datetime.now(); start = end - timedelta(days=a.days)
    tfs = sorted(set(["5m", "15m"] + MACRO_TFS))

    print("=" * 104)
    print(f"WIN/RR FRONTIER SWEEP | {a.days}d | anchored OOS | NET of fees | "
          f"goal: max net expectancy (R), not a decreed 50%/1:2")
    print("=" * 104)

    best = {}   # (symbol, entry) -> (label, exp, win, pf, tr)
    for sym in syms:
        try:
            mtf = loader.fetch_mtf(instrument=sym, timeframes=tfs, start_from=start, to=end)
        except Exception as e:
            print(f"\n{sym}: fetch error {e}"); continue
        print(f"\n### {sym}")
        print(f"{'entry':<18}{'variant':<13}{'OOS_tr':>7}{'win%':>7}{'PF':>7}{'exp(R-ish)':>12}{'win/loss':>9}")
        print("-" * 104)
        for ename, cond, base in ENTRIES:
            if base == "5m":
                mtf_precompute(mtf["5m"], mtf)   # populate O(1) cache for this symbol
            for label, exit_cfg in VARIANTS:
                cfg = {"name": f"{ename}_{label}", "entry": {"conditions": [cond]},
                       "exit": exit_cfg, "timeframes": [base] + MACRO_TFS}
                try:
                    ev = SignalEvaluator.build(cfg, mtf_candles=mtf)
                    ho = wf.anchored_holdout(strategy=cfg, candles=mtf[base],
                                             signal_generator=ev, mtf_candles=mtf)
                    if ho is None:
                        continue
                    o = ho["out_of_sample"]
                    tr = o.get("trade_count", 0); win = o.get("win_rate", 0.0) * 100
                    pf = o.get("profit_factor", 0.0); exp = o.get("expectancy", 0.0)
                    aw = o.get("avg_win", 0.0); al = abs(o.get("avg_loss", 0.0))
                    rr = aw / al if al > 0 else 0.0
                    avg_r = o.get("avg_r", 0.0)
                    print(f"{ename:<18}{label:<13}{tr:>7}{win:>6.1f}%{pf:>7.2f}{avg_r:>12.3f}{rr:>9.2f}")
                    key = (sym, ename)
                    if tr >= 15 and (key not in best or avg_r > best[key][1]):
                        best[key] = (label, avg_r, win, pf, tr)
                except Exception as e:
                    print(f"{ename:<18}{label:<13}  err {str(e)[:50]}")

    print("\n" + "=" * 104)
    print("BEST OPERATING POINT per (symbol, entry) — max net avg-R, ≥15 OOS trades:")
    print("-" * 104)
    for (sym, ename), (label, avg_r, win, pf, tr) in sorted(best.items()):
        tag = "EDGE" if avg_r > 0 and pf > 1.0 else "no edge"
        print(f"  {sym:<8} {ename:<18} → {label:<12} avgR={avg_r:+.3f} win={win:.0f}% PF={pf:.2f} tr={tr}  [{tag}]")
    print("=" * 104)


if __name__ == "__main__":
    run()
