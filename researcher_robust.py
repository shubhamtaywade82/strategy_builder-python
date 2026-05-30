"""
Robust Strategy Researcher — walk-forward + gatekeeper (no overfitting).
========================================================================
The shipped researcher.py sorts strategies by raw profit_factor with NO
minimum-trade filter and NO out-of-sample split — so a strategy that took 2
lucky trades ranks above one with a real 50-trade edge. That's how you get a
"PF 2.42 on 5 trades" recommendation that means nothing.

This version uses the infra the repo already ships but the script ignored:
  * WalkForward (5-fold in-sample/out-of-sample) — every fold trains then tests
    on unseen forward data, so we measure edge that PERSISTS, not edge fit to
    history.
  * Gatekeeper — hard gates: ≥20 OOS trades, OOS profit_factor ≥ 1.1, OOS
    expectancy > 0, stability ≥ 0.6 of folds positive, IS→OOS degradation cap.
Fees (0.05%/side) + slippage (2bps) are modelled by the engine, so all numbers
are net.

Ranks surviving strategies by OOS expectancy. Anything that fails the gates is
reported as REJECT and must NOT be traded — a strategy that can't clear an
out-of-sample gate has no demonstrated edge regardless of how good its in-sample
profit_factor looks.

Usage:
    ./venv/bin/python researcher_robust.py --symbol BTCUSDT --days 90
    ./venv/bin/python researcher_robust.py --symbol ETHUSDT --days 120 --folds 6
"""
import argparse
import logging
import time
from datetime import datetime, timedelta

from strategy_builder.market_data.candle_loader import CandleLoader
from strategy_builder.backtest.engine import BacktestEngine
from strategy_builder.backtest.signal_evaluator import SignalEvaluator
from strategy_builder.backtest.walk_forward import WalkForward
from strategy_builder.ranking.gatekeeper import Gatekeeper

# reuse the same strategy library the shipped researcher defines
from researcher import STRATEGY_LIBRARY

# Strategies that produce a real sample (≥~40 trades in 30d). The rest are
# low-frequency setups that can never clear a min-trade gate in this window, so
# scoring them by profit_factor is meaningless. Default focus set.
FOCUS = {
    "Liquidity_Sweep_MSS", "Bollinger_Band_Walk", "Opening_Range_Breakout",
    "Order_Block_Mitigation", "RSI_Divergence_Structure", "Optimized_Baseline_2.5R",
    "Session_Fade_Pro", "Delta_Neutral_Basis_Trade", "Filtered_Trend_Following",
}


def _holdout_gate(oos, oos_tr):
    """Pass/fail on a single anchored OOS block (pooled, not mean-of-folds)."""
    pf = oos.get("profit_factor", 0.0)
    exp = oos.get("expectancy", 0.0)
    wr = oos.get("win_rate", 0.0)
    fails = []
    if oos_tr < 30:
        fails.append(f"trades {oos_tr}<30")
    if pf < 1.1:
        fails.append(f"PF {pf:.2f}<1.1")
    if exp <= 0:
        fails.append("exp<=0")
    if wr < 0.25:
        fails.append(f"wr {wr:.0%}<25%")
    status = "pass" if not fails else ("watchlist" if len(fails) <= 1 else "reject")
    return status, fails

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RobustResearcher")


def research(symbol: str, days: int, folds: int, holdout: bool, focus: bool):
    symbol = symbol.upper()
    mode = "anchored-holdout 65/35" if holdout else f"{folds}-fold walk-forward"
    print("\n" + "=" * 78)
    print(f"ROBUST RESEARCHER ({mode} + gates): {symbol}  |  {days}d")
    print("=" * 78)

    loader = CandleLoader(market_data_source="binance")
    engine = BacktestEngine()
    wf = WalkForward(engine=engine)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    print(f"Fetching {symbol} 1h+15m {start_date.date()} → {end_date.date()} …")
    mtf = loader.fetch_mtf(instrument=symbol, timeframes=["1h", "15m"],
                           start_from=start_date, to=end_date)
    time.sleep(1.0)
    print(f"Got {len(mtf['15m'])} 15m candles, {len(mtf['1h'])} 1h candles")

    library = [c for c in STRATEGY_LIBRARY if (not focus or c["name"] in FOCUS)]
    rows = []
    for cfg in library:
        cfg = dict(cfg)
        cfg["timeframes"] = ["1h", "15m"]
        try:
            evaluator = SignalEvaluator.build(cfg, mtf_candles=mtf)
            if holdout:
                ho = wf.anchored_holdout(strategy=cfg, candles=mtf["15m"],
                                         signal_generator=evaluator, mtf_candles=mtf)
                if ho is None:
                    raise RuntimeError("insufficient data for holdout")
                oos = ho["out_of_sample"]; is_m = ho["in_sample"]
                oos_tr = oos.get("trade_count", 0)
                status, fails = _holdout_gate(oos, oos_tr)
                row = {"name": cfg["name"], "status": status,
                       "oos_pf": oos.get("profit_factor", 0.0),
                       "oos_exp": oos.get("expectancy", 0.0), "oos_tr": oos_tr,
                       "oos_wr": oos.get("win_rate", 0.0), "stab": 0.0,
                       "deg": 0.0, "is_pf": is_m.get("profit_factor", 0.0),
                       "fails": "; ".join(fails[:2])}
            else:
                wf_res = wf.run(strategy=cfg, candles=mtf["15m"],
                                signal_generator=evaluator, folds=folds, mtf_candles=mtf)
                agg = wf_res["aggregate"]
                gate = Gatekeeper.evaluate(wf_res)
                row = {"name": cfg["name"], "status": gate["status"],
                       "oos_pf": agg.get("oos_profit_factor", 0.0),
                       "oos_exp": agg.get("oos_expectancy", 0.0),
                       "oos_tr": agg.get("oos_trade_count", 0),
                       "oos_wr": agg.get("oos_win_rate", 0.0),
                       "stab": wf_res.get("stability_score", 0.0),
                       "deg": agg.get("avg_degradation", 1.0),
                       "is_pf": agg.get("is_profit_factor", 0.0),
                       "fails": "; ".join(gate["failures"][:2])}
        except Exception as e:
            logger.warning("skip %s: %s", cfg["name"], e)
            row = {"name": cfg["name"], "status": "error", "oos_pf": 0.0, "oos_exp": 0.0,
                   "oos_tr": 0, "oos_wr": 0.0, "stab": 0.0, "deg": 1.0, "is_pf": 0.0,
                   "fails": str(e)[:40]}
        rows.append(row)

    # Rank: passes first, then by OOS expectancy (with a real trade count).
    order = {"pass": 0, "watchlist": 1, "reject": 2, "error": 3}
    rows.sort(key=lambda r: (order.get(r["status"], 9), -r.get("oos_exp", 0.0)))

    print("\n" + "-" * 78)
    print(f"{'Strategy':<28}{'Gate':<11}{'OOS_PF':>7}{'OOS_exp':>9}{'OOS_tr':>7}{'WR':>6}{'stab':>6}{'degr':>6}")
    print("-" * 78)
    for r in rows:
        wr = f"{r.get('oos_wr', 0.0) * 100:.0f}%"
        print(f"{r['name']:<28}{r['status']:<11}{r['oos_pf']:>7.2f}{r['oos_exp']:>9.4f}"
              f"{r['oos_tr']:>7}{wr:>6}{r['stab']:>6.2f}{r['deg']:>6.2f}")

    passes = [r for r in rows if r["status"] == "pass"]
    watch = [r for r in rows if r["status"] == "watchlist"]
    print("\n" + "*" * 78)
    if passes:
        b = passes[0]
        print(f"{symbol}: TRADEABLE → {b['name']}  (OOS PF {b['oos_pf']:.2f}, "
              f"{b['oos_tr']} OOS trades, stability {b['stab']:.2f})")
    elif watch:
        b = watch[0]
        print(f"{symbol}: NO CLEAN PASS. Best watchlist candidate: {b['name']} "
              f"(OOS PF {b['oos_pf']:.2f}, {b['oos_tr']} trades) — needs more validation.")
    else:
        print(f"{symbol}: NO EDGE. No strategy cleared the out-of-sample gates. DO NOT trade.")
    print("*" * 78)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--holdout", action="store_true",
                   help="single anchored 65/35 OOS split (fast) instead of k-fold")
    p.add_argument("--focus", action="store_true",
                   help="only test strategies that produce a real trade sample")
    a = p.parse_args()
    research(a.symbol, a.days, a.folds, a.holdout, a.focus)
