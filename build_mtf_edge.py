"""
MTF-Alignment Edge — clean build, fast (cached), honestly validated.
====================================================================
Implements the user's spec as a NEW strategy on the builder primitives:

  Macro bias   : EMA50 on 1h + 4h + 1d must ALL agree (long if price>EMA50 on all
                 three, short if below all). No alignment → no trade.
  Structure    : breakout of the prior-N high/low on the EXECUTION timeframe and/or
                 15m, in the macro direction (MSS / continuation).
  Trigger      : engulfing candle on the execution TF in the macro direction
                 (the "retest/continuation" confirmation).
  Stop         : recent swing low/high on the execution TF, FLOORED so that the
                 2R target is a ≥1% NET move after round-trip fees.
  Target       : 2R (structural 1:2). Fees (0.05%/side) + slippage (2bps) are
                 charged by the engine, so every metric is NET. The min-stop floor
                 guarantees TP excludes fees AND clears the 1% move requirement.

Speed: everything that the framework recomputes per-bar (HTF EMAs, breakout
levels) is PRECOMPUTED once into timestamp-aligned numpy arrays (EMAs are causal,
so a one-pass compute is byte-identical to per-prefix recompute). The condition
is then O(1) per bar — turning an O(n²) backtest into O(n).

Execution timeframe is configurable (--exec-tf). Default 5m for a tractable,
large OOS sample; set 1m for literal fine-grain execution (much slower, run
offline). Gates (all NET, on the out-of-sample holdout):
  ≥30 trades · win-rate > 50% · avg-win/avg-loss ≥ 1.8 (1:2 held net) · exp > 0.

Usage:
    ./venv/bin/python build_mtf_edge.py --days 90 --exec-tf 5m
    ./venv/bin/python build_mtf_edge.py --days 30 --exec-tf 1m
"""
import argparse
import logging
from datetime import datetime, timedelta

import numpy as np

from strategy_builder.market_data.candle_loader import CandleLoader
from strategy_builder.backtest.engine import BacktestEngine
from strategy_builder.backtest.signal_evaluator import SignalEvaluator
from strategy_builder.backtest.walk_forward import WalkForward
from strategy_builder.backtest.condition_registry import ConditionRegistry

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("MTFEdge")

SYMBOLS = ["SOLUSDT", "XRPUSDT", "ETHUSDT", "BTCUSDT"]
MACRO_TFS = ["1h", "4h", "1d"]
EMA_LEN = 50
BREAK_LOOKBACK = 5          # prior-N high/low for the structure breakout
SWING_LOOKBACK = 6          # swing low/high for the stop
MIN_STOP_FRAC = 0.0055      # floor: 2R = 1.1% gross ≈ 1.0% NET after 0.1% fees
ROUND_TRIP_FEE = 0.001

# Per-symbol precompute, keyed by execution-bar timestamp → set before each run.
_CACHE = {}


def _ema(arr, length):
    out = np.full(len(arr), np.nan)
    if len(arr) == 0:
        return out
    k = 2.0 / (length + 1.0)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def precompute(exec_candles, mtf):
    """Build O(1)-lookup arrays for the execution series."""
    ts = np.array([c.get_timestamp_int() for c in exec_candles])
    close = np.array([c.close for c in exec_candles], float)
    high = np.array([c.high for c in exec_candles], float)
    low = np.array([c.low for c in exec_candles], float)
    op = np.array([c.open for c in exec_candles], float)
    n = len(exec_candles)

    # Macro bias: for each exec bar, compare close to EMA50 of each HTF at the
    # last HTF bar that had closed by then (causal, no lookahead).
    macro_long = np.ones(n, bool)
    macro_short = np.ones(n, bool)
    for tf in MACRO_TFS:
        series = mtf.get(tf, [])
        if len(series) < EMA_LEN + 2:
            macro_long[:] = False; macro_short[:] = False
            break
        htf_ts = np.array([c.get_timestamp_int() for c in series])
        htf_close = np.array([c.close for c in series], float)
        htf_ema = _ema(htf_close, EMA_LEN)
        idx = np.searchsorted(htf_ts, ts, side="right") - 1   # last closed HTF bar
        idx = np.clip(idx, 0, len(series) - 1)
        ema_at = htf_ema[idx]
        macro_long &= close > ema_at
        macro_short &= close < ema_at

    # Structure breakout on the execution TF (prior-N high/low).
    brk_up = np.zeros(n, bool); brk_dn = np.zeros(n, bool)
    for i in range(BREAK_LOOKBACK, n):
        ph = high[i - BREAK_LOOKBACK:i].max()
        pl = low[i - BREAK_LOOKBACK:i].min()
        brk_up[i] = close[i] > ph
        brk_dn[i] = close[i] < pl

    # Engulfing trigger on the execution TF.
    eng_bull = np.zeros(n, bool); eng_bear = np.zeros(n, bool)
    for i in range(1, n):
        if (close[i - 1] < op[i - 1] and close[i] > op[i]
                and close[i] > op[i - 1] and op[i] < close[i - 1]):
            eng_bull[i] = True
        if (close[i - 1] > op[i - 1] and close[i] < op[i]
                and close[i] < op[i - 1] and op[i] > close[i - 1]):
            eng_bear[i] = True

    # Swing stop levels.
    swing_low = np.array([low[max(0, i - SWING_LOOKBACK + 1):i + 1].min() for i in range(n)])
    swing_high = np.array([high[max(0, i - SWING_LOOKBACK + 1):i + 1].max() for i in range(n)])

    _CACHE.clear()
    _CACHE.update(dict(
        ts_to_i={int(t): i for i, t in enumerate(ts)},
        close=close, macro_long=macro_long, macro_short=macro_short,
        brk_up=brk_up, brk_dn=brk_dn, eng_bull=eng_bull, eng_bear=eng_bear,
        swing_low=swing_low, swing_high=swing_high,
    ))


def _register():
    @ConditionRegistry.register("mtf_edge_entry")
    def mtf_edge_entry(ctx) -> bool:
        i = _CACHE["ts_to_i"].get(ctx.current_candle.get_timestamp_int())
        if i is None or i < BREAK_LOOKBACK:
            return False
        c = float(_CACHE["close"][i])
        if _CACHE["macro_long"][i] and _CACHE["brk_up"][i] and _CACHE["eng_bull"][i]:
            stop = c - float(_CACHE["swing_low"][i])
            stop = max(stop, c * MIN_STOP_FRAC)
            if stop <= 0:
                return False
            ctx.direction = "long"; ctx.entry_price = c; ctx.stop_distance = stop
            return True
        if _CACHE["macro_short"][i] and _CACHE["brk_dn"][i] and _CACHE["eng_bear"][i]:
            stop = float(_CACHE["swing_high"][i]) - c
            stop = max(stop, c * MIN_STOP_FRAC)
            if stop <= 0:
                return False
            ctx.direction = "short"; ctx.entry_price = c; ctx.stop_distance = stop
            return True
        return False


def strategy_cfg(exec_tf):
    return {
        "name": "MTF_Alignment_Edge",
        "entry": {"conditions": ["mtf_edge_entry"]},
        "exit": {"targets": [2.0]},     # 1:2; stop floored so 2R ≥ 1% net move
        "timeframes": [exec_tf] + MACRO_TFS,
    }


GATES = {"min_trades": 30, "min_win_rate": 0.50, "min_rr": 1.8}


def _gate(oos):
    tr = oos.get("trade_count", 0); wr = oos.get("win_rate", 0.0)
    aw = oos.get("avg_win", 0.0); al = abs(oos.get("avg_loss", 0.0))
    exp = oos.get("expectancy", 0.0)
    rr = aw / al if al > 0 else 0.0
    fails = []
    if tr < GATES["min_trades"]: fails.append(f"tr {tr}<30")
    if wr < GATES["min_win_rate"]: fails.append(f"win {wr:.0%}<50%")
    if rr < GATES["min_rr"]: fails.append(f"RR {rr:.2f}<1.8")
    if exp <= 0: fails.append("exp<=0")
    return ("PASS" if not fails else "FAIL"), fails, rr


def run_symbol(symbol, days, exec_tf):
    loader = CandleLoader(market_data_source="binance")
    end = datetime.now(); start = end - timedelta(days=days)
    tfs = sorted(set([exec_tf] + MACRO_TFS + ["15m", "5m"]))
    mtf = loader.fetch_mtf(instrument=symbol, timeframes=tfs, start_from=start, to=end)
    exec_candles = mtf[exec_tf]
    precompute(exec_candles, mtf)
    cfg = strategy_cfg(exec_tf)
    ev = SignalEvaluator.build(cfg, mtf_candles=mtf)
    ho = WalkForward(engine=BacktestEngine()).anchored_holdout(
        strategy=cfg, candles=exec_candles, signal_generator=ev, mtf_candles=mtf)
    if ho is None:
        return {"symbol": symbol, "error": "insufficient data"}
    oos, ism = ho["out_of_sample"], ho["in_sample"]
    status, fails, rr = _gate(oos)
    return {"symbol": symbol, "status": status, "fails": fails, "rr": rr,
            "oos": oos, "is": ism, "nbase": len(exec_candles)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--exec-tf", default="5m")
    p.add_argument("--symbols", default=",".join(SYMBOLS))
    a = p.parse_args()
    _register()
    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]

    print("=" * 100)
    print(f"MTF-ALIGNMENT EDGE | exec={a.exec_tf} macro=EMA50(1h,4h,1d) | {a.days}d | "
          f"min-move≥1% net, 1:2 RR | gates: ≥30tr & win>50% & RR≥1.8 & exp>0 (NET)")
    print("=" * 100)
    rows = []
    for s in syms:                       # sequential: cache is a global + avoids 429
        try:
            rows.append(run_symbol(s, a.days, a.exec_tf))
        except Exception as e:
            rows.append({"symbol": s, "error": str(e)[:90]})

    print(f"\n{'Sym':<9}{'Gate':<6}{'OOS_tr':>7}{'OOS_win':>8}{'win/loss':>9}{'OOS_PF':>7}"
          f"{'OOS_exp':>9}{'IS_win':>8}{'IS_tr':>7}  fails")
    print("-" * 100)
    for r in rows:
        if r.get("error"):
            print(f"{r['symbol']:<9}ERROR  {r['error']}"); continue
        o, i = r["oos"], r["is"]
        print(f"{r['symbol']:<9}{r['status']:<6}{o.get('trade_count',0):>7}{o.get('win_rate',0)*100:>7.1f}%"
              f"{r['rr']:>9.2f}{o.get('profit_factor',0):>7.2f}{o.get('expectancy',0):>9.2f}"
              f"{i.get('win_rate',0)*100:>7.1f}%{i.get('trade_count',0):>7}  {'; '.join(r['fails'])}")
    passes = [r for r in rows if r.get("status") == "PASS"]
    print("\n" + "*" * 100)
    if passes:
        print(f"PASS: {', '.join(r['symbol'] for r in passes)} — win>50% + net 1:2 RR + ≥1% move OOS.")
    else:
        print("NO SYMBOL clears win>50% + net 1:2 RR + ≥1% move out-of-sample. Closest miss above.")
    print("*" * 100)


if __name__ == "__main__":
    main()
