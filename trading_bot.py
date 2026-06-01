"""
Live 5-Condition Rule Bot (signal + risk only, no order execution)
==================================================================
Evaluates the EXACT same rule that rule_strategy.py backtests — it imports the
shared ``build_signal_mask`` so the live signal can never drift from the
backtested one — and sizes/gates each signal through the shared RiskGuard.

This is deliberately execution-free: it logs the signal, the size the guard
would use, and the risk verdict. Wire your exchange client into ``_on_signal``
to go live. The same leakage-safe trailing-window conditions used in research
apply here, so there is no "looks different live" surprise.

Run:  python trading_bot.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from strategy_research._ui_engine.data_fetcher import fetch_klines, INTERVAL_MS
from strategy_research._ui_engine.rule_strategy import build_signal_mask, RuleConfig
from risk_guard import RiskGuard

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("TradingBot")

SYMBOL = "SOLUSDT"
LEVERAGE = 10.0
RISK_PCT = 0.02
STOP_LOSS_PCT = 0.005
TAKE_PROFIT_PCT = 0.010
POLL_SECONDS = 60

# Lookback windows per timeframe (enough warmup: 4H EMA200 needs ~200 bars).
LOOKBACK_BARS = {"1m": 1440, "15m": 200, "1h": 200, "4h": 320}


def _fetch_frames(symbol: str) -> dict:
    end = int(time.time() * 1000)
    frames = {}
    for tf, bars in LOOKBACK_BARS.items():
        start = end - bars * INTERVAL_MS[tf]
        frames[tf] = fetch_klines(symbol, tf, start, end)
    return frames


def _on_signal(side: str, price: float, size: dict) -> None:
    """Hook point — replace with real order placement. Signal-only by default."""
    stop = price * (1 - STOP_LOSS_PCT) if side == "long" else price * (1 + STOP_LOSS_PCT)
    tgt = price * (1 + TAKE_PROFIT_PCT) if side == "long" else price * (1 - TAKE_PROFIT_PCT)
    emoji = "🟢" if side == "long" else "🔴"
    log.warning("%s %s SIGNAL @ %.4f | stop %.4f | target %.4f", emoji, side.upper(), price, stop, tgt)
    log.warning("   size: notional=%.2f margin=%.2f qty=%.4f lev=%sx risk=%.2f%%",
                size["notional"], size["margin_required"], size["qty"],
                size["leverage"], size["risk_pct"] * 100)


def run_bot():
    log.info("=" * 60)
    log.info("LIVE 5-CONDITION RULE BOT | %s (signal-only)", SYMBOL)
    log.info("Target +%.1f%% / Stop -%.1f%% | risk %.1f%% | base lev %gx",
             TAKE_PROFIT_PCT * 100, STOP_LOSS_PCT * 100, RISK_PCT * 100, LEVERAGE)
    log.info("Rule + sizing shared with the backtester (no drift).")
    log.info("=" * 60)

    cfg = RuleConfig()
    guard = RiskGuard(account=100.0, max_risk_pct=RISK_PCT, base_leverage=LEVERAGE)

    while True:
        try:
            frames = _fetch_frames(SYMBOL)
            if frames["1m"].empty or frames["4h"].empty:
                log.warning("no data (geo-block / rate limit?), retrying...")
                time.sleep(5)
                continue

            mask = build_signal_mask(frames, cfg)
            last = mask.iloc[-1]
            price = float(last["close"])
            bar_idx = len(mask) - 1
            ts = last["open_time"]

            sig_long = bool(last["sig_long"])
            sig_short = bool(last["sig_short"])
            log.info("[%s] %.4f | long=%s short=%s", ts, price, sig_long, sig_short)

            if sig_long or sig_short:
                ok, reason = guard.can_trade(bar_idx)
                if not ok:
                    log.info("   signal suppressed by risk guard: %s", reason)
                else:
                    side = "long" if sig_long else "short"
                    stop = price * (1 - STOP_LOSS_PCT) if side == "long" else price * (1 + STOP_LOSS_PCT)
                    size = guard.position_size(price, stop, bar_idx)
                    _on_signal(side, price, size)
                    guard.mark_entry(bar_idx)

            sleep_time = max(5, POLL_SECONDS - (time.time() % POLL_SECONDS))
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            log.info("stopped.")
            break
        except Exception as e:  # keep the loop alive on transient errors
            log.error("loop error: %s", e)
            time.sleep(10)


if __name__ == "__main__":
    run_bot()
