from __future__ import annotations
"""In-process entry point for the dashboard's multi-RR grid search.
Returns the dict the frontend consumes — no file I/O, no subprocess.
"""
import logging
from typing import List, Optional
from ._ui_engine.data_fetcher import fetch_symbol_mtf
from ._ui_engine.grid_search import discover_for_rr, RR_CONFIGS
from ._ui_engine.features import build_mtf_features
from ._ui_engine.rule_strategy import backtest_rule

log = logging.getLogger("ui_research")


def _player_block(symbol: str, data: dict, days: int, rr: str,
                  cost: float, leverage: float, horizon: int) -> dict:
    """Honest 5-condition rule backtest (momentum + reversion), robustness-gated.

    Replaces the frontend's hardcoded StrategyPlayer mock. 4H EMA200 needs ~33d
    warmup, so refetch 4H with extra history; early bars are pre-window context.
    """
    frames = {tf: data[tf] for tf in ("1m", "15m", "1h") if tf in data}
    warm = fetch_symbol_mtf(symbol, days=days + 40, timeframes=["4h"])
    frames["4h"] = warm.get("4h", data.get("4h"))
    if any(tf not in frames or frames[tf] is None or len(frames[tf]) == 0
           for tf in ("1m", "15m", "1h", "4h")):
        return {"error": "insufficient MTF data for rule strategy"}
    cfg = RR_CONFIGS.get(rr, RR_CONFIGS["2:1"])
    # The rule enters with a LIMIT order at the FVG (maker), so the correct cost
    # model is the maker round-trip (~0.04%), not taker. Using taker here would
    # understate the rule's real edge. Capped at the run's cost so a caller can
    # only make it more conservative, never cheaper than reality.
    maker_cost = min(0.0004, cost)
    return backtest_rule(frames, {"up_pct": cfg["up_pct"], "dn_pct": cfg["dn_pct"],
                                  "label": cfg.get("label", rr)},
                         cost=maker_cost, leverage=leverage, horizon=horizon)


def _fetch_and_build(symbol: str, days: int):
    """Fetch MTF data and build features. Shared by sync and streaming paths."""
    data = fetch_symbol_mtf(symbol, days=days)
    if "1m" not in data or len(data["1m"]) < 1000:
        raise ValueError(f"Insufficient 1m data for {symbol}")
    features = build_mtf_features(data)
    return data, features


def run_research(
    symbol: str,
    days: int = 60,
    leverage: float = 10.0,
    cost: float = 0.0009,
    horizon: int = 120,
    rrs: Optional[List[str]] = None,
) -> dict:
    rrs = rrs or ["2:1"]
    data, features = _fetch_and_build(symbol, days)

    results = {}
    for rr in rrs:
        if rr not in RR_CONFIGS:
            continue
        try:
            results[rr] = discover_for_rr(features, data["1m"], rr, leverage, cost, horizon)
        except Exception as exc:  # one RR failing must not kill the run
            results[rr] = {"long": {"error": str(exc)}, "short": {"error": str(exc)}}

    try:
        player = _player_block(symbol, data, days, rrs[0], cost, leverage, horizon)
    except Exception as exc:  # rule strategy failing must not kill the run
        log.warning("rule strategy failed: %s", exc)
        player = {"error": str(exc)}

    return {"symbol": symbol, "results": results, "player": player}


def run_research_streaming(
    symbol: str,
    days: int = 60,
    leverage: float = 10.0,
    cost: float = 0.0009,
    horizon: int = 120,
    rrs: Optional[List[str]] = None,
):
    """Generator that yields partial results after each RR, then the final result.

    Yields dicts with keys: type ('partial' | 'complete'), symbol, results, player.
    """
    rrs = rrs or ["2:1"]
    data, features = _fetch_and_build(symbol, days)

    results = {}
    for rr in rrs:
        if rr not in RR_CONFIGS:
            continue
        try:
            results[rr] = discover_for_rr(features, data["1m"], rr, leverage, cost, horizon)
        except Exception as exc:
            results[rr] = {"long": {"error": str(exc)}, "short": {"error": str(exc)}}
        yield {"type": "partial", "symbol": symbol, "results": dict(results), "player": {}}

    try:
        player = _player_block(symbol, data, days, rrs[0], cost, leverage, horizon)
    except Exception as exc:
        log.warning("rule strategy failed: %s", exc)
        player = {"error": str(exc)}

    yield {"type": "complete", "symbol": symbol, "results": results, "player": player}
