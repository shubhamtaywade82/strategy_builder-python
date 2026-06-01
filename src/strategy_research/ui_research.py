from __future__ import annotations
"""In-process entry point for the dashboard's multi-RR grid search.
Returns the dict the frontend consumes — no file I/O, no subprocess.

Honesty contract:
- ``data_source`` is always present: "binance" for live klines, "synthetic" for
  the deterministic offline generator. Synthetic data is never passed off as
  real — the UI shows an explicit badge.
- ``player`` carries the REAL backtested 5-condition rule (out-of-sample folds,
  bootstrap p-value, random-entry baseline), not hardcoded placeholders.
"""
import logging
import os
from typing import List, Optional

from ._ui_engine.data_fetcher import fetch_symbol_mtf
from ._ui_engine.grid_search import discover_for_rr, RR_CONFIGS
from ._ui_engine.features import build_mtf_features
from ._ui_engine.rule_strategy import run_player
from ._ui_engine.synthetic import make_synthetic_mtf

log = logging.getLogger("ui_research")

_MIN_1M_BARS = 1000


def _allow_synthetic(flag: Optional[bool]) -> bool:
    if flag is not None:
        return flag
    return os.getenv("STRATEGY_RESEARCH_ALLOW_SYNTHETIC", "").lower() in ("1", "true", "yes")


def _load_data(symbol: str, days: int, allow_synthetic: bool):
    """Return (frames, data_source). Falls back to clearly-labelled synthetic
    data only when live klines are unavailable AND synthetic is allowed."""
    try:
        data = fetch_symbol_mtf(symbol, days=days)
        if "1m" in data and len(data["1m"]) >= _MIN_1M_BARS:
            return data, "binance"
        reason = f"only {len(data.get('1m', []))} 1m bars"
    except Exception as exc:  # network blocked / geo-restricted / rate limited
        reason = f"fetch failed: {exc}"

    if allow_synthetic:
        log.warning("Live data unavailable for %s (%s) — using deterministic "
                    "synthetic data (labelled data_source=synthetic).", symbol, reason)
        return make_synthetic_mtf(symbol, days=days), "synthetic"
    raise ValueError(f"Insufficient 1m data for {symbol} ({reason})")


def run_research(
    symbol: str,
    days: int = 60,
    leverage: float = 10.0,
    cost: float = 0.0009,
    horizon: int = 120,
    rrs: Optional[List[str]] = None,
    allow_synthetic: Optional[bool] = None,
) -> dict:
    rrs = rrs or ["2:1"]
    data, data_source = _load_data(symbol, days, _allow_synthetic(allow_synthetic))

    features = build_mtf_features(data)

    results = {}
    for rr in rrs:
        if rr not in RR_CONFIGS:
            continue
        try:
            results[rr] = discover_for_rr(features, data["1m"], rr, leverage, cost, horizon)
        except Exception as exc:  # one RR failing must not kill the run
            results[rr] = {"long": {"error": str(exc)}, "short": {"error": str(exc)}}

    # Real backtested 5-condition rule for the Strategy Player, at the primary RR.
    primary = RR_CONFIGS.get(rrs[0], RR_CONFIGS["2:1"])
    try:
        player = run_player(data, up_pct=primary["up_pct"], dn_pct=primary["dn_pct"],
                            leverage=leverage, cost=cost, horizon=horizon)
    except Exception as exc:
        log.exception("rule strategy (player) failed")
        player = {"error": str(exc)}

    return {"symbol": symbol, "data_source": data_source,
            "results": results, "player": player}
