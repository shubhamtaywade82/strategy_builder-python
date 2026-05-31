from __future__ import annotations
"""In-process entry point for the dashboard's multi-RR grid search.
Returns the dict the frontend consumes — no file I/O, no subprocess.
"""
from typing import List, Optional
from ._ui_engine.data_fetcher import fetch_symbol_mtf
from ._ui_engine.grid_search import discover_for_rr, RR_CONFIGS
from ._ui_engine.features import build_mtf_features


def run_research(
    symbol: str,
    days: int = 60,
    leverage: float = 10.0,
    cost: float = 0.0009,
    horizon: int = 120,
    rrs: Optional[List[str]] = None,
) -> dict:
    rrs = rrs or ["2:1"]
    data = fetch_symbol_mtf(symbol, days=days)
    if "1m" not in data or len(data["1m"]) < 1000:
        raise ValueError(f"Insufficient 1m data for {symbol}")

    features = build_mtf_features(data)

    results = {}
    for rr in rrs:
        if rr not in RR_CONFIGS:
            continue
        try:
            results[rr] = discover_for_rr(features, data["1m"], rr, leverage, cost, horizon)
        except Exception as exc:  # one RR failing must not kill the run
            results[rr] = {"long": {"error": str(exc)}, "short": {"error": str(exc)}}
    return {"symbol": symbol, "results": results}
