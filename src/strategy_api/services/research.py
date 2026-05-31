from __future__ import annotations
import math
from typing import Any, Dict, List

import anyio
from strategy_research import ui_research

RR_CFGS = {
    "3:1": {"upPct": 0.015, "dnPct": 0.005}, "2:1": {"upPct": 0.010, "dnPct": 0.005},
    "1:1": {"upPct": 0.010, "dnPct": 0.010}, "1:2": {"upPct": 0.005, "dnPct": 0.010},
    "1:3": {"upPct": 0.005, "dnPct": 0.015},
}

# Engine metrics can be non-finite (e.g. profit_factor is +inf when a side has
# zero losing trades). JSON has no Infinity/NaN literal, so the response would be
# invalid JSON the browser cannot parse. Clamp to finite sentinels at the API
# boundary; finite values pass through untouched.
_INF_SENTINEL = 1e9


def _json_safe(value: Any) -> Any:
    """Recursively coerce engine output into JSON-serializable, finite values.

    Engine metrics may be numpy scalars (np.float32/np.int64) or non-finite
    floats (profit_factor is +inf with zero losing trades). JSON has no
    Infinity/NaN literal and the default encoder rejects numpy scalars, so both
    must be normalized at the API boundary. Finite native values pass through.
    """
    if isinstance(value, bool):
        return value
    if hasattr(value, "item") and not isinstance(value, (dict, list)):
        # numpy scalar -> native Python scalar
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float):
        if math.isnan(value):
            return 0.0
        if math.isinf(value):
            return _INF_SENTINEL if value > 0 else -_INF_SENTINEL
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _run_blocking(*, symbol, days, leverage, horizon, rrs):
    return ui_research.run_research(symbol=symbol, days=days, leverage=leverage,
                                    horizon=horizon, rrs=rrs)


async def run(symbol, days, leverage, horizon, rrs) -> Dict:
    raw = await anyio.to_thread.run_sync(
        lambda: _run_blocking(symbol=symbol, days=days, leverage=leverage,
                              horizon=horizon, rrs=rrs))
    return _json_safe(_map_result(raw, symbol, rrs, leverage, horizon))


def _map_strategy(strat: Dict, rr: str, side: str) -> Dict:
    m = strat.get("metrics", {})
    return {
        "id": f"{rr}_{strat.get('name', side)}",
        "name": strat.get("name", f"{rr} {side}"),
        "description": strat.get("description", ""),
        "side": strat.get("side", side),
        "conditions": [{"feature": c.get("feature"), "operator": c.get("operator"),
                        "threshold": c.get("threshold"), "importance": c.get("importance", 0)}
                       for c in strat.get("conditions", [])],
        "metrics": {
            "tradeCount": m.get("trade_count", 0), "winCount": m.get("win_count", 0),
            "lossCount": m.get("loss_count", 0), "winRate": m.get("win_rate", 0),
            "profitFactor": m.get("profit_factor", 0), "expectancy": m.get("expectancy", 0),
            "netPnl": m.get("net_pnl", 0), "avgWin": m.get("avg_win", 0),
            "avgLoss": m.get("avg_loss", 0), "maxDrawdown": m.get("max_drawdown", 0),
            "sharpe": m.get("sharpe", 0), "avgBarsHeld": m.get("avg_bars_held", 0),
            "targetHitRate": m.get("target_hit_rate", 0),
            "stopHitRate": m.get("stop_hit_rate", 0)},
        "isViable": strat.get("is_viable", False)}


def _camel_walk_forward(wf: Dict) -> Dict:
    return {
        "folds": wf.get("folds", 0), "avgTestAuc": wf.get("avg_test_auc", 0.5),
        "minTestAuc": wf.get("min_test_auc", 0.5), "stability": wf.get("stability", 0),
        "degradation": wf.get("degradation", 0), "isValid": wf.get("is_valid", False),
        "foldResults": [{"fold": f.get("fold"), "trainAuc": f.get("train_auc", 0.5),
                         "testAuc": f.get("test_auc", 0.5),
                         "testPrecision": f.get("test_precision", 0),
                         "testRecall": f.get("test_recall", 0),
                         "nTrain": f.get("n_train", 0), "nTest": f.get("n_test", 0)}
                        for f in wf.get("fold_results", [])]}


def _map_result(raw: Dict, symbol: str, rrs: List[str], leverage: float,
                horizon: int = 120) -> Dict:
    strategies, top_features = [], []
    walk_forward = {"folds": 0, "avgTestAuc": 0.5, "minTestAuc": 0.5, "stability": 0,
                    "degradation": 0, "isValid": False, "foldResults": []}
    labels = {"longWins": 0, "shortWins": 0, "noTrade": 0, "total": 0}
    # Per-RR view consumed by ResultsOverview (RR comparison chart) and ValidationPanel.
    # walk_forward here carries BOTH the engine's snake_case keys and camel aliases
    # (foldResults/isValid) because those components read a mixed-case shape.
    results_map: Dict = {}

    for rr in rrs:
        rr_data = (raw.get("results") or {}).get(rr)
        if not rr_data:
            continue
        rr_strategies: List = []
        rr_entry: Dict = {"strategies": rr_strategies}
        for side in ("long", "short"):
            side_data = rr_data.get(side)
            if not side_data or side_data.get("error"):
                continue
            for strat in side_data.get("strategies", []):
                mapped = _map_strategy(strat, rr, side)
                strategies.append(mapped)
                rr_strategies.append(mapped)
            wf = side_data.get("walk_forward")
            if wf:
                if walk_forward["folds"] == 0:
                    walk_forward = _camel_walk_forward(wf)
                # First valid side supplies this RR's walk-forward + shuffle view.
                if "walk_forward" not in rr_entry:
                    rr_entry["walk_forward"] = dict(wf, isValid=wf.get("is_valid", False),
                                                    foldResults=wf.get("fold_results", []))
                    rr_entry["shuffle_test"] = side_data.get("shuffle_test")
            tf = side_data.get("top_features")
            if tf and not top_features:
                top_features = [{"feature": f.get("feature", f.get("name", "")),
                                 "direction": f.get("direction", "high"),
                                 "threshold": f.get("threshold", f.get("median_val", 0)),
                                 "importance": f.get("importance", 0),
                                 "winRateAbove": f.get("win_rate_above", 0),
                                 "winRateBelow": f.get("win_rate_below", 0),
                                 "shapValue": f.get("shap_value", 0)} for f in tf]
            ls = side_data.get("label_stats")
            if ls and labels["total"] == 0:
                labels = {"total": ls.get("total", 0), "longWins": ls.get("long_wins", 0),
                          "shortWins": ls.get("short_wins", 0), "noTrade": ls.get("no_trade", 0)}
        results_map[rr] = rr_entry

    primary = rrs[0] if rrs else "2:1"
    cfg = RR_CFGS.get(primary, {"upPct": 0.01, "dnPct": 0.005})
    return {"symbol": symbol,
            "config": {"symbol": symbol, "rr": primary, "upPct": cfg["upPct"],
                       "dnPct": cfg["dnPct"], "leverage": leverage, "horizon": horizon, "side": "both"},
            "strategies": strategies, "walkForward": walk_forward,
            "topFeatures": top_features, "labels": labels, "results": results_map}
