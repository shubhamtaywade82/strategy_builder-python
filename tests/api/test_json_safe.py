import math

from strategy_api.services.research import _json_safe, _map_result


def test_json_safe_replaces_non_finite():
    out = _json_safe({"a": float("inf"), "b": float("-inf"), "c": float("nan"),
                      "d": [1.5, float("inf")], "e": "x", "f": 3})
    assert out["a"] == 1e9
    assert out["b"] == -1e9
    assert out["c"] == 0.0
    assert out["d"] == [1.5, 1e9]
    assert out["e"] == "x"
    assert out["f"] == 3


def test_mapped_profit_factor_infinity_is_finite():
    raw = {"symbol": "X", "results": {"1:1": {"long": {
        "strategies": [{"name": "s", "side": "long", "conditions": [],
                        "metrics": {"profit_factor": float("inf"), "win_rate": 1.0}}],
        "top_features": [], "walk_forward": {"folds": 0, "fold_results": []},
        "label_stats": {"total": 1, "long_wins": 1, "short_wins": 0, "no_trade": 0}},
        "short": {"error": "x"}}}}
    mapped = _json_safe(_map_result(raw, "X", ["1:1"], 10))
    pf = mapped["strategies"][0]["metrics"]["profitFactor"]
    assert math.isfinite(pf) and pf == 1e9
