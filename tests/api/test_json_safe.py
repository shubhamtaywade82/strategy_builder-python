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


def test_json_safe_handles_numpy_scalars():
    np = __import__("numpy")
    out = _json_safe({"a": np.float32("inf"), "b": np.float64(2.5),
                      "c": np.int64(7), "d": [np.float32(1.0)]})
    assert out["a"] == 1e9
    assert out["b"] == 2.5 and isinstance(out["b"], float)
    assert out["c"] == 7 and isinstance(out["c"], int)
    assert out["d"] == [1.0]


def test_map_result_uses_request_horizon():
    raw = {"symbol": "X", "results": {}}
    mapped = _map_result(raw, "X", ["2:1"], 10, horizon=240)
    assert mapped["config"]["horizon"] == 240


def test_results_map_feeds_validation_and_overview_panels():
    # Shape ValidationPanel + ResultsOverview read: results.results[rr].{strategies,
    # walk_forward (mixed snake + camel), shuffle_test}.
    raw = {"symbol": "X", "results": {"1:1": {
        "long": {
            "strategies": [{"name": "s", "side": "long", "conditions": [],
                            "metrics": {"win_rate": 0.6, "profit_factor": 1.5,
                                        "expectancy": 0.01, "trade_count": 40},
                            "is_viable": True}],
            "top_features": [],
            "walk_forward": {"folds": 2, "avg_test_auc": 0.6, "min_test_auc": 0.55,
                             "stability": 0.8, "degradation": -0.02, "is_valid": True,
                             "fold_results": [{"fold": 1, "train_auc": 0.7, "test_auc": 0.6,
                                               "test_precision": 0.5, "test_recall": 0.4}]},
            "shuffle_test": {"real_auc": 0.6, "shuffled_auc_mean": 0.5, "p_value": 0.01,
                             "is_significant": True},
            "label_stats": {"total": 10, "long_wins": 4, "short_wins": 3, "no_trade": 3}},
        "short": {"error": "x"}}}}
    out = _json_safe(_map_result(raw, "X", ["1:1"], 10))
    rr = out["results"]["1:1"]
    # ResultsOverview rrData reads camel strategy metrics
    assert rr["strategies"][0]["isViable"] is True
    assert rr["strategies"][0]["metrics"]["winRate"] == 0.6
    # ValidationPanel reads mixed keys off walk_forward
    wf = rr["walk_forward"]
    assert wf["isValid"] is True              # camel alias
    assert wf["avg_test_auc"] == 0.6          # engine snake
    assert wf["foldResults"][0]["test_auc"] == 0.6   # camel alias, snake items
    assert rr["shuffle_test"]["real_auc"] == 0.6


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
