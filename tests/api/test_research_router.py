from fastapi.testclient import TestClient
from strategy_api.main import create_app
from strategy_api.services import research as research_svc

RAW = {"symbol": "SOLUSDT", "results": {"2:1": {"long": {
    "strategies": [{"name": "long_threshold_0.6", "description": "d", "side": "long",
                    "conditions": [{"feature": "f", "operator": ">=", "threshold": 1.0, "importance": 0.2}],
                    "metrics": {"trade_count": 30, "win_rate": 0.6, "expectancy": 0.01},
                    "is_viable": True}],
    "top_features": [{"feature": "f", "direction": "high", "threshold": 1.0, "importance": 0.2,
                      "win_rate_above": 0.6, "win_rate_below": 0.4, "shap_value": 0.1}],
    "walk_forward": {"folds": 3, "avg_test_auc": 0.6, "min_test_auc": 0.55, "stability": 0.9,
                     "degradation": 0.05, "is_valid": True, "fold_results": []},
    "label_stats": {"total": 100, "long_wins": 40, "short_wins": 30, "no_trade": 30}},
    "short": {"error": "insufficient"}}}}

def test_run_maps_to_research_result(monkeypatch):
    def fake_run(**kw): return RAW
    monkeypatch.setattr(research_svc, "_run_blocking", fake_run)
    c = TestClient(create_app())
    r = c.post("/api/research/run", json={"symbol": "SOLUSDT", "rrs": ["2:1"], "leverage": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "SOLUSDT"
    assert body["strategies"][0]["metrics"]["winRate"] == 0.6
    assert body["walkForward"]["avgTestAuc"] == 0.6
    assert body["topFeatures"][0]["feature"] == "f"
    assert body["labels"]["total"] == 100
