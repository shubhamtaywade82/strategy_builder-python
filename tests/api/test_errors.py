from fastapi.testclient import TestClient
from strategy_api.main import create_app
from strategy_api.services import research as research_svc

def test_research_value_error_returns_400(monkeypatch):
    def boom(**kw): raise ValueError("Insufficient 1m data for X")
    monkeypatch.setattr(research_svc, "_run_blocking", boom)
    c = TestClient(create_app(), raise_server_exceptions=False)
    r = c.post("/api/research/run", json={"symbol": "XRPUSDT", "rrs": ["1:1"]})
    assert r.status_code == 400
    assert r.json()["error"]["message"].startswith("Insufficient")
