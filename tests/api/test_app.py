from fastapi.testclient import TestClient
from strategy_api.main import create_app

def test_ping():
    client = TestClient(create_app())
    r = client.get("/api/ping")
    assert r.status_code == 200
    assert r.json()["ok"] is True
