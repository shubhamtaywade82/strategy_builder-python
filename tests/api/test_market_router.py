from fastapi.testclient import TestClient
from strategy_api.main import create_app
from strategy_api.services import binance


def test_mark_price_endpoint(monkeypatch):
    async def fake(symbol):
        return {"symbol": symbol, "markPrice": 1.0, "indexPrice": 1.0,
                "fundingRate": 0.0, "nextFundingTime": 0, "time": 0}

    monkeypatch.setattr(binance, "mark_price", fake)
    client = TestClient(create_app())
    r = client.get("/api/market/mark-price?symbol=SOLUSDT")
    assert r.status_code == 200
    assert r.json()["symbol"] == "SOLUSDT"


def test_invalid_symbol_422():
    client = TestClient(create_app())
    r = client.get("/api/market/mark-price?symbol=bad!")
    assert r.status_code == 422
