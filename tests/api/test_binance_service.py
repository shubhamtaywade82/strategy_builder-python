import httpx
import pytest
from strategy_api.services import binance


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mark_price_parses(monkeypatch):
    payload = {"symbol": "SOLUSDT", "markPrice": "150.5", "indexPrice": "150.4",
               "lastFundingRate": "0.0001", "nextFundingTime": 123, "time": 456}

    async def fake_get(self, url, **kw):
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    out = await binance.mark_price("SOLUSDT")
    assert out["markPrice"] == 150.5
    assert out["fundingRate"] == 0.0001
