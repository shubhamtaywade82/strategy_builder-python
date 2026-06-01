import httpx
import pytest
from strategy_api.services import ollama


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_chat_completion(monkeypatch):
    async def fake_post(self, url, **kw):
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "ok"},
                  "done": True, "total_duration": 1},
            request=httpx.Request("POST", url),
        )
    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    out = await ollama.chat_completion([{"role": "user", "content": "hi"}], "m")
    assert out["message"]["content"] == "ok"


@pytest.mark.anyio
async def test_health_false_on_error(monkeypatch):
    async def boom(self, url, **kw):
        raise httpx.ConnectError("no")
    monkeypatch.setattr(httpx.AsyncClient, "get", boom)
    assert await ollama.health_check() is False
