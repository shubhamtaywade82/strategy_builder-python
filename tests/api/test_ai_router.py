from fastapi.testclient import TestClient
from strategy_api.main import create_app
from strategy_api.services import ollama
from strategy_api.db.models import Base
from strategy_api.db.session import get_session, make_engine, make_session_factory


def _client_with_tmp_db(tmp_path):
    """Build a TestClient whose get_session dependency uses a fresh tmp SQLite DB,
    so tests never touch the real sqlite.db and are repeatable."""
    app = create_app()
    engine = make_engine(f"sqlite:///{tmp_path/'ai.db'}")
    Base.metadata.create_all(engine)
    SessionFactory = make_session_factory(engine)

    def _override():
        s = SessionFactory()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_session] = _override
    return TestClient(app)


def test_health(monkeypatch):
    async def ok():
        return True

    async def models():
        return ["m1"]

    monkeypatch.setattr(ollama, "health_check", ok)
    monkeypatch.setattr(ollama, "list_models", models)
    c = TestClient(create_app())
    r = c.get("/api/ai/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_chat_persists(monkeypatch, tmp_path):
    async def fake_chat(messages, model=None, temperature=0.7):
        return {"message": {"content": "hello"}, "total_duration": 1,
                "prompt_eval_count": 2, "eval_count": 3}

    monkeypatch.setattr(ollama, "chat_completion", fake_chat)
    c = _client_with_tmp_db(tmp_path)
    r = c.post("/api/ai/chat", json={"messages": [{"role": "user", "content": "hi"}],
                                     "model": "m", "sessionId": "s1"})
    assert r.status_code == 200 and r.json()["content"] == "hello"
    hist = c.get("/api/ai/history?sessionId=s1").json()
    assert len(hist) == 2
