# Unified Backend + Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Node/Hono/tRPC backend and the three duplicated Python engines with a single Python (FastAPI) backend that imports one unified quant core in-process and serves the existing React frontend (rewired tRPC → REST), losing no functionality.

**Architecture:** A new `src/strategy_api/` FastAPI package exposes REST endpoints that mirror every current tRPC procedure 1:1. Research runs in-process via a new canonical `strategy_research.ui_research` module (absorbed from `app/public/engine`) on a threadpool. SQLite (3 tables) is accessed via SQLAlchemy against the existing `sqlite.db`. The React app keeps all components/visuals; only its data layer (`providers/trpc.tsx` + 4 call sites) is swapped for a typed fetch client + React Query hooks. `app/` is renamed `frontend/`; Node/Drizzle code and duplicate engines are deleted.

**Tech Stack:** Python 3.8+, FastAPI, uvicorn, httpx, SQLAlchemy, pydantic-settings, pandas, numpy, xgboost, shap, pytest. React 19, Vite, @tanstack/react-query, zod, TypeScript.

---

## Reference: current state (read before starting)

- Engine the Node backend spawns: `app/public/engine/main.py` → `data_fetcher.fetch_symbol_mtf(symbol, days)` then `grid_search.run_grid_search(data, symbol, leverage, cost, horizon, output)` which writes JSON `{ results: { "<rr>": { long: {...}, short: {...} } } }`.
- Node `mapResult` (in `app/api/research.ts`) reads `results[rr][side].{strategies[].{name,description,side,conditions[],metrics{},is_viable}, walk_forward{folds,avg_test_auc,min_test_auc,stability,degradation,is_valid,fold_results[]}, top_features[]{feature,direction,threshold,importance,win_rate_above,win_rate_below,shap_value}, label_stats{total,long_wins,short_wins,no_trade}}` and produces the `ResearchResult` shape in `frontend/src/types/index.ts` (camelCase). **This mapping moves into the Python research router.**
- tRPC procedures (exhaustive): `ping`; `research.run`; `ai.{health,chat,history,generateStrategy,marketAnalysis,saveSession,sessions,insights,clearHistory}`; `market.{markPrice,ticker24h,klines,topSymbols}`.
- Frontend tRPC call sites (exhaustive): `App.tsx` → `research.run`; `AIChat.tsx` → `ai.health`, `ai.chat`; `AIStrategyInsights.tsx` → `ai.generateStrategy`. Provider: `src/providers/trpc.tsx`, mounted in `src/main.tsx`.
- **Env discrepancy to fix:** `app/api/lib/ollama.ts` reads `OLLAMA_URL`/`OLLAMA_MODEL`, but `.env` actually defines `OLLAMA_BASE_URL`/`OLLAMA_AGENT_MODEL`. The Python config MUST read `OLLAMA_BASE_URL` and `OLLAMA_AGENT_MODEL`.
- DB tables (Drizzle `app/db/schema.ts`): `chat_messages(id,session_id,role,content,model,created_at)`, `research_sessions(id,session_id UNIQUE,symbol,rr_config,leverage,days,status,result_json,best_strategy,win_rate,profit_factor,expectancy,created_at,updated_at)`, `strategy_insights(id,session_id,strategy_name,insight_type,content,model,created_at)`. Timestamps are stored as **integer epoch** (Drizzle `mode:'timestamp'`).

---

## Phase 0 — Backend scaffold

### Task 0.1: Add backend dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add deps to `[project].dependencies`**

Add these lines inside the existing `dependencies = [ ... ]` list:

```toml
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
    "httpx>=0.27.0",
    "sqlalchemy>=2.0.0",
    "pydantic-settings>=2.0.0",
    "anyio>=4.0.0",
```

- [ ] **Step 2: Register the package + console script**

In `[tool.hatch.build.targets.wheel]` change packages to include the API:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/strategy_builder", "src/strategy_research", "src/strategy_api"]
```

In `[project.scripts]` add:

```toml
strategy-api = "strategy_api.main:run"
```

- [ ] **Step 3: Install**

Run: `venv/bin/pip install -e ".[dev]"`
Expected: installs fastapi, uvicorn, httpx, sqlalchemy, pydantic-settings without error.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add FastAPI backend dependencies and strategy_api package"
```

### Task 0.2: App config from .env

**Files:**
- Create: `src/strategy_api/__init__.py` (empty)
- Create: `src/strategy_api/config.py`
- Test: `tests/api/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_config.py
import os
from strategy_api.config import Settings

def test_settings_reads_ollama_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example:11434")
    monkeypatch.setenv("OLLAMA_AGENT_MODEL", "test-model")
    s = Settings()
    assert s.ollama_base_url == "http://example:11434"
    assert s.ollama_agent_model == "test-model"

def test_settings_defaults():
    s = Settings()
    assert s.sqlite_path.endswith("sqlite.db")
    assert s.binance_fapi == "https://fapi.binance.com"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/api/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: strategy_api.config`

- [ ] **Step 3: Implement config**

```python
# src/strategy_api/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_REPO_ROOT / ".env"), extra="ignore")

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_agent_model: str = "qwen3.5:4b"
    binance_fapi: str = "https://fapi.binance.com"
    sqlite_path: str = str(_REPO_ROOT / "sqlite.db")
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    frontend_dist: str = str(_REPO_ROOT / "frontend" / "dist")

settings = Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/api/test_config.py -v`
Expected: PASS (create empty `tests/api/__init__.py` if needed for discovery).

- [ ] **Step 5: Commit**

```bash
git add src/strategy_api/__init__.py src/strategy_api/config.py tests/api/
git commit -m "feat(api): add settings loaded from .env"
```

### Task 0.3: FastAPI app factory + /api/ping

**Files:**
- Create: `src/strategy_api/main.py`
- Test: `tests/api/test_app.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_app.py
from fastapi.testclient import TestClient
from strategy_api.main import create_app

def test_ping():
    client = TestClient(create_app())
    r = client.get("/api/ping")
    assert r.status_code == 200
    assert r.json()["ok"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/api/test_app.py -v`
Expected: FAIL — `ModuleNotFoundError: strategy_api.main`

- [ ] **Step 3: Implement app factory**

```python
# src/strategy_api/main.py
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strategy_api.config import settings

def create_app() -> FastAPI:
    app = FastAPI(title="Strategy Builder API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/ping")
    def ping():
        return {"ok": True, "ts": int(time.time() * 1000)}

    return app

app = create_app()

def run():
    import uvicorn
    uvicorn.run("strategy_api.main:app", host="127.0.0.1", port=8000, reload=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/api/test_app.py -v`
Expected: PASS

- [ ] **Step 5: Boot check**

Run: `venv/bin/uvicorn strategy_api.main:app --port 8000 &` then `curl -s localhost:8000/api/ping` then kill the server.
Expected: `{"ok":true,"ts":...}`

- [ ] **Step 6: Commit**

```bash
git add src/strategy_api/main.py tests/api/test_app.py
git commit -m "feat(api): FastAPI app factory with /api/ping"
```

---

## Phase 1 — Unified research core

### Task 1.1: Create `strategy_research.ui_research` (absorb grid_search)

**Files:**
- Create: `src/strategy_research/ui_research.py`
- Create: `src/strategy_research/_ui_engine/__init__.py` (empty)
- Create: `src/strategy_research/_ui_engine/{features,labels,backtest,validation,regime,position_size,smc_features}.py` (moved from `app/public/engine/`)
- Test: `tests/api/test_ui_research_shape.py`

> The current `app/public/engine` modules already work and are the source of the UI's JSON. Move them verbatim into `src/strategy_research/_ui_engine/` (preserving their relative imports — they import each other by bare module name, so keeping them in one package and using package-relative imports works), then expose one clean entry function. This is the lowest-risk path to "lose nothing"; reconciliation with the older `strategy_research` modules is a follow-up, not this pass.

- [ ] **Step 1: Move engine modules into the package**

```bash
mkdir -p src/strategy_research/_ui_engine
git mv app/public/engine/features.py        src/strategy_research/_ui_engine/features.py
git mv app/public/engine/labels.py          src/strategy_research/_ui_engine/labels.py
git mv app/public/engine/backtest.py         src/strategy_research/_ui_engine/backtest.py
git mv app/public/engine/validation.py       src/strategy_research/_ui_engine/validation.py
git mv app/public/engine/regime.py           src/strategy_research/_ui_engine/regime.py
git mv app/public/engine/position_size.py    src/strategy_research/_ui_engine/position_size.py
git mv app/public/engine/smc_features.py     src/strategy_research/_ui_engine/smc_features.py
git mv app/public/engine/grid_search.py      src/strategy_research/_ui_engine/grid_search.py
git mv app/public/engine/data_fetcher.py     src/strategy_research/_ui_engine/data_fetcher.py
touch src/strategy_research/_ui_engine/__init__.py
```

- [ ] **Step 2: Fix intra-package imports**

In each moved file, change bare sibling imports to package-relative. Example in `grid_search.py`:

```python
# BEFORE
from features import build_mtf_features
from labels import BarrierCfg, label_both_sides
from backtest import backtest
from validation import walk_forward, shuffle_test
# AFTER
from .features import build_mtf_features
from .labels import BarrierCfg, label_both_sides
from .backtest import backtest
from .validation import walk_forward, shuffle_test
```

Apply the same `.module` prefix to every cross-module import inside `_ui_engine/*.py` (check `grid_search`, `backtest`, `labels`, `validation`, `regime`, `position_size`, `smc_features`, `data_fetcher`). Do not change pandas/numpy/xgboost imports.

- [ ] **Step 3: Verify `run_grid_search` writes per-side `label_stats`**

Open `src/strategy_research/_ui_engine/grid_search.py` and confirm each `results[rr][side]` dict includes a `label_stats` key with `total,long_wins,short_wins,no_trade`. If it only computes `lw`/`sw` locally (as the head shows), add to the per-side result dict:

```python
results[side_name] = {
    "strategies": strategies,
    "top_features": top_features,
    "walk_forward": wf,
    "label_stats": {
        "total": int(n),
        "long_wins": int(lw),
        "short_wins": int(sw),
        "no_trade": int(n - lw - sw),
    },
}
```

(Match the exact key the existing code already uses for strategies/top_features/walk_forward; only add `label_stats` if absent.)

- [ ] **Step 4: Write the clean entry module**

```python
# src/strategy_research/ui_research.py
"""In-process entry point for the dashboard's multi-RR grid search.
Returns the dict the frontend consumes — no file I/O, no subprocess.
"""
from typing import Optional
from ._ui_engine.data_fetcher import fetch_symbol_mtf
from ._ui_engine.grid_search import discover_for_rr, RR_CONFIGS

def run_research(
    symbol: str,
    days: int = 60,
    leverage: float = 10.0,
    cost: float = 0.0009,
    horizon: int = 120,
    rrs: Optional[list[str]] = None,
) -> dict:
    rrs = rrs or ["2:1"]
    data = fetch_symbol_mtf(symbol, days=days)
    if "1m" not in data or len(data["1m"]) < 1000:
        raise ValueError(f"Insufficient 1m data for {symbol}")

    from ._ui_engine.features import build_mtf_features
    features = build_mtf_features(data)

    results = {}
    for rr in rrs:
        if rr not in RR_CONFIGS:
            continue
        try:
            results[rr] = discover_for_rr(features, data["1m"], rr, leverage, cost, horizon)
        except Exception as exc:  # one RR failing must not kill the run
            results[rr] = {"long": {"error": str(exc)}, "short": {"error": str(exc)}}
    return {"symbol": symbol, "results": results}
```

> If `discover_for_rr`'s signature differs from `(features, data_1m, rr, leverage, cost, horizon)`, adapt the call to match the actual signature in the moved `grid_search.py`. Read it first.

- [ ] **Step 5: Write the golden-shape test**

```python
# tests/api/test_ui_research_shape.py
import pytest
from strategy_research import ui_research

REQUIRED_SIDE_KEYS = {"strategies", "top_features", "walk_forward", "label_stats"}

@pytest.mark.slow
def test_run_research_shape():
    out = ui_research.run_research("SOLUSDT", days=30, rrs=["1:1"])
    assert out["symbol"] == "SOLUSDT"
    rr = out["results"]["1:1"]
    for side in ("long", "short"):
        assert side in rr
        if "error" in rr[side]:
            continue
        assert REQUIRED_SIDE_KEYS.issubset(rr[side].keys())
        ls = rr[side]["label_stats"]
        assert {"total", "long_wins", "short_wins", "no_trade"}.issubset(ls.keys())
```

- [ ] **Step 6: Run the shape test (network — hits Binance)**

Run: `venv/bin/pytest tests/api/test_ui_research_shape.py -v -m slow`
Expected: PASS (requires network for Binance REST). If it fails on shape, fix the per-side dict in `grid_search.py`; if it fails on network, note it and run later.

- [ ] **Step 7: Commit**

```bash
git add src/strategy_research/ tests/api/test_ui_research_shape.py
git commit -m "feat(research): unify UI grid-search engine into strategy_research.ui_research"
```

### Task 1.2: Register the `slow` marker

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add pytest config**

```toml
[tool.pytest.ini_options]
markers = ["slow: tests that hit the network or take >5s"]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "test: register slow pytest marker"
```

---

## Phase 2 — Market service + router

### Task 2.1: Binance market service (port of market.ts)

**Files:**
- Create: `src/strategy_api/services/__init__.py` (empty)
- Create: `src/strategy_api/services/binance.py`
- Test: `tests/api/test_binance_service.py`

- [ ] **Step 1: Write the failing test (mocked transport)**

```python
# tests/api/test_binance_service.py
import httpx
import pytest
from strategy_api.services import binance

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

@pytest.fixture
def anyio_backend():
    return "asyncio"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/api/test_binance_service.py -v`
Expected: FAIL — `ModuleNotFoundError: strategy_api.services.binance`

- [ ] **Step 3: Implement the service**

```python
# src/strategy_api/services/binance.py
import httpx
from strategy_api.config import settings

_TIMEOUT = httpx.Timeout(8.0)

async def _get(path: str):
    async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
        r = await c.get(f"{settings.binance_fapi}{path}", headers={"Accept": "application/json"})
        r.raise_for_status()
        return r.json()

async def mark_price(symbol: str) -> dict:
    d = await _get(f"/fapi/v1/premiumIndex?symbol={symbol}")
    return {
        "symbol": d["symbol"],
        "markPrice": float(d["markPrice"]),
        "indexPrice": float(d["indexPrice"]),
        "fundingRate": float(d["lastFundingRate"]),
        "nextFundingTime": d["nextFundingTime"],
        "time": d["time"],
    }

async def ticker_24h(symbol: str) -> dict:
    d = await _get(f"/fapi/v1/ticker/24hr?symbol={symbol}")
    return {
        "symbol": d["symbol"],
        "priceChange": float(d["priceChange"]),
        "priceChangePct": float(d["priceChangePercent"]),
        "high24h": float(d["highPrice"]),
        "low24h": float(d["lowPrice"]),
        "volume24h": float(d["volume"]),
        "quoteVolume24h": float(d["quoteVolume"]),
        "lastPrice": float(d["lastPrice"]),
        "openPrice": float(d["openPrice"]),
        "count": d["count"],
    }

async def klines(symbol: str, interval: str, limit: int = 100) -> list[dict]:
    rows = await _get(f"/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}")
    return [{
        "openTime": r[0], "open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
        "close": float(r[4]), "volume": float(r[5]), "closeTime": r[6],
        "quoteVolume": float(r[7]), "nTrades": r[8],
        "takerBuyBase": float(r[9]), "takerBuyQuote": float(r[10]),
    } for r in rows]

async def top_symbols() -> list[dict]:
    rows = await _get("/fapi/v1/ticker/24hr")
    usdt = [d for d in rows if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > 0]
    usdt.sort(key=lambda d: float(d["quoteVolume"]), reverse=True)
    return [{
        "symbol": d["symbol"], "lastPrice": float(d["lastPrice"]),
        "priceChangePct": float(d["priceChangePercent"]), "quoteVolume": float(d["quoteVolume"]),
    } for d in usdt[:30]]
```

- [ ] **Step 4: Add anyio test dep + run**

Run: `venv/bin/pip install anyio && venv/bin/pytest tests/api/test_binance_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/strategy_api/services/__init__.py src/strategy_api/services/binance.py tests/api/test_binance_service.py
git commit -m "feat(api): Binance market data service (httpx port of market.ts)"
```

### Task 2.2: Market router

**Files:**
- Create: `src/strategy_api/routers/__init__.py` (empty)
- Create: `src/strategy_api/routers/market.py`
- Modify: `src/strategy_api/main.py`
- Test: `tests/api/test_market_router.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_market_router.py
from fastapi.testclient import TestClient
from strategy_api.main import create_app
from strategy_api.services import binance

def test_mark_price_endpoint(monkeypatch):
    async def fake(symbol): return {"symbol": symbol, "markPrice": 1.0, "indexPrice": 1.0,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/api/test_market_router.py -v`
Expected: FAIL — market route not mounted (404) / module missing.

- [ ] **Step 3: Implement router**

```python
# src/strategy_api/routers/market.py
from fastapi import APIRouter, Query
from strategy_api.services import binance

router = APIRouter(prefix="/api/market", tags=["market"])
_SYMBOL = r"^[A-Z0-9]{3,12}$"

@router.get("/mark-price")
async def mark_price(symbol: str = Query(..., pattern=_SYMBOL)):
    return await binance.mark_price(symbol)

@router.get("/ticker-24h")
async def ticker_24h(symbol: str = Query(..., pattern=_SYMBOL)):
    return await binance.ticker_24h(symbol)

@router.get("/klines")
async def klines(
    symbol: str = Query(..., pattern=_SYMBOL),
    interval: str = Query(..., pattern=r"^(1m|3m|5m|15m|30m|1h|4h|1d)$"),
    limit: int = Query(100, ge=1, le=1500),
):
    return await binance.klines(symbol, interval, limit)

@router.get("/top-symbols")
async def top_symbols():
    return await binance.top_symbols()
```

- [ ] **Step 4: Mount router in main.py**

In `src/strategy_api/main.py`, inside `create_app()` before `return app`:

```python
    from strategy_api.routers import market
    app.include_router(market.router)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `venv/bin/pytest tests/api/test_market_router.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/strategy_api/routers/ src/strategy_api/main.py tests/api/test_market_router.py
git commit -m "feat(api): /api/market/* endpoints"
```

---

## Phase 3 — DB + Ollama + AI router

### Task 3.1: SQLAlchemy models + session

**Files:**
- Create: `src/strategy_api/db/__init__.py` (empty)
- Create: `src/strategy_api/db/models.py`
- Create: `src/strategy_api/db/session.py`
- Test: `tests/api/test_db.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_db.py
from strategy_api.db.models import Base, ChatMessage
from strategy_api.db.session import make_engine, make_session_factory

def test_chat_message_roundtrip(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(engine)
    Session = make_session_factory(engine)
    with Session() as s:
        s.add(ChatMessage(session_id="abc", role="user", content="hi", model="m"))
        s.commit()
    with Session() as s:
        rows = s.query(ChatMessage).all()
        assert len(rows) == 1 and rows[0].content == "hi"
        assert isinstance(rows[0].created_at, int)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/api/test_db.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement models (match Drizzle schema; epoch-int timestamps)**

```python
# src/strategy_api/db/models.py
import time
from sqlalchemy import Integer, Text, Float
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

def _now() -> int:
    return int(time.time())

class Base(DeclarativeBase):
    pass

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, default="qwen3.5:4b")
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)

class ResearchSession(Base):
    __tablename__ = "research_sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    symbol: Mapped[str] = mapped_column(Text, nullable=False)
    rr_config: Mapped[str] = mapped_column(Text, nullable=False)
    leverage: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    days: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    best_strategy: Mapped[str | None] = mapped_column(Text, nullable=True)
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit_factor: Mapped[float | None] = mapped_column(Float, nullable=True)
    expectancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)

class StrategyInsight(Base):
    __tablename__ = "strategy_insights"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(Text, nullable=False)
    strategy_name: Mapped[str] = mapped_column(Text, nullable=False)
    insight_type: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, default="qwen3.5:4b")
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)
```

```python
# src/strategy_api/db/session.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from strategy_api.config import settings
from strategy_api.db.models import Base

def make_engine(url: str | None = None):
    return create_engine(url or f"sqlite:///{settings.sqlite_path}", future=True)

def make_session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)

_engine = make_engine()
Base.metadata.create_all(_engine)  # creates tables only if missing; preserves existing rows
SessionLocal = make_session_factory(_engine)

def get_session():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/api/test_db.py -v`
Expected: PASS

- [ ] **Step 5: Move the live DB file to repo root**

```bash
[ -f app/sqlite.db ] && git mv app/sqlite.db sqlite.db || echo "no app/sqlite.db; fresh DB will be created at repo root"
```

- [ ] **Step 6: Commit**

```bash
git add src/strategy_api/db/ tests/api/test_db.py sqlite.db 2>/dev/null; git add src/strategy_api/db/ tests/api/test_db.py
git commit -m "feat(api): SQLAlchemy models + session matching Drizzle schema"
```

### Task 3.2: Ollama service (port of ollama.ts)

**Files:**
- Create: `src/strategy_api/services/ollama.py`
- Test: `tests/api/test_ollama_service.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_ollama_service.py
import httpx, pytest
from strategy_api.services import ollama

@pytest.fixture
def anyio_backend(): return "asyncio"

@pytest.mark.anyio
async def test_chat_completion(monkeypatch):
    async def fake_post(self, url, **kw):
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"},
                                         "done": True, "total_duration": 1}, request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    out = await ollama.chat_completion([{"role": "user", "content": "hi"}], "m")
    assert out["message"]["content"] == "ok"

@pytest.mark.anyio
async def test_health_false_on_error(monkeypatch):
    async def boom(self, url, **kw): raise httpx.ConnectError("no")
    monkeypatch.setattr(httpx.AsyncClient, "get", boom)
    assert await ollama.health_check() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/api/test_ollama_service.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement service**

Translate `app/api/lib/ollama.ts` to Python, preserving the system/user prompt text **verbatim** in `generate_strategy` and `analyze_market`. Signatures:

```python
# src/strategy_api/services/ollama.py
import httpx
from strategy_api.config import settings

async def chat_completion(messages: list[dict], model: str | None = None,
                          temperature: float = 0.7) -> dict:
    model = model or settings.ollama_agent_model
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as c:
        r = await c.post(f"{settings.ollama_base_url}/api/chat", json={
            "model": model, "messages": messages, "stream": False,
            "options": {"temperature": temperature},
        })
        if r.status_code != 200:
            raise RuntimeError(f"Ollama error ({r.status_code}): {r.text}")
        return r.json()

async def generate_strategy(symbol: str, rr: str, features: list[str],
                            metrics: dict, model: str | None = None) -> str:
    system_prompt = ("You are an expert quantitative trading strategist specializing in crypto "
                     "futures. You analyze backtest results and feature importance to generate "
                     "human-readable strategy rules. You only output strategies that pass "
                     "statistical validation. You are concise and specific.")
    metric_lines = "\n".join(
        f"- {k}: {v:.4f}" if isinstance(v, (int, float)) else f"- {k}: {v}"
        for k, v in metrics.items())
    user_prompt = (
        f"Analyze this strategy research data for {symbol} with R:R {rr}:\n\n"
        f"BACKTEST METRICS:\n{metric_lines}\n\n"
        f"TOP PREDICTIVE FEATURES:\n" + "\n".join(features) + "\n\n"
        "Generate a human-readable trading rule in this exact format:\n\n"
        "STRATEGY NAME: [descriptive name]\nSIDE: [LONG or SHORT]\nENTRY CONDITIONS:\n"
        "1. [feature] [operator] [threshold]\n2. [feature] [operator] [threshold]\n...\n\n"
        "EXIT RULES:\n- Target: [price target]\n- Stop: [stop loss]\n- Max Hold: [time limit]\n\n"
        f"RATIONALE: [1-2 sentence explanation of why these conditions work for {symbol}]\n\n"
        "RISK WARNING: [specific risk for this setup]")
    res = await chat_completion(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}], model, temperature=0.4)
    return res["message"]["content"]

async def analyze_market(symbol: str, price_data: dict, model: str | None = None) -> str:
    system_prompt = ("You are a crypto market analyst. Provide brief, actionable analysis. Focus "
                     "on what matters for short-term futures trading (1-4 hour holds).")
    chg = price_data["change24h"]
    user_prompt = (
        f"Analyze {symbol} market conditions:\n"
        f"- Price: ${price_data['currentPrice']}\n"
        f"- 24h Change: {'+' if chg > 0 else ''}{chg*100:.2f}%\n"
        f"- 24h High: ${price_data['high24h']}\n- 24h Low: ${price_data['low24h']}\n"
        f"- 24h Volume: ${price_data['volume24h']/1e6:.1f}M\n\n"
        "Provide:\n1. MARKET REGIME: (trending up / ranging / trending down / volatile)\n"
        "2. KEY LEVELS: support and resistance\n"
        "3. BEST RR SETUP: which risk:reward ratio is likely optimal right now\n"
        "4. CAUTION: what could invalidate the setup")
    res = await chat_completion(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}], model, temperature=0.5)
    return res["message"]["content"]

async def list_models() -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as c:
            r = await c.get(f"{settings.ollama_base_url}/api/tags")
            if r.status_code != 200:
                return []
            return [m.get("name") or m.get("model") for m in r.json().get("models", [])]
    except Exception:
        return []

async def health_check() -> bool:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as c:
            r = await c.get(f"{settings.ollama_base_url}/api/tags")
            return r.status_code == 200
    except Exception:
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/api/test_ollama_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/strategy_api/services/ollama.py tests/api/test_ollama_service.py
git commit -m "feat(api): Ollama service (httpx port of ollama.ts)"
```

### Task 3.3: AI router (mirrors ai.ts)

**Files:**
- Create: `src/strategy_api/routers/ai.py`
- Create: `src/strategy_api/schemas.py`
- Modify: `src/strategy_api/main.py`
- Test: `tests/api/test_ai_router.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_ai_router.py
from fastapi.testclient import TestClient
from strategy_api.main import create_app
from strategy_api.services import ollama

def test_health(monkeypatch):
    async def ok(): return True
    async def models(): return ["m1"]
    monkeypatch.setattr(ollama, "health_check", ok)
    monkeypatch.setattr(ollama, "list_models", models)
    c = TestClient(create_app())
    r = c.get("/api/ai/health")
    assert r.status_code == 200 and r.json()["ok"] is True

def test_chat_persists(monkeypatch):
    async def fake_chat(messages, model=None, temperature=0.7):
        return {"message": {"content": "hello"}, "total_duration": 1,
                "prompt_eval_count": 2, "eval_count": 3}
    monkeypatch.setattr(ollama, "chat_completion", fake_chat)
    c = TestClient(create_app())
    r = c.post("/api/ai/chat", json={"messages": [{"role": "user", "content": "hi"}],
                                     "model": "m", "sessionId": "s1"})
    assert r.status_code == 200 and r.json()["content"] == "hello"
    hist = c.get("/api/ai/history?sessionId=s1").json()
    assert len(hist) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/api/test_ai_router.py -v`
Expected: FAIL — ai route missing.

- [ ] **Step 3: Implement pydantic request schemas**

```python
# src/strategy_api/schemas.py
from pydantic import BaseModel, Field

class ChatMsg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    messages: list[ChatMsg]
    model: str = "qwen3.5:4b"
    temperature: float = 0.7
    sessionId: str | None = None

class GenStrategyReq(BaseModel):
    symbol: str
    rr: str
    features: list[str]
    metrics: dict[str, float]
    model: str = "qwen3.5:4b"
    sessionId: str | None = None

class PriceData(BaseModel):
    currentPrice: float
    change24h: float
    high24h: float
    low24h: float
    volume24h: float

class MarketAnalysisReq(BaseModel):
    symbol: str
    priceData: PriceData
    model: str = "qwen3.5:4b"

class SaveSessionReq(BaseModel):
    sessionId: str
    symbol: str
    rrConfig: str
    leverage: int = 10
    days: int = 60
    resultJson: str | None = None
    bestStrategy: str | None = None
    winRate: float | None = None
    profitFactor: float | None = None
    expectancy: float | None = None

class ResearchRunReq(BaseModel):
    symbol: str = Field("SOLUSDT", pattern=r"^[A-Z0-9]{3,12}$")
    days: int = Field(60, ge=7, le=365)
    leverage: float = Field(10, ge=1, le=125)
    horizon: int = Field(120, ge=30, le=480)
    rrs: list[str] = Field(default_factory=lambda: ["2:1"], min_length=1)
```

- [ ] **Step 4: Implement the AI router**

```python
# src/strategy_api/routers/ai.py
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, delete, desc
from sqlalchemy.orm import Session
from strategy_api.db.session import get_session
from strategy_api.db.models import ChatMessage, ResearchSession, StrategyInsight
from strategy_api.services import ollama
from strategy_api.schemas import (ChatReq, GenStrategyReq, MarketAnalysisReq, SaveSessionReq)

router = APIRouter(prefix="/api/ai", tags=["ai"])

@router.get("/health")
async def health():
    ok = await ollama.health_check()
    models = await ollama.list_models() if ok else []
    return {"ok": ok, "models": models,
            "message": "Ollama connected" if ok else "Ollama unavailable - check OLLAMA_BASE_URL"}

@router.post("/chat")
async def chat(req: ChatReq, db: Session = Depends(get_session)):
    res = await ollama.chat_completion([m.model_dump() for m in req.messages],
                                       req.model, req.temperature)
    content = res["message"]["content"]
    if req.sessionId:
        db.add(ChatMessage(session_id=req.sessionId, role="user",
                           content=req.messages[-1].content, model=req.model))
        db.add(ChatMessage(session_id=req.sessionId, role="assistant",
                           content=content, model=req.model))
        db.commit()
    return {"content": content, "model": req.model, "timing": {
        "total": res.get("total_duration"), "promptTokens": res.get("prompt_eval_count"),
        "completionTokens": res.get("eval_count")}}

@router.get("/history")
def history(sessionId: str = Query(...), db: Session = Depends(get_session)):
    rows = db.scalars(select(ChatMessage).where(ChatMessage.session_id == sessionId)
                      .order_by(ChatMessage.created_at)).all()
    return [{"id": r.id, "sessionId": r.session_id, "role": r.role, "content": r.content,
             "model": r.model, "createdAt": r.created_at} for r in rows]

@router.post("/generate-strategy")
async def generate_strategy(req: GenStrategyReq, db: Session = Depends(get_session)):
    analysis = await ollama.generate_strategy(req.symbol, req.rr, req.features, req.metrics, req.model)
    if req.sessionId:
        db.add(StrategyInsight(session_id=req.sessionId, strategy_name=f"{req.symbol}_{req.rr}",
                               insight_type="analysis", content=analysis, model=req.model))
        db.commit()
    return {"analysis": analysis}

@router.post("/market-analysis")
async def market_analysis(req: MarketAnalysisReq):
    analysis = await ollama.analyze_market(req.symbol, req.priceData.model_dump(), req.model)
    return {"analysis": analysis}

@router.post("/save-session")
def save_session(req: SaveSessionReq, db: Session = Depends(get_session)):
    db.add(ResearchSession(session_id=req.sessionId, symbol=req.symbol, rr_config=req.rrConfig,
                           leverage=req.leverage, days=req.days, status="completed",
                           result_json=req.resultJson, best_strategy=req.bestStrategy,
                           win_rate=req.winRate, profit_factor=req.profitFactor,
                           expectancy=req.expectancy))
    db.commit()
    return {"success": True}

@router.get("/sessions")
def sessions(db: Session = Depends(get_session)):
    rows = db.scalars(select(ResearchSession).order_by(desc(ResearchSession.created_at)).limit(50)).all()
    return [{"id": r.id, "sessionId": r.session_id, "symbol": r.symbol, "rrConfig": r.rr_config,
             "leverage": r.leverage, "days": r.days, "status": r.status,
             "bestStrategy": r.best_strategy, "winRate": r.win_rate,
             "profitFactor": r.profit_factor, "expectancy": r.expectancy,
             "createdAt": r.created_at} for r in rows]

@router.get("/insights")
def insights(sessionId: str = Query(...), db: Session = Depends(get_session)):
    rows = db.scalars(select(StrategyInsight).where(StrategyInsight.session_id == sessionId)
                      .order_by(desc(StrategyInsight.created_at))).all()
    return [{"id": r.id, "sessionId": r.session_id, "strategyName": r.strategy_name,
             "insightType": r.insight_type, "content": r.content, "model": r.model,
             "createdAt": r.created_at} for r in rows]

@router.post("/clear-history")
def clear_history(sessionId: str = Query(...), db: Session = Depends(get_session)):
    db.execute(delete(ChatMessage).where(ChatMessage.session_id == sessionId))
    db.commit()
    return {"success": True}
```

- [ ] **Step 5: Mount router in main.py**

In `create_app()` add alongside the market include:

```python
    from strategy_api.routers import ai
    app.include_router(ai.router)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `venv/bin/pytest tests/api/test_ai_router.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/strategy_api/routers/ai.py src/strategy_api/schemas.py src/strategy_api/main.py tests/api/test_ai_router.py
git commit -m "feat(api): /api/ai/* endpoints with DB persistence"
```

---

## Phase 4 — Research router (the core wire-up)

### Task 4.1: Research service + result mapper + router

**Files:**
- Create: `src/strategy_api/services/research.py`
- Create: `src/strategy_api/routers/research.py`
- Modify: `src/strategy_api/main.py`
- Test: `tests/api/test_research_router.py`

> `services/research.py` runs the CPU-heavy `ui_research.run_research` off the event loop and maps the raw dict to the `ResearchResult` shape — this is the Python port of the TS `mapResult` function.

- [ ] **Step 1: Write the failing test (research mocked)**

```python
# tests/api/test_research_router.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/api/test_research_router.py -v`
Expected: FAIL — module/route missing.

- [ ] **Step 3: Implement the service (run-off-loop + mapper)**

```python
# src/strategy_api/services/research.py
import anyio
from strategy_research import ui_research

RR_CFGS = {
    "3:1": {"upPct": 0.015, "dnPct": 0.005}, "2:1": {"upPct": 0.010, "dnPct": 0.005},
    "1:1": {"upPct": 0.010, "dnPct": 0.010}, "1:2": {"upPct": 0.005, "dnPct": 0.010},
    "1:3": {"upPct": 0.005, "dnPct": 0.015},
}

def _run_blocking(*, symbol, days, leverage, horizon, rrs):
    return ui_research.run_research(symbol=symbol, days=days, leverage=leverage,
                                    horizon=horizon, rrs=rrs)

async def run(symbol, days, leverage, horizon, rrs) -> dict:
    raw = await anyio.to_thread.run_sync(
        lambda: _run_blocking(symbol=symbol, days=days, leverage=leverage,
                              horizon=horizon, rrs=rrs))
    return _map_result(raw, symbol, rrs, leverage)

def _map_result(raw: dict, symbol: str, rrs: list[str], leverage: float) -> dict:
    strategies, top_features = [], []
    walk_forward = {"folds": 0, "avgTestAuc": 0.5, "minTestAuc": 0.5, "stability": 0,
                    "degradation": 0, "isValid": False, "foldResults": []}
    labels = {"longWins": 0, "shortWins": 0, "noTrade": 0, "total": 0}

    for rr in rrs:
        rr_data = (raw.get("results") or {}).get(rr)
        if not rr_data:
            continue
        for side in ("long", "short"):
            side_data = rr_data.get(side)
            if not side_data or side_data.get("error"):
                continue
            for strat in side_data.get("strategies", []):
                m = strat.get("metrics", {})
                strategies.append({
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
                    "isViable": strat.get("is_viable", False)})
            wf = side_data.get("walk_forward")
            if wf and walk_forward["folds"] == 0:
                walk_forward = {
                    "folds": wf.get("folds", 0), "avgTestAuc": wf.get("avg_test_auc", 0.5),
                    "minTestAuc": wf.get("min_test_auc", 0.5), "stability": wf.get("stability", 0),
                    "degradation": wf.get("degradation", 0), "isValid": wf.get("is_valid", False),
                    "foldResults": [{"fold": f.get("fold"), "trainAuc": f.get("train_auc", 0.5),
                                     "testAuc": f.get("test_auc", 0.5),
                                     "testPrecision": f.get("test_precision", 0),
                                     "testRecall": f.get("test_recall", 0),
                                     "nTrain": f.get("n_train", 0), "nTest": f.get("n_test", 0)}
                                    for f in wf.get("fold_results", [])]}
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

    primary = rrs[0] if rrs else "2:1"
    cfg = RR_CFGS.get(primary, {"upPct": 0.01, "dnPct": 0.005})
    return {"symbol": symbol,
            "config": {"symbol": symbol, "rr": primary, "upPct": cfg["upPct"],
                       "dnPct": cfg["dnPct"], "leverage": leverage, "horizon": 120, "side": "both"},
            "strategies": strategies, "walkForward": walk_forward,
            "topFeatures": top_features, "labels": labels}
```

- [ ] **Step 4: Implement the router**

```python
# src/strategy_api/routers/research.py
from fastapi import APIRouter
from strategy_api.schemas import ResearchRunReq
from strategy_api.services import research as research_svc

router = APIRouter(prefix="/api/research", tags=["research"])

@router.post("/run")
async def run(req: ResearchRunReq):
    return await research_svc.run(req.symbol, req.days, req.leverage, req.horizon, req.rrs)
```

- [ ] **Step 5: Mount router in main.py**

```python
    from strategy_api.routers import research
    app.include_router(research.router)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `venv/bin/pytest tests/api/test_research_router.py -v`
Expected: PASS

- [ ] **Step 7: Parity check against the old Node output (manual, network)**

Run: `venv/bin/uvicorn strategy_api.main:app --port 8000 &` then
`curl -s -X POST localhost:8000/api/research/run -H 'content-type: application/json' -d '{"symbol":"SOLUSDT","days":30,"rrs":["1:1"],"leverage":10}' | python -m json.tool | head -40`
Expected: a `ResearchResult` JSON with `strategies`, `walkForward`, `topFeatures`, `labels`. Kill server after.

- [ ] **Step 8: Commit**

```bash
git add src/strategy_api/services/research.py src/strategy_api/routers/research.py src/strategy_api/main.py tests/api/test_research_router.py
git commit -m "feat(api): /api/research/run wired to unified in-process engine"
```

### Task 4.2: Consistent error responses

**Files:**
- Modify: `src/strategy_api/main.py`
- Test: `tests/api/test_errors.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_errors.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/api/test_errors.py -v`
Expected: FAIL — returns 500, not 400.

- [ ] **Step 3: Add exception handlers in main.py**

Inside `create_app()`, before `return app`:

```python
    from fastapi import Request
    from fastapi.responses import JSONResponse
    import httpx

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"error": {"code": 400, "message": str(exc)}})

    @app.exception_handler(httpx.HTTPError)
    async def _upstream_error(request: Request, exc: httpx.HTTPError):
        return JSONResponse(status_code=502, content={"error": {"code": 502, "message": str(exc)}})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/api/test_errors.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/strategy_api/main.py tests/api/test_errors.py
git commit -m "feat(api): consistent error envelope for value/upstream errors"
```

---

## Phase 5 — Frontend rewire (tRPC → REST)

### Task 5.1: Typed API client

**Files:**
- Create: `app/src/lib/api.ts`
- Test: `app/src/lib/api.test.ts`

> Frontend still lives under `app/` until Task 6.1 renames it. Use the existing vitest setup.

- [ ] **Step 1: Write the failing test**

```typescript
// app/src/lib/api.test.ts
import { describe, it, expect, vi } from "vitest";
import { apiPost } from "./api";

describe("apiPost", () => {
  it("posts JSON and returns parsed body", async () => {
    global.fetch = vi.fn(async () =>
      new Response(JSON.stringify({ ok: true }), { status: 200, headers: { "content-type": "application/json" } })
    ) as any;
    const out = await apiPost<{ ok: boolean }>("/api/ping", {});
    expect(out.ok).toBe(true);
  });

  it("throws normalized error on non-2xx", async () => {
    global.fetch = vi.fn(async () =>
      new Response(JSON.stringify({ error: { message: "boom" } }), { status: 400 })
    ) as any;
    await expect(apiPost("/api/x", {})).rejects.toThrow("boom");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && npx vitest run src/lib/api.test.ts`
Expected: FAIL — `./api` not found.

- [ ] **Step 3: Implement the client**

```typescript
// app/src/lib/api.ts
const BASE = "";  // same-origin; Vite proxies /api in dev, FastAPI serves in prod

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = res.statusText;
    try { const b = await res.json(); msg = b?.error?.message ?? b?.detail ?? msg; } catch { /* noop */ }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export async function apiGet<T>(path: string): Promise<T> {
  return handle<T>(await fetch(`${BASE}${path}`, { credentials: "include" }));
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  return handle<T>(await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify(body),
  }));
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app && npx vitest run src/lib/api.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/src/lib/api.ts app/src/lib/api.test.ts
git commit -m "feat(frontend): typed REST api client"
```

### Task 5.2: React Query hooks

**Files:**
- Create: `app/src/hooks/api/index.ts`
- Create: `app/src/providers/query.tsx`

- [ ] **Step 1: Implement the QueryClient provider (replaces TRPCProvider)**

```tsx
// app/src/providers/query.tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

const queryClient = new QueryClient();

export function AppQueryProvider({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}
```

- [ ] **Step 2: Implement the hooks (cover the 4 call sites + health)**

```tsx
// app/src/hooks/api/index.ts
import { useMutation, useQuery } from "@tanstack/react-query";
import { apiGet, apiPost } from "@/lib/api";
import type { ResearchResult } from "@/types";

export interface RunResearchInput {
  symbol: string; days: number; leverage: number; horizon: number; rrs: string[];
}

export function useResearchRun(opts?: {
  onSuccess?: (d: ResearchResult) => void; onError?: (e: Error) => void;
}) {
  return useMutation<ResearchResult, Error, RunResearchInput>({
    mutationFn: (input) => apiPost<ResearchResult>("/api/research/run", input),
    onSuccess: opts?.onSuccess,
    onError: opts?.onError,
  });
}

export function useAiHealth() {
  return useQuery({
    queryKey: ["ai", "health"],
    queryFn: () => apiGet<{ ok: boolean; models: string[]; message: string }>("/api/ai/health"),
  });
}

export interface ChatInput {
  messages: { role: string; content: string }[]; model?: string; sessionId?: string;
}

export function useAiChat(opts?: {
  onSuccess?: (d: { content: string; model: string }) => void; onError?: (e: Error) => void;
}) {
  return useMutation<{ content: string; model: string }, Error, ChatInput>({
    mutationFn: (input) => apiPost("/api/ai/chat", input),
    onSuccess: opts?.onSuccess,
    onError: opts?.onError,
  });
}

export interface GenStrategyInput {
  symbol: string; rr: string; features: string[]; metrics: Record<string, number>;
  model?: string; sessionId?: string;
}

export function useGenerateStrategy(opts?: {
  onSuccess?: (d: { analysis: string }) => void; onError?: (e: Error) => void;
}) {
  return useMutation<{ analysis: string }, Error, GenStrategyInput>({
    mutationFn: (input) => apiPost("/api/ai/generate-strategy", input),
    onSuccess: opts?.onSuccess,
    onError: opts?.onError,
  });
}
```

- [ ] **Step 3: Typecheck**

Run: `cd app && npx tsc --noEmit`
Expected: no errors in the new files (existing tRPC files may still reference the provider until Task 5.3–5.5; that's fine — tsc is run again at the end of Phase 5).

- [ ] **Step 4: Commit**

```bash
git add app/src/hooks/api/index.ts app/src/providers/query.tsx
git commit -m "feat(frontend): react-query hooks + query provider"
```

### Task 5.3: Swap call site — App.tsx

**Files:**
- Modify: `app/src/App.tsx`

- [ ] **Step 1: Replace the import + mutation**

Change the tRPC import line:

```tsx
// REMOVE
import { trpc } from './providers/trpc';
// ADD
import { useResearchRun } from '@/hooks/api';
```

Replace the `runMutation` definition:

```tsx
// REMOVE the trpc.research.run.useMutation({...}) block
  const runMutation = useResearchRun({
    onSuccess: (data) => { setResults(data); setError(null); },
    onError: (err) => { setError(err.message); },
  });
```

(The rest of `App.tsx` — `runMutation.isPending`, `runMutation.mutate({...})` — is unchanged; the hook exposes the same React Query mutation surface.)

- [ ] **Step 2: Commit**

```bash
git add app/src/App.tsx
git commit -m "refactor(frontend): App.tsx uses useResearchRun"
```

### Task 5.4: Swap call site — AIChat.tsx

**Files:**
- Modify: `app/src/sections/AIChat.tsx`

- [ ] **Step 1: Replace imports + hooks**

```tsx
// REMOVE
import { trpc } from '../providers/trpc';
// ADD
import { useAiHealth, useAiChat } from '@/hooks/api';
```

Replace:

```tsx
// REMOVE: const healthQuery = trpc.ai.health.useQuery();
//         const chatMutation = trpc.ai.chat.useMutation({...});
  const healthQuery = useAiHealth();
  const chatMutation = useAiChat({
    // keep whatever onSuccess/onError the original block had — copy them verbatim
  });
```

> Read the original `AIChat.tsx` `useMutation` options and copy the `onSuccess`/`onError` bodies exactly into `useAiChat({...})`. All `.mutate(...)`, `.isPending`, `healthQuery.data` usages stay identical.

- [ ] **Step 2: Commit**

```bash
git add app/src/sections/AIChat.tsx
git commit -m "refactor(frontend): AIChat uses useAiHealth/useAiChat"
```

### Task 5.5: Swap call site — AIStrategyInsights.tsx

**Files:**
- Modify: `app/src/sections/AIStrategyInsights.tsx`

- [ ] **Step 1: Replace import + mutation**

```tsx
// REMOVE
import { trpc } from '../providers/trpc';
// ADD
import { useGenerateStrategy } from '@/hooks/api';
```

Replace:

```tsx
// REMOVE: const generateMutation = trpc.ai.generateStrategy.useMutation({...});
  const generateMutation = useGenerateStrategy({
    // copy the original onSuccess/onError verbatim
  });
```

- [ ] **Step 2: Commit**

```bash
git add app/src/sections/AIStrategyInsights.tsx
git commit -m "refactor(frontend): AIStrategyInsights uses useGenerateStrategy"
```

### Task 5.6: Swap provider in main.tsx + delete trpc provider

**Files:**
- Modify: `app/src/main.tsx`
- Delete: `app/src/providers/trpc.tsx`

- [ ] **Step 1: Update main.tsx**

```tsx
// REMOVE: import { TRPCProvider } from "@/providers/trpc"
// ADD:
import { AppQueryProvider } from "@/providers/query"
```

Replace `<TRPCProvider>...</TRPCProvider>` with `<AppQueryProvider>...</AppQueryProvider>`.

- [ ] **Step 2: Delete the tRPC provider**

```bash
git rm app/src/providers/trpc.tsx
```

- [ ] **Step 3: Typecheck the whole frontend**

Run: `cd app && npx tsc --noEmit`
Expected: no errors. If any file still imports `./providers/trpc` or `@/providers/trpc`, fix it.

- [ ] **Step 4: Run frontend tests**

Run: `cd app && npx vitest run`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/src/main.tsx && git rm app/src/providers/trpc.tsx
git commit -m "refactor(frontend): swap TRPCProvider for AppQueryProvider"
```

---

## Phase 6 — Rename, Vite wiring, prod static, deletions

### Task 6.1: Rename app/ → frontend/

**Files:**
- Move: `app/` → `frontend/`

- [ ] **Step 1: Rename**

```bash
git mv app frontend
```

- [ ] **Step 2: Verify nothing references the old path**

Run: `grep -rn "app/public\|app/api\|app/db\|/app/src" frontend src --exclude-dir=node_modules --exclude-dir=dist 2>/dev/null`
Expected: no matches (or only in docs/comments). Fix any code matches.

- [ ] **Step 3: Commit**

```bash
git commit -am "refactor: rename app/ -> frontend/"
```

### Task 6.2: Rewrite Vite config (drop Hono, proxy /api)

**Files:**
- Modify: `frontend/vite.config.ts`

- [ ] **Step 1: Replace vite.config.ts**

```typescript
import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"
const __dirname = import.meta.dirname

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  build: {
    outDir: path.resolve(__dirname, "dist"),
    emptyOutDir: true,
  },
})
```

> Removed: `@hono/vite-dev-server`, `kimi-plugin-inspect-react`, and the `@contracts`/`@db` aliases (no longer used by the frontend). If any frontend file still imports `@contracts`/`@db`, replace those imports with local types before this step.

- [ ] **Step 2: Verify dev server boots + proxies**

Run (two shells): `venv/bin/uvicorn strategy_api.main:app --port 8000` and `cd frontend && npm run dev`, then `curl -s localhost:3000/api/ping`.
Expected: `{"ok":true,...}` proxied through Vite. Stop both.

- [ ] **Step 3: Commit**

```bash
git add frontend/vite.config.ts
git commit -m "build(frontend): vite proxies /api to FastAPI, drop hono dev server"
```

### Task 6.3: FastAPI serves built frontend in prod

**Files:**
- Modify: `src/strategy_api/main.py`

- [ ] **Step 1: Mount StaticFiles with SPA fallback (only if dist exists)**

At the end of `create_app()`, before `return app`:

```python
    import os
    from fastapi.staticfiles import StaticFiles
    if os.path.isdir(settings.frontend_dist):
        app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="static")
```

> `html=True` serves `index.html` for unknown paths (SPA routing). API routes are registered before this mount, so they take precedence.

- [ ] **Step 2: Build the frontend + smoke test prod serving**

Run: `cd frontend && npm run build` then `venv/bin/uvicorn strategy_api.main:app --port 8000 &` then `curl -s -I localhost:8000/` (expect 200 text/html) and `curl -s localhost:8000/api/ping`. Stop server.
Expected: index.html served at `/`, ping still works.

- [ ] **Step 3: Commit**

```bash
git add src/strategy_api/main.py
git commit -m "feat(api): serve built frontend via StaticFiles in production"
```

### Task 6.4: Delete Node backend + DB layer + duplicate research modules

**Files:**
- Delete: `frontend/api/`, `frontend/db/`, `frontend/contracts/`
- Delete: `research/mtf_research.py`, `research/barrier_leverage.py`, `research/validation.py`
- Delete: `frontend/public/engine/` (now empty dir / __init__/__pycache__ leftovers)

- [ ] **Step 1: Remove the Node backend, DB, contracts**

```bash
git rm -r frontend/api frontend/db frontend/contracts
git rm -r frontend/public/engine 2>/dev/null || rm -rf frontend/public/engine
```

> The `app/public/engine/*.py` files were already `git mv`d in Task 1.1; this removes any leftover `__init__.py`/`__pycache__`. The "Download Engine" link in `App.tsx` (`href="/engine/main.py"`) now 404s — either point it at a kept file or remove the button. Remove the button: delete the `<a href="/engine/main.py" ...>` block in `frontend/src/App.tsx`.

- [ ] **Step 2: Rewire root research scripts off the deleted dup modules**

Run: `grep -rn "from mtf_research\|import mtf_research\|from barrier_leverage\|import barrier_leverage\|from validation\|import validation\|from research\." research run_research.py researcher.py researcher_robust.py 2>/dev/null`
For each hit, repoint to the canonical package, e.g.:

```python
# BEFORE (in research/run_research.py or similar)
from mtf_research import build_mtf_features
from barrier_leverage import triple_barrier_both_sides
from validation import purged_folds
# AFTER
from strategy_research.features.mtf_features import build_mtf_features
from strategy_research.labeling.triple_barrier import triple_barrier_both_sides
from strategy_research.validation.walkforward import purged_folds
```

> Verify the canonical symbol names exist (read the target modules). If a unique probe (leakage/liquidation) lived only in `research/validation.py`, move that function into `strategy_research/validation/walkforward.py` rather than deleting it — "lose nothing."

- [ ] **Step 3: Verify scripts still import**

Run: `venv/bin/python -c "import run_research"` (and for any other rewired root script).
Expected: no ImportError. If a script executes on import, instead run `venv/bin/python -m py_compile run_research.py researcher.py researcher_robust.py`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: delete Node backend, DB layer, and duplicate research engines"
```

### Task 6.5: Trim frontend package.json

**Files:**
- Modify: `frontend/package.json`

- [ ] **Step 1: Remove backend-only deps**

Delete these from `dependencies`: `@hono/node-server`, `hono`, `@trpc/client`, `@trpc/react-query`, `@trpc/server`, `better-sqlite3`, `drizzle-orm`, `mysql2`, `superjson`. From `devDependencies`: `@hono/vite-dev-server`, `@types/better-sqlite3`, `drizzle-kit`, `kimi-plugin-inspect-react`. Keep `@tanstack/react-query`, `zod`, React, Radix, Tailwind, recharts, sonner, vite, vitest.

- [ ] **Step 2: Remove DB scripts**

Delete `db:generate`, `db:migrate`, `db:push` from `scripts`. Replace the `build` script with:

```json
    "build": "vite build",
    "start": "echo 'Run the backend: uvicorn strategy_api.main:app' && exit 0",
```

- [ ] **Step 3: Reinstall + verify build**

Run: `cd frontend && rm -rf node_modules package-lock.json && npm install && npm run build`
Expected: clean install, successful build to `frontend/dist`.

- [ ] **Step 4: Commit**

```bash
git add frontend/package.json frontend/package-lock.json
git commit -m "build(frontend): drop Node/tRPC/Drizzle dependencies"
```

---

## Phase 7 — Final verification

### Task 7.1: Full backend test suite

- [ ] **Step 1: Run all non-slow tests**

Run: `venv/bin/pytest tests/ -v -m "not slow"`
Expected: all PASS (config, app, db, services, routers, errors).

- [ ] **Step 2: Run slow (network) shape + parity tests**

Run: `venv/bin/pytest tests/ -v -m slow`
Expected: PASS (needs network + enough Binance history).

### Task 7.2: End-to-end manual smoke

- [ ] **Step 1: Start both servers**

Run: `venv/bin/uvicorn strategy_api.main:app --port 8000` + `cd frontend && npm run dev`.

- [ ] **Step 2: In the browser at http://localhost:3000**

Verify, observing network calls hit `/api/*`:
1. Pick a symbol, select RRs, click **Run Research** → results render in Strategy Results / Validation / Feature Analysis tabs.
2. Open **AI Chat** → health indicator reflects Ollama; send a message → reply appears (if Ollama running).
3. **AI Insights** → generate analysis for a strategy.
4. Position Sizing / Strategy Player tabs render from the results.

- [ ] **Step 3: Confirm no console errors referencing tRPC or `/api/trpc`.**

### Task 7.3: Update README + docs

**Files:**
- Modify: `README.md`, `USAGE_GUIDE.md`

- [ ] **Step 1: Replace run instructions**

Document the new dev/prod commands:

```
# Dev: two processes
venv/bin/uvicorn strategy_api.main:app --reload --port 8000
cd frontend && npm run dev   # http://localhost:3000 (proxies /api)

# Prod: build frontend, serve everything from FastAPI
cd frontend && npm run build
venv/bin/uvicorn strategy_api.main:app --port 8000   # serves UI + /api
```

Remove references to Node backend, `PYTHON_BIN` spawn, and `app/public/engine`.

- [ ] **Step 2: Commit**

```bash
git add README.md USAGE_GUIDE.md
git commit -m "docs: update run instructions for unified FastAPI backend"
```

---

## Self-Review notes (coverage map)

- Spec §3.1 backend → Tasks 0.2, 0.3, 3.1, 4.2, 6.3
- Spec §3.2 endpoint map → Tasks 2.2 (market), 3.3 (ai), 4.1 (research), 0.3 (ping)
- Spec §3.3 frontend rewire → Tasks 5.1–5.6, 6.1, 6.2
- Spec §3.4 data flow → Task 4.1
- Spec §4 consolidation/deletion → Tasks 1.1, 6.4
- Spec §5 error handling → Task 4.2 + per-RR error handling in 1.1/4.1
- Spec §6 testing → golden-shape 1.1, unit tests across 2.1/3.1/3.2/3.3/4.1, suite 7.1
- Spec §7 dev/prod run → Tasks 6.2, 6.3, 7.3
- Spec §9 risks (shape drift, math divergence, blocking, DB types) → 1.1 golden test + 4.1 parity check, 3.1 epoch-int timestamps, 4.1 threadpool

**Env fix** (`OLLAMA_URL`→`OLLAMA_BASE_URL`) handled in Task 0.2/3.2.
**Known follow-ups (out of scope):** SSE progress, Binance WS ticker, reconciling `_ui_engine` math with the older `strategy_research` modules, exposing CLI-only tools in the UI.
