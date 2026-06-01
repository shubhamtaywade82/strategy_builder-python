# Unified Backend + Frontend — Strategy Builder

**Date:** 2026-05-31
**Status:** Approved design (pending spec review)
**Goal:** Collapse the repo's three duplicated Python engines and Node/Hono/tRPC backend into a single Python (FastAPI) backend that serves the existing React frontend. Lose no existing functionality. Keep the UI surface as-is (fix issues only). Heavy data analysis runs on the backend against Binance via REST (WebSocket optional/later).

---

## 1. Problem

The codebase grew in two phases and now has **split-brain duplication**:

- **`src/strategy_builder/`** — full quant engine (manual strategy library, AI agents, live exec, feature engines, full backtest). CLI-only.
- **`src/strategy_research/`** — clean XGBoost research pipeline (data → features → labels → train → backtest → walk-forward). CLI + Node API.
- **`research/`** — standalone scripts that re-implement the research pipeline (`mtf_research.py`, `barrier_leverage.py`, `validation.py` are duplicates).
- **`app/public/engine/`** — a *third* stripped Python engine (`grid_search.py`, `features.py`, `labels.py`, `backtest.py`, `validation.py`, `regime.py`, `position_size.py`, `smc_features.py`) that the Node backend spawns as a child process.

The same feature engineering, triple-barrier labeling, backtest, and walk-forward math exists in **three** places. A fix in one is silently missed in the others.

On top of that, the backend is **Node (Hono + tRPC)** which must `spawn()` a Python subprocess and shuttle JSON over a temp file — an awkward bridge for what is fundamentally a Python/Pandas workload.

## 2. Goal & Non-Goals

**Goals**
- One Python source of truth for all quant math.
- One Python backend (FastAPI) — no Node, no subprocess spawn; the API imports the engine in-process.
- React frontend preserved visually and functionally; only its data-fetch layer is rewired (tRPC → REST).
- All existing CLI tools and capabilities keep working on the unified core.
- SQLite data (chat history, research sessions, insights) preserved.

**Non-Goals (this pass)**
- No new UI surfaces for CLI-only tools (frontier sweep, AI agents, regime switcher, live exec). They keep running via CLI on the shared core; UI exposure is a later pass.
- No SSE/streaming progress UI — the run stays a blocking request with the existing loading state.
- No Binance WebSocket integration — market data stays REST-proxied (WS is a noted later enhancement).
- No redesign of UI components/visuals beyond fixing bugs surfaced by the rewire.

## 3. Target Architecture

```
python_strategy_builder/
├── pyproject.toml              # + fastapi, uvicorn[standard], httpx, sqlalchemy, pydantic-settings
├── sqlite.db                   # preserved (moved from app/ to repo root)
├── src/
│   ├── strategy_builder/       # UNCHANGED — full engine + CLI (single source of truth)
│   ├── strategy_research/      # CANONICAL research pipeline (absorbs UI-only modules)
│   │   ├── ui_research.py      # NEW: grid_search logic from app/public/engine, emits UI JSON shape
│   │   ├── regime.py           # moved from app/public/engine (if not already covered by state/)
│   │   ├── position_sizing.py  # moved from app/public/engine
│   │   └── ... (existing features/labeling/backtest/validation/ai_research)
│   └── strategy_api/           # NEW FastAPI backend package
│       ├── main.py             # app factory: routers, CORS, StaticFiles mount (prod)
│       ├── config.py           # pydantic-settings reading .env
│       ├── db/
│       │   ├── models.py       # SQLAlchemy models: chat_messages, research_sessions, strategy_insights
│       │   └── session.py      # engine + session factory (sqlite.db)
│       ├── services/
│       │   ├── ollama.py       # httpx port of api/lib/ollama.ts
│       │   ├── binance.py      # httpx port of api/market.ts (REST proxy)
│       │   └── research.py     # calls strategy_research.ui_research in a threadpool
│       └── routers/
│           ├── ping.py
│           ├── research.py     # POST /api/research/run
│           ├── ai.py           # /api/ai/*
│           └── market.py       # /api/market/*
├── frontend/                   # was app/ (React only; api/, db/, public/engine/ removed)
│   ├── package.json            # Node API + drizzle deps removed
│   ├── vite.config.ts          # dev proxy /api -> http://127.0.0.1:8000
│   └── src/
│       ├── lib/api.ts          # NEW typed fetch client (zod-validated)
│       ├── hooks/api/*.ts      # NEW React Query hooks (useResearchRun, useAiChat, ...)
│       ├── types/index.ts      # UNCHANGED — canonical response shapes
│       └── sections/*          # call-site swap only (trpc.* -> use*),  visuals unchanged
├── research/                   # root scripts kept; dup modules deleted, import from src/
└── researcher.py, frontier_sweep.py, build_*.py, run_research.py   # unchanged, share core
```

### 3.1 Backend (FastAPI — `src/strategy_api/`)

- **App factory** (`main.py`): mounts routers under `/api`, enables CORS for the Vite dev origin, and in production serves the built React bundle (`frontend/dist`) via `StaticFiles` with SPA fallback.
- **Config** (`config.py`): `pydantic-settings` loads the existing `.env` (`COINDCX_API_KEY/SECRET`, `OLLAMA_BASE_URL`, `OLLAMA_AGENT_MODEL`, `STRATEGY_BUILDER_MARKET_DATA_SOURCE`). `PYTHON_BIN` becomes obsolete.
- **In-process research**: `services/research.py` imports `strategy_research.ui_research.run_research(...)` and runs it via `anyio.to_thread.run_sync` / `run_in_executor` so the XGBoost/Pandas work doesn't block the event loop. No subprocess, no temp file.
- **DB**: SQLAlchemy models mirroring the three Drizzle tables exactly (same column names, types, defaults). Points at the existing `sqlite.db` so historical rows survive. `Base.metadata.create_all` for fresh installs; existing DB already has the tables.
- **Ollama service**: port `chatCompletion`, `generateStrategy`, `analyzeMarket`, `listModels`, `healthCheck` from `ollama.ts` to `httpx`.
- **Binance service**: port `markPrice`, `ticker24h`, `klines`, `topSymbols` from `market.ts` to `httpx` against `https://fapi.binance.com`.

### 3.2 Endpoint Map (REST mirrors tRPC 1:1)

| tRPC procedure | REST endpoint | Method |
|---|---|---|
| `ping` | `/api/ping` | GET |
| `research.run` | `/api/research/run` | POST |
| `ai.health` | `/api/ai/health` | GET |
| `ai.chat` | `/api/ai/chat` | POST |
| `ai.history` | `/api/ai/history?sessionId=` | GET |
| `ai.generateStrategy` | `/api/ai/generate-strategy` | POST |
| `ai.marketAnalysis` | `/api/ai/market-analysis` | POST |
| `ai.saveSession` | `/api/ai/save-session` | POST |
| `ai.sessions` | `/api/ai/sessions` | GET |
| `ai.insights` | `/api/ai/insights?sessionId=` | GET |
| `ai.clearHistory` | `/api/ai/clear-history` | POST |
| `market.markPrice` | `/api/market/mark-price?symbol=` | GET |
| `market.ticker24h` | `/api/market/ticker-24h?symbol=` | GET |
| `market.klines` | `/api/market/klines?symbol=&interval=&limit=` | GET |
| `market.topSymbols` | `/api/market/top-symbols` | GET |

Request/response field names match the current tRPC payloads so the frontend `types/index.ts` is unchanged. Pydantic request models reproduce the existing zod validation (symbol regex, ranges, defaults).

### 3.3 Frontend (`frontend/`)

- Components, sections, Tailwind, Radix, and `types/index.ts` are **unchanged**.
- Replace `src/providers/trpc.tsx` with `src/lib/api.ts`: a thin typed `fetch` wrapper (base `/api`, JSON, error normalization) returning the existing TS types, with zod parsing at the boundary.
- Add `src/hooks/api/` React Query hooks (the lib is already a dependency) that wrap the fetch client and expose the same call ergonomics the sections use (`mutate`, `data`, `isPending`, `error`). One hook per endpoint.
- Mechanical swap in each of the 12 sections: `trpc.research.run.useMutation()` → `useResearchRun()`, `trpc.ai.chat.useMutation()` → `useAiChat()`, etc.
- `vite.config.ts` dev server proxies `/api` to `http://127.0.0.1:8000`.
- `package.json`: drop `hono`, `@hono/*`, `@trpc/*`, `drizzle-orm`, `drizzle-kit`, `better-sqlite3`, `mysql2`, `superjson`. Keep React, React Query, Radix, Tailwind, zod, recharts, sonner.

### 3.4 Data Flow (after refactor)

```
React section --useResearchRun()--> POST /api/research/run
   -> FastAPI research router (pydantic validate)
   -> services/research.run() --(threadpool)--> strategy_research.ui_research.run_research()
        data (binance REST) -> mtf features -> triple-barrier labels
        -> xgboost train -> strategy gen -> backtest -> walk-forward
   -> returns dict shaped results[rr][side]{strategies, walk_forward, top_features, label_stats}
   -> router maps to ResearchResult JSON (same shape mapResult produced)
   -> React renders StrategyTable / ValidationPanel / FeaturePanel (unchanged)
```

## 4. Consolidation Plan (delete duplicates)

**Canonical:** `src/strategy_research/` for the ML research pipeline; `src/strategy_builder/` for the full manual/AI engine (unchanged).

**Absorb into `src/strategy_research/`** (from `app/public/engine/`): `grid_search.py` → `ui_research.py`, plus `regime.py`, `position_size.py`, `smc_features.py` if not already represented. Reconcile any math differences against the existing canonical modules; the canonical version wins unless the UI output depends on a specific behavior (then keep that behavior and add a test).

**Delete:**
- `app/api/`, `app/db/`, `app/public/engine/`
- `research/mtf_research.py`, `research/barrier_leverage.py`, `research/validation.py` (replace imports with `strategy_research` equivalents)

**Rewire:** `research/run_research.py` and any root script importing the deleted dup modules now import from `src/strategy_research`.

## 5. Error Handling

- Backend: a FastAPI exception handler returns a consistent JSON error `{ error: { code, message } }` mirroring `contracts/errors.ts` semantics. Binance/Ollama timeouts → 502 with a clear message. Validation errors → 422.
- Research failures (insufficient data, training error) return a structured error per RR/side (matching today's `sideData.error` handling) rather than failing the whole run.
- Frontend: the fetch client throws typed errors; sections surface them via the existing `sonner` toasts / inline error states.

## 6. Testing

- **Backend (pytest):**
  - **Golden-shape test** (critical): `ui_research.run_research()` output contains `results[rr][side].{strategies, walk_forward, top_features, label_stats}` with the exact field names the frontend consumes. Guards the contract against drift.
  - Ollama service unit tests with a mocked httpx transport.
  - Binance service unit tests with mocked transport (parse mark price, klines, 24h, top symbols).
  - DB round-trip tests against a temp SQLite (insert/read chat messages, sessions, insights).
  - Endpoint tests via FastAPI `TestClient` for each route (validation + happy path with research mocked).
- **Frontend (vitest):** smoke test the api client + one hook against a mocked fetch.
- **Existing `tests/`** kept and must still pass.

## 7. Dev & Run

- **Dev:** `uvicorn strategy_api.main:app --reload --port 8000` + `cd frontend && npm run dev` (Vite proxies `/api`).
- **Prod:** `npm run build` (frontend) → `uvicorn strategy_api.main:app` serves API + static bundle on one port.
- Add console script `strategy-api = "strategy_api.main:run"` to `pyproject.toml`.

## 8. Migration Order (high level — detailed plan follows in writing-plans)

1. Add backend deps to `pyproject.toml`; scaffold `src/strategy_api/` skeleton (app + ping) and confirm it boots.
2. Build `strategy_research.ui_research` by absorbing `app/public/engine/grid_search.py` etc.; add golden-shape test. Verify parity against current Node output for one symbol.
3. Port Binance service + `/api/market/*`; verify against live REST.
4. Port Ollama service + DB models + `/api/ai/*`; verify chat history persists against existing `sqlite.db`.
5. Implement `/api/research/run` (threadpool) end-to-end; verify JSON matches old `mapResult` output.
6. Frontend: add `lib/api.ts` + `hooks/api/*`; swap call sites section by section; delete `providers/trpc.tsx`.
7. Rename `app/` → `frontend/`; strip Node/drizzle deps; wire Vite proxy + prod StaticFiles.
8. Delete `app/api`, `app/db`, `app/public/engine`, `research/` dup modules; rewire root-script imports.
9. Full run: backend tests, frontend build, manual UI smoke (run research, chat, market panel).

## 9. Risks

- **JSON shape drift** — mitigated by the golden-shape test + a one-symbol parity check against the current Node output before deleting `app/public/engine`.
- **Engine math divergence** during consolidation — reconcile carefully; keep UI-affecting behavior and cover with tests.
- **Long research blocking** — threadpool offload; revisit with SSE if runs exceed comfortable request timeouts.
- **DB column/type mismatch** (Drizzle timestamp ints vs SQLAlchemy) — model timestamps as integer epoch to match existing rows.
