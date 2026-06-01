# Strategy Builder (Python)

A port of the Ruby `strategy_builder` codebase to Python 3.

## Features
- **Web Dashboard:** React UI for the multi-RR XGBoost research pipeline, served by a single Python FastAPI backend (no Node backend, one engine).
- **Multi-Role Agent Desk:** Observer, Pattern Analyst, Trade Designer, and Skeptic roles powered by LLM.
- **Backtest Engine:** Event-driven backtester with slippage, fees, and partial exits.
- **Walk-Forward Analysis:** Robust validation across multiple folds and out-of-sample data.
- **Feature Engine:** Parallel market data discovery and technical feature computation.
- **Strategy Catalog:** Persistent storage and ranking of strategy candidates.

## Architecture

Single Python backend + single React frontend:

- **`src/strategy_api/`** — FastAPI backend. REST under `/api/*` (research, ai, market, ping). Imports the quant engine in-process (no subprocess), persists to SQLite via SQLAlchemy, proxies Binance USD-M REST, talks to Ollama.
- **`src/strategy_research/`** — the one research engine (data → MTF features → triple-barrier labels → XGBoost → backtest → purged walk-forward). `ui_research.run_research()` is the dashboard entry point.
- **`src/strategy_builder/`** — full manual/AI engine + CLI (unchanged).
- **`frontend/`** — React + Vite dashboard. Data layer is plain REST + React Query (`src/lib/api.ts`, `src/hooks/api/`).

## Honest, deterministic research

This engine is built to **not** flatter itself:

- **Out-of-sample, not in-sample.** Model-confidence strategies are backtested on
  purged walk-forward *out-of-sample* predictions (`validation.oos_predict`), never
  the in-sample fit. A model that memorises the training set scores >90% in-sample
  and ~30–50% out-of-sample — only the OOS number is reported as performance.
- **Robustness gate.** `robustness.score_robustness` marks a strategy viable only
  when it is profitable OOS, consistent across folds, significant (shuffle p<0.05),
  and has enough trades. In-sample expectancy ranks nothing. The 5-condition rule
  (`rule_strategy.py`) is additionally checked against a random-entry baseline and a
  bootstrap p-value. One shared `build_signal_mask` powers the backtest, the
  dashboard's Strategy Player, and the live bot, so they can't drift.
- **Deterministic.** Single seed + single-threaded XGBoost ⇒ identical inputs
  produce byte-identical AUCs/p-values/metrics.
- **Never silently faked.** Every result carries `data_source` (`binance` or
  `synthetic`). When live klines are unavailable, set
  `STRATEGY_RESEARCH_ALLOW_SYNTHETIC=1` to run on the deterministic offline
  generator — clearly badged in the UI/report, never passed off as real.

Regenerate the analysis report straight from a real run (no hand-typed numbers):

```bash
venv/bin/python generate_report.py SOLUSDT 60 2:1   # -> docs/strategy_analysis_report/
```

## Web Dashboard

**Dev (two processes):**
```bash
# 1. Backend (port 8000)
venv/bin/uvicorn strategy_api.main:app --reload --port 8000
# 2. Frontend (port 3000, proxies /api -> 8000)
cd frontend && npm install && npm run dev
```
Open http://localhost:3000.

**Production (one process serves API + built UI):**
```bash
cd frontend && npm run build      # emits frontend/dist
venv/bin/uvicorn strategy_api.main:app --port 8000   # serves UI + /api on :8000
```

## Installation

```bash
cd python_strategy_builder
pip install .
```

For development:
```bash
pip install -e ".[dev]"
```

## Documentation

For a detailed guide on installation, configuration, and execution, see the **[USAGE_GUIDE.md](USAGE_GUIDE.md)**.

## Quick Start

1. **Install**: `pip install -e .`
2. **Setup**: Create a `.env` with your `COINDCX` keys and `OLLAMA` settings.
3. **Research**: 
   ```bash
   strategy-builder pipeline "Research BTC breakout" --instruments B-BTC_USDT --days 7
   ```
4. **Trade (Sandbox)**:
   ```bash
   python examples/crypto_futures_bot.py
   ```