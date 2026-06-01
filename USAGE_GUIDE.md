# Strategy Builder: Usage & Execution Guide

This guide provides comprehensive instructions for setting up, researching, and running the Python Strategy Builder and Crypto Trading Bot.

---

## 1. Prerequisites

- **Python 3.8+**
- **Ollama** (Optional, for LLM-based strategy research)
- **API Keys**:
  - **CoinDCX**: For trading execution and account balances.
  - **Binance**: Used for high-quality public market data (no keys required for public data, but recommended for higher rate limits).

---

## 2. Installation

1. **Clone the repository and enter the directory**:

   ```bash
   cd python_strategy_builder
   ```

2. **Create a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the package**:

   ```bash
   # Regular installation
   pip install .

   # Development installation (includes testing tools)
   pip install -e ".[dev]"
   ```

---

## 3. Configuration

Create a `.env` file in the root directory to store your credentials and configuration:

```bash
# --- Exchange Credentials ---
COINDCX_API_KEY="your_coindcx_api_key"
COINDCX_API_SECRET="your_coindcx_api_secret"

# --- LLM Configuration (For Research) ---
OLLAMA_BASE_URL="http://127.0.0.1:11434"
OLLAMA_AGENT_MODEL="qwen3.5:4b"

# --- Market Data ---
STRATEGY_BUILDER_MARKET_DATA_SOURCE="binance"
```

---

## 4. Usage Mode 0: Web Dashboard (FastAPI + React)

The dashboard runs the multi-RR XGBoost research pipeline from a browser. It is served by a single Python **FastAPI** backend (`src/strategy_api/`) that imports the research engine in-process — there is no Node backend and no duplicate engine. The React frontend lives in `frontend/` and talks to the backend over REST (`/api/*`).

### Backend dependencies

Already covered by `pip install -e ".[dev]"` (adds `fastapi`, `uvicorn`, `httpx`, `sqlalchemy`, `pydantic-settings`).

### Development (two processes)

```bash
# 1. Start the FastAPI backend on port 8000
venv/bin/uvicorn strategy_api.main:app --reload --port 8000

# 2. In another shell, start the Vite dev server (proxies /api -> :8000)
cd frontend
npm install
npm run dev      # http://localhost:3000
```

Open http://localhost:3000, configure symbol / R:R ratios / leverage / days, and click **Run Research**.

### Production (single process)

```bash
cd frontend && npm run build           # emits frontend/dist
venv/bin/uvicorn strategy_api.main:app --port 8000
```

FastAPI serves the built React bundle and the `/api/*` endpoints from the same port (8000).

### REST endpoints

- `POST /api/research/run` — run the multi-RR grid search (body: `{symbol, days, leverage, horizon, rrs}`)
- `GET  /api/ai/health`, `POST /api/ai/chat`, `POST /api/ai/generate-strategy`, `GET /api/ai/history`, `GET /api/ai/sessions`, `GET /api/ai/insights`, `POST /api/ai/save-session`, `POST /api/ai/market-analysis`, `POST /api/ai/clear-history`
- `GET  /api/market/{mark-price,ticker-24h,klines,top-symbols}`
- `GET  /api/ping`

Chat history, research sessions, and AI insights persist to `sqlite.db` at the repo root.

> **Note:** `OLLAMA_BASE_URL` and `OLLAMA_AGENT_MODEL` from your `.env` drive the AI features. `PYTHON_BIN` is no longer needed (the old Node backend used it to spawn Python — the backend is now Python itself).

---

## 5. Usage Mode 1: Strategy Research (CLI)

The `strategy-builder` CLI is the entry point for the "AI Agent Desk". It uses LLMs to discover market patterns and propose strategies.

### Run the Full Research Pipeline

```bash
strategy-builder pipeline "Research BTC momentum strategies" \
  --instruments B-BTC_USDT \
  --days 14 \
  --timeframes 15m 1h 4h \
  --fresh
```

### Command Reference

| Command | Description |
| :--- | :--- |
| `strategy-builder research "query"` | Run the iterative agent research loop. |
| `strategy-builder discover` | Analyze market features (regime, volatility, structure). |
| `strategy-builder propose` | Generate strategy candidates based on discovered features. |
| `strategy-builder backtest` | Run backtests on all current strategy candidates. |
| `strategy-builder catalog` | List all discovered strategies and their pass/fail status. |
| `strategy-builder templates` | List built-in strategy templates. |

---

## 6. Usage Mode 2: Execution Engine (Bot)

To run the live trading bot that executes orders based on specific technical logic:

### Running the Default Bot

```bash
python examples/crypto_futures_bot.py
```

### Safety Features

- **Sandbox Mode**: By default, the `ExchangeManager` in `crypto_futures_bot.py` uses `sandbox=True`. Change to `False` in the script only for live trading.
- **Risk Guard**: The bot uses a `RiskConfig` class that limits risk per trade (default 1%) and includes a daily loss circuit breaker (default 3%).

### Bot Logic Architecture

1. **Market Data**: Fetches OHLCV from Binance USDT-M Futures.
2. **Indicator Suite**: Calculates Bollinger Bands, RSI, and Volume Anomalies.
3. **Execution**: Maps Binance symbols to CoinDCX pairs (e.g., `BTC/USDT` -> `B-BTC_USDT`) and signs requests with HMAC-SHA256.

---

## 7. Usage Mode 3: Custom Backtesting

For running professional, multi-timeframe backtests on local strategies, use the `examples/crypto_futures_pro.py` template:

```python
from strategy_builder.backtest.engine import BacktestEngine
from strategy_builder.market_data.candle_loader import CandleLoader

# Fetch data
loader = CandleLoader(market_data_source="binance")
mtf_data = loader.fetch_mtf(instrument="BTCUSDT", timeframes=["1h", "15m"])

# Execute backtest
engine = BacktestEngine()
results = engine.run(strategy=my_strategy_config, candles=mtf_data["15m"])
```

---

## 8. Troubleshooting

- **Ollama Connection**: Ensure the Ollama server is running locally (`ollama serve`) before running research commands.
- **API Errors**: If CoinDCX returns `401`, double-check your `X-AUTH-APIKEY` and secret in the `.env` file.
- **Data Gaps**: The `CandleLoader` uses Binance by default. If you experience rate limits, consider using a smaller `--days` value in the CLI.

---

*Note: Trading cryptocurrencies involves significant risk. Always test strategies thoroughly in the backtester and sandbox before live deployment.*
