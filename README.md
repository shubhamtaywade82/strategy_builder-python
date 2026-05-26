# Strategy Builder (Python)

A port of the Ruby `strategy_builder` codebase to Python 3.

## Features
- **Multi-Role Agent Desk:** Observer, Pattern Analyst, Trade Designer, and Skeptic roles powered by LLM.
- **Backtest Engine:** Event-driven backtester with slippage, fees, and partial exits.
- **Walk-Forward Analysis:** Robust validation across multiple folds and out-of-sample data.
- **Feature Engine:** Parallel market data discovery and technical feature computation.
- **Strategy Catalog:** Persistent storage and ranking of strategy candidates.

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