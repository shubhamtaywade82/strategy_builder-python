# LLM Prompt: Multi-Timeframe (MTF) Crypto Strategy Generation

**Goal:** Create a clean, performant, and robust trading strategy for Python-based trading bots that operates across multiple timeframes (1m, 5m, 15m, 30m, 1h, 4h, 1d) with a >50% win rate and a minimum 1:2 Risk-Reward (RR) ratio.

## Strategy Requirements

1.  **Multi-Timeframe Analysis:**
    *   **Fine-Grain (Execution):** 1-minute (1m) candles.
    *   **Intermediate (Trend/Structure):** 5m, 15m, 30m, 1h.
    *   **High (Macro/Bias):** 4h, 1d.
    *   The strategy MUST use the 1m TF to identify internal price movements and precision entry points, but only when aligned with the macro trend from higher timeframes.

2.  **Entry Conditions (The "Alignment" Rule):**
    *   **Trend Alignment:** Identify the prevailing trend on the 1h, 4h, and 1d timeframes (e.g., using EMAs like 50/200 or RSI > 50).
    *   **Structure Confirmation:** Wait for a market structure shift (MSS) or a breakout on the 5m or 15m timeframe in the direction of the macro trend.
    *   **Execution Trigger:** Entry on the 1m timeframe following a retest of the breakout level or a specific candlestick pattern (e.g., Bullish/Bearish Engulfing) that confirms the continuation of the trend.

3.  **Risk Management & Exits:**
    *   **Risk-Reward Ratio:** 1:2 Minimum.
    *   **Target Profit (TP):** Calculate the TP target such that the net profit EXCLUDES trading fees (e.g., assume a 0.1% round-trip fee and adjust the target price accordingly).
    *   **Stop Loss (SL):** Place SL below/above the recent swing low/high on the 1m or 5m timeframe.
    *   **Partial Exits (Optional):** Consider a 50% exit at 1:1 RR to secure profit, letting the rest run to 1:2 or higher.

4.  **Target Symbols:**
    *   SOLUSDT
    *   XRPUSDT
    *   ETHUSDT
    *   BTCUSDT

## Implementation Context (Python/Backtest Engine)

*   **Data Handling:** Use `ohlcv` data.
*   **Concurrency:** Leverage Python's `asyncio` or `multiprocessing` for parallel instrument scanning and backtest evaluation where applicable.
*   **Performance:** Aim for high capital efficiency and low drawdown.

## Expected Deliverables (For AI)

*   A Python dictionary configuration for the `python_strategy_builder` framework.
*   Custom logic for a `ConditionRegistry` entry that handles the MTF alignment check.
*   Instructions on how to test and validate this strategy using the provided `researcher.py` script.
