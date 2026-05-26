"""
Institutional Crypto Futures Strategy: The "Kill-Zone Breakout"
=============================================================

This script provides a complete end-to-end example of building, 
backtesting, and preparing a professional crypto futures strategy.

Strategy Logic:
1. Regime: Must be 'breakout_environment' or 'trend_expansion'.
2. Session: Must be within the London/NY Overlap (Kill Zone).
3. Volume: RVOL must be > 1.4 (Institutional participation).
4. Entry: Break of the previous 1H high/low.
5. Exit: 3:1 Reward-to-Risk ratio with an ATR-based trailing stop.
"""

import os
from datetime import datetime, timedelta
from strategy_builder.market_data.candle_loader import CandleLoader
from strategy_builder.backtest.engine import BacktestEngine
from strategy_builder.backtest.signal_evaluator import SignalEvaluator

# 1. DEFINE THE STRATEGY CONFIGURATION
# This is the "Blueprint" for your strategy.
crypto_futures_strategy = {
    "name": "BTC_Institutional_Breakout",
    "timeframes": ["1h", "15m"],  # Uses 1H for structure, 15m for entry
    "entry": {
        "conditions": [
            "kill_zone_london_ny",        # Filter: Only trade high-liquidity overlap
            "regime_breakout_environment", # Filter: Only trade in expansion phases
            "high_rvol_confirmation",     # Filter: Ensure big players are active
            "session_high_break"          # Trigger: Price breaks a key level
        ]
    },
    "filters": {
        "min_atr_percent": 0.5,           # Filter: Ensure there is enough volatility to move
        "min_volume_zscore": 1.0          # Filter: Confirm volume surge
    },
    "exit": {
        "targets": [2.0, 3.5],            # Take profit at 2R and 3.5R
        "trail": "atr_2.0",               # Trailing stop follows at 2x ATR distance
        "time_stop_candles": 48           # Force exit after 12 hours (48 * 15m)
    }
}

def run_strategy():
    # --- STEP 1: LOAD DATA ---
    # We use the updated CandleLoader for Binance USDT-M Futures
    loader = CandleLoader(market_data_source="binance")
    
    symbol = "BTCUSDT"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30) # Backtest last 30 days

    print(f"[*] Fetching historical data for {symbol}...")
    
    # Fetch Multi-Timeframe (MTF) data
    # This is critical for futures to align execution (15m) with macro bias (1h)
    mtf_data = loader.fetch_mtf(
        instrument=symbol,
        timeframes=["1h", "15m"],
        start_from=start_date,
        to=end_date
    )
    
    primary_candles = mtf_data["15m"]
    
    # --- STEP 2: BUILD EVALUATOR ---
    # The evaluator compiles your JSON rules into executable code
    evaluator = SignalEvaluator.build(crypto_futures_strategy, mtf_candles=mtf_data)

    # --- STEP 3: RUN BACKTEST ---
    print(f"[*] Running backtest for {crypto_futures_strategy['name']}...")
    engine = BacktestEngine()
    
    results = engine.run(
        strategy=crypto_futures_strategy,
        candles=primary_candles,
        signal_generator=evaluator,
        mtf_candles=mtf_data
    )

    # --- STEP 4: ANALYZE PERFORMANCE ---
    metrics = results["metrics"]
    print("\n" + "="*40)
    print(f"STRATEGY: {results['strategy_name']}")
    print("="*40)
    print(f"Total Trades:  {metrics.total_trades}")
    print(f"Win Rate:      {metrics.win_rate:.2f}%")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"Net Profit:    {metrics.net_profit:.2f} USDT")
    print(f"Max Drawdown:  {metrics.max_drawdown_pct:.2f}%")
    print("="*40)

if __name__ == "__main__":
    # Ensure you have your .env file with BINANCE keys if running locally
    # run_strategy()
    print("Professional Strategy Script Created Successfully.")
    print("File: python_strategy_builder/examples/crypto_futures_pro.py")
