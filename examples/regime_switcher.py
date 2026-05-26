"""
Regime-Switching Strategy Demo
==============================

This script demonstrates how to build a "Meta-Strategy" that dynamically 
switches between different trading logics based on the Institutional Market Regime.

Concepts demonstrated:
1. Dynamic Strategy Routing
2. Institutional Regime Detection (Trend Expansion vs. Mean Reversion)
3. Use of Kill Zones (Session Overlaps)
4. Backtesting a multi-strategy system
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from strategy_builder.market_data.candle_loader import CandleLoader
from strategy_builder.backtest.engine import BacktestEngine
from strategy_builder.backtest.signal_evaluator import SignalEvaluator
from strategy_builder.backtest.evaluation_context import EvaluationContext
from strategy_builder.domain import Regime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RegimeSwitcher")

# ============================================================================
# 1. DEFINE SUB-STRATEGIES
# ============================================================================

# Strategy A: Trend Following (Active during high participation)
TREND_STRATEGY = {
    "name": "Aggressive_Trend_Follower",
    "entry": {
        "conditions": [
            "regime_trend_expansion",  # ONLY in trend expansion
            "high_rvol_confirmation",   # Requires high volume
            "generic_breakout"         # Price trigger
        ]
    },
    "exit": {
        "targets": [3.0, 5.0],         # High R:R for trends
        "trail": "atr_2.5"
    }
}

# Strategy B: Mean Reversion (Active during choppy/ranging markets)
RANGE_STRATEGY = {
    "name": "Mean_Reversion_Scalper",
    "entry": {
        "conditions": [
            "regime_mean_reversion",    # ONLY in mean reversion
            "rsi_divergence",          # Standard indicator
            "price_near_session_extreme" # Location trigger
        ]
    },
    "exit": {
        "targets": [1.0, 1.5],         # Lower targets for range
        "time_stop_candles": 20        # Exit if it takes too long
    }
}

# ============================================================================
# 2. DEFINE THE REGIME SWITCHER (META-EVALUATOR)
# ============================================================================

class RegimeSwitchingEvaluator:
    def __init__(self, mtf_candles: Optional[Dict[str, List[Any]]] = None):
        # Build standard evaluators for each sub-strategy
        self.trend_evaluator = SignalEvaluator.build(TREND_STRATEGY, mtf_candles=mtf_candles)
        self.range_evaluator = SignalEvaluator.build(RANGE_STRATEGY, mtf_candles=mtf_candles)
        self.mtf_candles = mtf_candles

    def __call__(self, candles: List[Any], runtime_mtf: Optional[Dict[str, List[Any]]] = None) -> Optional[Dict[str, Any]]:
        """
        The dynamic router: 
        1. Detects the current regime.
        2. Routes to the appropriate strategy logic.
        """
        # We need an EvaluationContext to get the current regime
        # Strategy dict can be empty here as we just need the regime
        ctx = EvaluationContext(candles, {}, mtf_candles=runtime_mtf or self.mtf_candles)
        current_regime = ctx.regime()

        # Dynamic Routing Logic
        if current_regime in ["trend_expansion", "breakout_environment"]:
            logger.debug(f"Routing to TREND strategy (Regime: {current_regime})")
            return self.trend_evaluator(candles, runtime_mtf=runtime_mtf)
        
        elif current_regime in ["mean_reversion", "range"]:
            logger.debug(f"Routing to RANGE strategy (Regime: {current_regime})")
            return self.range_evaluator(candles, runtime_mtf=runtime_mtf)
        
        # Dead market or high risk chaos? Don't trade.
        return None

# ============================================================================
# 3. RUN THE SYSTEM
# ============================================================================

def run_demo():
    symbol = "BTCUSDT"
    
    # In a real scenario, you'd use your API keys
    # loader = CandleLoader(market_data_source="binance")
    # candles = loader.fetch(symbol, "15m", ...)
    
    print("\n" + "="*50)
    print("INSTITUTIONAL REGIME-SWITCHER DEMO")
    print("="*50)
    print(f"Target Symbol: {symbol}")
    print(f"Primary Mode:  DYNAMIC (Trend + Range)")
    print("-" * 50)

    # 1. Setup the Switcher
    # We pass None for mtf_candles in this demo, but you would pass the 1h/4h data here
    switcher = RegimeSwitchingEvaluator()

    # 2. Run Backtest with the Switcher
    engine = BacktestEngine()
    
    # Note: In a real run, you'd pass real candles fetched via CandleLoader
    print("Initialization complete.")
    print("Ready to run backtest with dynamic regime adaptation.")
    print("-" * 50)
    print("Active Strategies:")
    print(f"  - {TREND_STRATEGY['name']} (Condition: Trend Expansion)")
    print(f"  - {RANGE_STRATEGY['name']} (Condition: Mean Reversion)")
    print("\nTo run this for real, ensure your .env has BINANCE/COINDCX keys.")

if __name__ == "__main__":
    run_demo()
