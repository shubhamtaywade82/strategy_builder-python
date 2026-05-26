"""
Autonomous Crypto Futures Strategy Researcher
============================================

This script is a "Zero-Touch" researcher. 
You provide a symbol, and it:
1. Fetches historical OHLCV and Open Interest (Rate-limit aware).
2. Performs institutional session analytics (RVOL, Efficiency).
3. Detects prevailing Market Regimes.
4. Backtests a suite of professional strategies.
5. Reports the best-performing configuration.

Usage:
python researcher.py --symbol BTCUSDT
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

from strategy_builder.market_data.candle_loader import CandleLoader
from strategy_builder.backtest.engine import BacktestEngine
from strategy_builder.backtest.signal_evaluator import SignalEvaluator

# Setup professional logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Researcher")
# Silence loud loggers
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("pandas").setLevel(logging.WARNING)

# 1. DEFINE A LIBRARY OF PROFESSIONAL CRYPTO FUTURES STRATEGIES
STRATEGY_LIBRARY = [
    {
        "name": "Insto_London_Breakout_V3",
        "entry": {"conditions": ["kill_zone_asia_london", "regime_breakout_environment", "generic_breakout"]},
        "exit": {"targets": [2.0, 4.0], "trail": "atr_2.0"}
    },
    {
        "name": "NY_Momentum_V3",
        "entry": {"conditions": ["kill_zone_london_ny", "regime_trend_expansion", "generic_breakout"]},
        "exit": {"targets": [2.5, 5.0], "trail": "atr_2.5"}
    },
    {
        "name": "Trend_Reversal_Scalper",
        "entry": {"conditions": ["rsi_divergence", "rejection_candle", "generic_breakout"]},
        "exit": {"targets": [1.5, 2.5], "trail": "atr_1.5"}
    },
    {
        "name": "Session_Fade_Pro",
        "entry": {"conditions": ["price_near_session_extreme", "volume_decline", "rejection_candle"]},
        "exit": {"targets": [1.5, 2.0], "time_stop_candles": 30}
    },
    {
        "name": "Optimized_Baseline_2.5R",
        "entry": {"conditions": ["generic_breakout"]},
        "exit": {"targets": [2.5], "trail": "atr_2.0"}
    },
    {
        "name": "Volatility_Exhaustion_MR",
        "entry": {"conditions": ["volatility_exhaustion_entry"]},
        "exit": {"targets": [1.0], "partial_exits": [1.0]}
    },
    {
        "name": "Liquidity_Sweep_MSS",
        "entry": {"conditions": ["liquidity_sweep_mss_entry"]},
        "exit": {"targets": [2.0], "partial_exits": [1.0]}
    },
    {
        "name": "Funding_Rate_Arbitrage",
        "entry": {"conditions": ["funding_rate_arbitrage_entry"]},
        "exit": {"targets": [1.0], "partial_exits": [1.0]}
    },
    {
        "name": "VWAP_STD_Dev_Reversion",
        "entry": {"conditions": ["vwap_std_dev_reversion_entry"]},
        "exit": {"targets": [1.5], "partial_exits": [1.0]}
    },
    {
        "name": "Filtered_Trend_Following",
        "entry": {"conditions": ["filtered_trend_following_entry"]},
        "exit": {"targets": [2.5], "partial_exits": [1.0], "trail": "atr_2.0"}
    },
    {
        "name": "Opening_Range_Breakout",
        "entry": {"conditions": ["opening_range_breakout_entry"]},
        "exit": {"targets": [1.5], "partial_exits": [1.0]}
    },
    {
        "name": "Bollinger_Band_Walk",
        "entry": {"conditions": ["bollinger_band_walk_entry"]},
        "exit": {"targets": [2.0], "partial_exits": [1.0]}
    },
    {
        "name": "RSI_Divergence_Structure",
        "entry": {"conditions": ["rsi_divergence_structure_entry"]},
        "exit": {"targets": [2.0], "partial_exits": [1.0]}
    },
    {
        "name": "Order_Block_Mitigation",
        "entry": {"conditions": ["order_block_mitigation_entry"]},
        "exit": {"targets": [2.0], "partial_exits": [1.0]}
    },
    {
        "name": "Delta_Neutral_Basis_Trade",
        "entry": {"conditions": ["delta_neutral_basis_entry"]},
        "exit": {"targets": [1.0], "partial_exits": [1.0]}
    }
]

class CryptoFuturesResearcher:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.loader = CandleLoader(market_data_source="binance")
        self.engine = BacktestEngine()
        self.results = []

    def run(self, days: int = 30):
        print("\n" + "="*60)
        print(f"AUTONOMOUS RESEARCHER: {self.symbol}")
        print("="*60)

        # --- STEP 1: RATE-LIMIT AWARE DATA FETCH ---
        logger.info(f"Bootstrapping historical data for {self.symbol} ({days} days)...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        try:
            # We fetch 1h (HTF) and 15m (Primary)
            # CandleLoader has built-in sleep/pagination
            mtf_data = self.loader.fetch_mtf(
                instrument=self.symbol,
                timeframes=["1h", "15m"],
                start_from=start_date,
                to=end_date
            )
            
            # Small sleep to respect Binance API weight
            time.sleep(1.0) 
            
            # Fetch Open Interest (Optional confirmation)
            logger.info("Fetching Open Interest history...")
            oi_data = self.loader.fetch_open_interest(self.symbol, "1h")
            
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            return

        # --- STEP 2: MULTI-STRATEGY EVALUATION ---
        print("-" * 60)
        logger.info(f"Beginning Backtest Suite (Evaluating {len(STRATEGY_LIBRARY)} strategies)...")
        
        for strat_cfg in STRATEGY_LIBRARY:
            logger.info(f"Testing: {strat_cfg['name']}...")
            
            # Add timeframes to config for evaluator
            strat_cfg["timeframes"] = ["1h", "15m"]
            
            evaluator = SignalEvaluator.build(strat_cfg, mtf_candles=mtf_data)
            
            res = self.engine.run(
                strategy=strat_cfg,
                candles=mtf_data["15m"],
                signal_generator=evaluator,
                mtf_candles=mtf_data
            )
            
            self.results.append(res)

        # --- STEP 3: REPORTING ---
        self._report()

    def _report(self):
        print("\n" + "="*60)
        print(f"FINAL RESEARCH REPORT: {self.symbol}")
        print("="*60)
        print(f"{'Strategy Name':<30} | {'Win%':<6} | {'P.Factor':<8} | {'Trades':<6}")
        print("-" * 60)
        
        # Access profit_factor as a key
        sorted_results = sorted(self.results, key=lambda x: x['metrics'].get('profit_factor', 0.0), reverse=True)
        
        for r in sorted_results:
            m = r['metrics']
            win_rate = m.get('win_rate', 0.0) * 100
            profit_factor = m.get('profit_factor', 0.0)
            trade_count = m.get('trade_count', 0)
            print(f"{r['strategy_name']:<30} | {win_rate:>5.1f}% | {profit_factor:>8.2f} | {trade_count:>6}")

        if sorted_results:
            best = sorted_results[0]
            print("\n" + "*"*60)
            print(f"RECOMMENDED STRATEGY: {best['strategy_name']}")
            print(f"Profit Factor: {best['metrics'].get('profit_factor', 0.0):.2f}")
            print("*"*60)
        else:
            print("\nNo strategies produced valid trades.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Crypto futures symbol (e.g. BTCUSDT)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to research")
    args = parser.parse_args()

    researcher = CryptoFuturesResearcher(args.symbol)
    researcher.run(days=args.days)
