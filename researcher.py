"""
Autonomous Crypto Futures Strategy Researcher
============================================

This script is a "Zero-Touch" researcher. 
You provide a symbol, and it:
1. Fetches historical OHLCV and Open Interest (Rate-limit aware).
2. Performs institutional session analytics (RVOL, Efficiency).
3. Detects prevailing Market Regimes.
4. Backtests a suite of professional strategies across MULTIPLE timeframe combos.
5. Reports the best-performing configuration per combo and overall.

Usage:
python researcher.py --symbol BTCUSDT --days 30
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
    },
    {
        "name": "MTF_Trend_Alignment_1m_to_1d",
        "entry": {"conditions": ["mtf_trend_alignment_entry"]},
        "exit": {"targets": [2.0], "trail": None}
    },
    {
        "name": "Advanced_MTF_Trend_Alignment",
        "entry": {"conditions": ["advanced_mtf_trend_alignment_entry"]},
        "exit": {
            # 1.05R to account for ~0.1% fee on a 1:1 trade, 2.1R for 1:2
            "targets": [1.05, 2.1], 
            "partial_exits": [0.5, 1.0], 
            "trail": "atr_2.0"
        }
    },
    {
        "name": "Ignition_Momentum_1Pct_Edge",
        "entry": {"conditions": ["ignition_momentum_continuation_entry"]},
        "exit": {
            # Since min stop is 0.75%, 1.5R target guarantees > 1.1% minimum price move.
            # 2.5R target captures a 1.87% move.
            "targets": [1.5, 2.5], 
            "partial_exits": [0.5, 1.0],
            "trail": "atr_1.5"
        }
    },
    {
        "name": "Institutional_Edge_Sweep_MSS",
        "entry": {"conditions": ["institutional_edge_sweep_mss"]},
        "exit": {
            # 2.0R on a 0.5% stop = 1% move target.
            "targets": [2.0],
            "trail": "atr_2.0"
        }
    }
]

# Timeframe combos to test: (primary_tf, htf_tf, label)
TIMEFRAME_COMBOS = [
    ("15m", "1h", "15m/1h"),
    ("1h", "4h", "1h/4h"),
    ("1m", "1h", "1m/1h_MTF_Full"),
    ("1m", "1d", "1m/1d_Advanced_MTF"),
    ("30m", "4h", "30m/4h_Swing_Edge"),
]


class CryptoFuturesResearcher:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.loader = CandleLoader(market_data_source="binance")
        self.engine = BacktestEngine()

    def run(self, days: int = 30):
        print("\n" + "="*60)
        print(f"AUTONOMOUS RESEARCHER: {self.symbol}")
        print("="*60)

        # --- STEP 1: RATE-LIMIT AWARE DATA FETCH ---
        logger.info(f"Bootstrapping historical data for {self.symbol} ({days} days)...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Fetch all timeframes we might need
        # Explicitly include all TFs for the MTF alignment strategy
        all_tfs = list(dict.fromkeys([
            "1m", "5m", "15m", "30m", "1h", "4h", "1d"
        ] + [tf for combo in TIMEFRAME_COMBOS for tf in combo[:2]]))
        try:
            mtf_data = self.loader.fetch_mtf(
                instrument=self.symbol,
                timeframes=all_tfs,
                start_from=start_date,
                to=end_date
            )
            time.sleep(1.0)
            logger.info("Fetching Open Interest history...")
            oi_data = self.loader.fetch_open_interest(self.symbol, "1h")
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            return

        # --- STEP 2: MULTI-STRATEGY + MULTI-TIMEFRAME EVALUATION ---
        all_results = []  # list of (combo_label, results_list)

        for primary_tf, htf_tf, combo_label in TIMEFRAME_COMBOS:
            if primary_tf not in mtf_data or htf_tf not in mtf_data:
                logger.warning(f"Skipping {combo_label}: data not available")
                continue

            print("-" * 60)
            logger.info(f"Evaluating {len(STRATEGY_LIBRARY)} strategies on {combo_label}...")
            combo_results = []

            for strat_cfg in STRATEGY_LIBRARY:
                logger.info(f"Testing: {strat_cfg['name']} @ {combo_label}...")
                cfg = dict(strat_cfg)
                cfg["timeframes"] = [htf_tf, primary_tf]
                
                evaluator = SignalEvaluator.build(cfg, mtf_candles=mtf_data)
                res = self.engine.run(
                    strategy=cfg,
                    candles=mtf_data[primary_tf],
                    signal_generator=evaluator,
                    mtf_candles=mtf_data
                )
                res["strategy_name"] = cfg["name"]
                res["combo"] = combo_label
                combo_results.append(res)
            
            all_results.append((combo_label, combo_results))

        # --- STEP 3: REPORTING ---
        self._report(all_results)

    def _report(self, all_results):
        print("\n" + "="*60)
        print(f"FINAL RESEARCH REPORT: {self.symbol}")
        print("="*60)

        best_overall = None
        best_pf = -1.0

        for combo_label, results in all_results:
            print(f"\n--- Timeframe: {combo_label} ---")
            print(f"{'Strategy Name':<30} | {'Win%':<6} | {'P.Factor':<8} | {'Trades':<6}")
            print("-" * 60)
            
            sorted_results = sorted(
                results,
                key=lambda x: x['metrics'].get('profit_factor', 0.0),
                reverse=True
            )
            
            for r in sorted_results:
                m = r['metrics']
                win_rate = m.get('win_rate', 0.0) * 100
                profit_factor = m.get('profit_factor', 0.0)
                trade_count = m.get('trade_count', 0)
                print(f"{r['strategy_name']:<30} | {win_rate:>5.1f}% | {profit_factor:>8.2f} | {trade_count:>6}")
                
                if profit_factor > best_pf and trade_count >= 5:
                    best_pf = profit_factor
                    best_overall = r

        print("\n" + "*"*60)
        if best_overall:
            print(f"BEST OVERALL: {best_overall['strategy_name']} @ {best_overall['combo']}")
            print(f"Profit Factor: {best_pf:.2f}")
            print(f"Trades: {best_overall['metrics'].get('trade_count', 0)}")
            print(f"Win Rate: {best_overall['metrics'].get('win_rate', 0.0)*100:.1f}%")
        else:
            print("No strategies produced valid trades.")
        print("*"*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None, help="Crypto futures symbol (e.g. BTCUSDT)")
    parser.add_argument("--days", type=int, default=7, help="Number of days to research")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else ["SOLUSDT", "XRPUSDT", "ETHUSDT", "BTCUSDT"]
    
    for symbol in symbols:
        researcher = CryptoFuturesResearcher(symbol)
        researcher.run(days=args.days)
