import json
from typing import List, Dict, Any, Optional

class StrategyTemplates:
    TEMPLATES = [
        # 1. Session Breakout Continuation
        {
            "name": "Asia Range Breakout",
            "family": "session_breakout",
            "timeframes": ["15m", "5m", "1m"],
            "session": ["london"],
            "entry": {
                "conditions": ["asia_range_defined", "session_high_break", "volume_confirmation"]
            },
            "exit": {
                "targets": [1.0, 2.0, 3.0],
                "partial_exits": [0.33, 0.33, 0.34],
                "trail": "atr_1_5_after_1R"
            },
            "risk": {
                "stop": "below_asia_low",
                "position_sizing": "fixed_risk_percent",
                "max_risk_percent": 1.0
            },
            "filters": {
                "min_volume_zscore": 1.5,
                "min_atr_percent": 0.4,
                "required_regime": ["normal", "expansion"]
            },
            "invalidation": ["no_volume_confirmation", "failed_breakout", "session_end"],
            "parameter_ranges": {
                "atr_multiplier_stop": [0.5, 2.0],
                "trail_activation_r": [0.8, 1.5],
                "min_range_atr_ratio": [0.5, 1.5]
            }
        },
        # 2. Session Range Mean Reversion
        {
            "name": "Session Range Mean Reversion",
            "family": "session_mean_reversion",
            "timeframes": ["15m", "5m"],
            "session": ["asia", "london", "new_york"],
            "entry": {
                "conditions": ["price_near_session_extreme", "rsi_divergence", "volume_decline"]
            },
            "exit": {
                "targets": [0.5, 1.0],
                "partial_exits": [0.5, 0.5],
                "trail": "none"
            },
            "risk": {
                "stop": "beyond_session_extreme",
                "position_sizing": "fixed_risk_percent",
                "max_risk_percent": 0.75
            },
            "filters": {
                "max_atr_percent": 1.5,
                "required_regime": ["normal", "compression"],
                "min_candles_in_range": 10
            },
            "invalidation": ["breakout_confirmed", "trend_continuation", "high_volume_break"],
            "parameter_ranges": {
                "extreme_threshold_pct": [0.8, 0.95],
                "rsi_divergence_lookback": [5, 20]
            }
        },
        # 3. MTF Trend Pullback
        {
            "name": "MTF Trend Pullback Entry",
            "family": "mtf_pullback",
            "timeframes": ["1h", "15m", "5m"],
            "session": ["london", "new_york", "london_ny"],
            "entry": {
                "conditions": ["higher_tf_trend_bullish", "lower_tf_pullback_to_ema", "structure_hold", "trigger_candle"]
            },
            "exit": {
                "targets": [1.5, 2.5, 4.0],
                "partial_exits": [0.33, 0.33, 0.34],
                "trail": "atr_2_0_after_1_5R"
            },
            "risk": {
                "stop": "below_pullback_low",
                "position_sizing": "fixed_risk_percent",
                "max_risk_percent": 1.0
            },
            "filters": {
                "min_mtf_alignment": 0.4,
                "min_volume_zscore": 1.0,
                "required_structure": ["bullish"]
            },
            "invalidation": ["structure_break", "higher_tf_reversal", "volume_divergence"],
            "parameter_ranges": {
                "ema_period": [10, 30],
                "pullback_depth_atr": [0.5, 1.5],
                "trigger_candle_type": ["engulfing", "pin_bar", "inside_break"]
            }
        },
        # 4. Compression to Expansion Breakout
        {
            "name": "Compression Expansion Breakout",
            "family": "compression_breakout",
            "timeframes": ["4h", "1h", "15m"],
            "session": ["london", "new_york"],
            "entry": {
                "conditions": ["compression_detected", "range_break", "volume_surge", "direction_bias"]
            },
            "exit": {
                "targets": [1.0, 2.0, 3.0],
                "partial_exits": [0.4, 0.3, 0.3],
                "trail": "atr_1_5_after_1R"
            },
            "risk": {
                "stop": "opposite_side_of_compression",
                "position_sizing": "fixed_risk_percent",
                "max_risk_percent": 1.0
            },
            "filters": {
                "max_compression_atr_ratio": 0.6,
                "min_volume_zscore_on_break": 2.0,
                "min_compression_candles": 8
            },
            "invalidation": ["false_breakout", "no_volume_follow_through", "retest_failure"],
            "parameter_ranges": {
                "compression_threshold": [0.4, 0.7],
                "volume_surge_zscore": [1.5, 3.0],
                "min_bars_in_compression": [5, 20]
            }
        },
        # 5. Failed Breakout Reversal
        {
            "name": "Failed Breakout Reversal",
            "family": "failed_breakout",
            "timeframes": ["1h", "15m", "5m"],
            "session": ["asia", "london", "new_york"],
            "entry": {
                "conditions": ["breakout_attempt", "retest_below_level", "rejection_candle", "volume_divergence"]
            },
            "exit": {
                "targets": [1.0, 2.0],
                "partial_exits": [0.5, 0.5],
                "trail": "atr_1_0_after_1R"
            },
            "risk": {
                "stop": "above_false_breakout_high",
                "position_sizing": "fixed_risk_percent",
                "max_risk_percent": 0.75
            },
            "filters": {
                "min_rejection_wick_ratio": 0.6,
                "max_time_above_level_candles": 5,
                "required_volume_pattern": "declining_on_breakout"
            },
            "invalidation": ["sustained_breakout", "volume_confirmation_of_break", "structure_continuation"],
            "parameter_ranges": {
                "rejection_lookback": [3, 10],
                "wick_ratio_threshold": [0.5, 0.8]
            }
        },
        # 6. VWAP/MA Reclaim with Structure
        {
            "name": "VWAP Reclaim Continuation",
            "family": "vwap_reclaim",
            "timeframes": ["15m", "5m", "1m"],
            "session": ["london", "new_york", "london_ny"],
            "entry": {
                "conditions": ["price_reclaims_vwap", "structure_bullish_shift", "volume_on_reclaim", "momentum_confirmation"]
            },
            "exit": {
                "targets": [1.0, 1.5, 2.5],
                "partial_exits": [0.33, 0.33, 0.34],
                "trail": "atr_1_5_after_1R"
            },
            "risk": {
                "stop": "below_reclaim_candle_low",
                "position_sizing": "fixed_risk_percent",
                "max_risk_percent": 1.0
            },
            "filters": {
                "min_volume_zscore": 1.0,
                "required_rsi_range": [40, 60],
                "required_structure_shift": True
            },
            "invalidation": ["immediate_rejection_below_vwap", "no_follow_through", "session_end"],
            "parameter_ranges": {
                "vwap_proximity_pct": [0.1, 0.5],
                "confirmation_candles": [1, 3]
            }
        },
        # 7. Volatility Exhaustion (Mean Reversion)
        {
            "name": "Volatility Exhaustion (Mean Reversion)",
            "family": "session_mean_reversion",
            "timeframes": ["15m", "5m"],
            "entry": {
                "conditions": ["volatility_exhaustion_entry"]
            },
            "exit": {
                "targets": [1.0],
                "partial_exits": [1.0],
                "trail": "none"
            },
            "risk": {
                "stop": "atr_1_5_extreme",
                "position_sizing": "fixed_risk_percent",
                "max_risk_percent": 1.0
            },
            "filters": {
                "min_volume_zscore": 2.0,
                "required_regime": ["mean_reversion", "normal"]
            },
            "invalidation": ["failed_mean_reversion", "trend_breakout"],
            "parameter_ranges": {
                "rsi_extreme_threshold": [15, 25],
                "volume_multiplier": [1.5, 2.5]
            }
        },
        # 8. Liquidity Sweep + MSS
        {
            "name": "Liquidity Sweep + MSS",
            "family": "structure_shift",
            "timeframes": ["15m", "5m"],
            "entry": {
                "conditions": ["liquidity_sweep_mss_entry"]
            },
            "exit": {
                "targets": [2.0],
                "partial_exits": [1.0],
                "trail": "none"
            },
            "risk": {
                "stop": "sweep_wick_extreme",
                "position_sizing": "fixed_risk_percent",
                "max_risk_percent": 1.0
            },
            "filters": {
                "min_volume_zscore": 1.0,
                "required_regime": ["normal", "expansion"]
            },
            "invalidation": ["sweep_extreme_violated", "invalid_mss"],
            "parameter_ranges": {
                "pivot_lookback": [5, 15]
            }
        },
        # 9. Funding Rate Arbitrage
        {
            "name": "Funding Rate Arbitrage",
            "family": "custom",
            "timeframes": ["1h", "15m"],
            "entry": {
                "conditions": ["funding_rate_arbitrage_entry"]
            },
            "exit": {
                "targets": [1.0],
                "partial_exits": [1.0],
                "trail": "none"
            },
            "risk": {
                "stop": "near_zero",
                "position_sizing": "fixed_risk_percent",
                "max_risk_percent": 0.5
            },
            "filters": {
                "min_volume_zscore": 1.5
            },
            "invalidation": ["funding_rate_normalized", "leg_disconnect_risk"],
            "parameter_ranges": {
                "funding_rate_apr_threshold": [10.0, 20.0]
            }
        },
        # 10. VWAP + STD Dev Reversion
        {
            "name": "VWAP + STD Dev Reversion",
            "family": "vwap_reclaim",
            "timeframes": ["15m", "5m"],
            "entry": {
                "conditions": ["vwap_std_dev_reversion_entry"]
            },
            "exit": {
                "targets": [1.5],
                "partial_exits": [1.0],
                "trail": "none"
            },
            "risk": {
                "stop": "atr_1_0",
                "position_sizing": "fixed_risk_percent",
                "max_risk_percent": 1.0
            },
            "filters": {
                "min_volume_zscore": 1.5
            },
            "invalidation": ["vwap_extreme_invalidated"],
            "parameter_ranges": {
                "std_dev_threshold": [2.0, 3.5]
            }
        },
        # 11. Filtered Trend Following (ADX + EMA)
        {
            "name": "Filtered Trend Following",
            "family": "custom",
            "timeframes": ["1h", "15m"],
            "entry": {
                "conditions": ["filtered_trend_following_entry"]
            },
            "exit": {
                "targets": [2.5],
                "partial_exits": [1.0],
                "trail": "atr_2_0_after_1_5R"
            },
            "risk": {
                "stop": "ema_21_cross",
                "position_sizing": "fixed_risk_percent",
                "max_risk_percent": 0.5
            },
            "filters": {
                "min_volume_zscore": 1.0,
                "required_regime": ["trend_expansion", "expansion"]
            },
            "invalidation": ["ema_cross_invalidated", "adx_dropped"],
            "parameter_ranges": {
                "adx_threshold": [20, 30]
            }
        },
        # 12. Opening Range Breakout (ORBO)
        {
            "name": "Opening Range Breakout",
            "family": "session_breakout",
            "timeframes": ["15m", "5m"],
            "entry": {
                "conditions": ["opening_range_breakout_entry"]
            },
            "exit": {
                "targets": [1.5],
                "partial_exits": [1.0],
                "trail": "none"
            },
            "risk": {
                "stop": "opposite_range_extreme",
                "position_sizing": "fixed_risk_percent",
                "max_risk_percent": 1.0
            },
            "filters": {
                "min_volume_zscore": 1.5,
                "required_regime": ["expansion", "trend_expansion"]
            },
            "invalidation": ["range_reentry", "time_stop_hit"],
            "parameter_ranges": {
                "range_duration_minutes": [30, 60]
            }
        },
        # 13. Bollinger Band Walk (Trend Ride)
        {
            "name": "Bollinger Band Walk",
            "family": "custom",
            "timeframes": ["1h", "15m"],
            "entry": {
                "conditions": ["bollinger_band_walk_entry"]
            },
            "exit": {
                "targets": [2.0],
                "partial_exits": [1.0],
                "trail": "middle_band_cross"
            },
            "risk": {
                "stop": "middle_band",
                "position_sizing": "fixed_risk_percent",
                "max_risk_percent": 0.5
            },
            "filters": {
                "min_volume_zscore": 1.0,
                "required_regime": ["trend_expansion", "expansion"]
            },
            "invalidation": ["middle_band_violated", "adx_trend_ended"],
            "parameter_ranges": {
                "adx_threshold": [25, 35]
            }
        },
        # 14. RSI Divergence + Structure
        {
            "name": "RSI Divergence + Structure",
            "family": "failed_breakout",
            "timeframes": ["15m", "5m"],
            "entry": {
                "conditions": ["rsi_divergence_structure_entry"]
            },
            "exit": {
                "targets": [2.0],
                "partial_exits": [1.0],
                "trail": "none"
            },
            "risk": {
                "stop": "atr_1_0_extreme",
                "position_sizing": "fixed_risk_percent",
                "max_risk_percent": 1.0
            },
            "filters": {
                "min_volume_zscore": 1.0
            },
            "invalidation": ["divergence_failed", "structure_reclaimed"],
            "parameter_ranges": {
                "rsi_period": [10, 20]
            }
        },
        # 15. Order Block Mitigation (SMC)
        {
            "name": "Order Block Mitigation",
            "family": "structure_shift",
            "timeframes": ["4h", "1h", "15m"],
            "entry": {
                "conditions": ["order_block_mitigation_entry"]
            },
            "exit": {
                "targets": [2.0],
                "partial_exits": [1.0],
                "trail": "none"
            },
            "risk": {
                "stop": "block_extreme",
                "position_sizing": "fixed_risk_percent",
                "max_risk_percent": 1.0
            },
            "filters": {
                "min_volume_zscore": 1.0
            },
            "invalidation": ["block_invalidated", "structure_reversal"],
            "parameter_ranges": {
                "block_lookback": [20, 100]
            }
        },
        # 16. Delta-Neutral Basis Trade
        {
            "name": "Delta-Neutral Basis Trade",
            "family": "custom",
            "timeframes": ["1h", "15m"],
            "entry": {
                "conditions": ["delta_neutral_basis_entry"]
            },
            "exit": {
                "targets": [1.0],
                "partial_exits": [1.0],
                "trail": "none"
            },
            "risk": {
                "stop": "near_zero",
                "position_sizing": "fixed_risk_percent",
                "max_risk_percent": 0.5
            },
            "filters": {
                "min_volume_zscore": 1.0
            },
            "invalidation": ["basis_converged", "leg_execution_failed"],
            "parameter_ranges": {
                "basis_deviation_std": [2.0, 3.0]
            }
        }
    ]

    @staticmethod
    def all() -> List[Dict[str, Any]]:
        return StrategyTemplates.TEMPLATES

    @staticmethod
    def by_family(family: str) -> List[Dict[str, Any]]:
        return [t for t in StrategyTemplates.TEMPLATES if t["family"] == family]

    @staticmethod
    def families() -> List[str]:
        return list(set(t["family"] for t in StrategyTemplates.TEMPLATES))

    @staticmethod
    def for_regime(regime: Any) -> List[Dict[str, Any]]:
        regime_str = str(regime.value) if hasattr(regime, "value") else str(regime)
        matched = []
        for t in StrategyTemplates.TEMPLATES:
            required = t.get("filters", {}).get("required_regime", [])
            if any(r == regime_str or regime_str.startswith(r.split("_")[0]) for r in required):
                matched.append(t)
        
        return matched if matched else StrategyTemplates.TEMPLATES[:2]

    @staticmethod
    def to_json_catalog() -> str:
        return "\n\n---\n\n".join(json.dumps(t, indent=2) for t in StrategyTemplates.TEMPLATES)
