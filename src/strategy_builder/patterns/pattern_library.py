from typing import List, Dict, Any, Optional

class PatternLibrary:
    PATTERNS = {
        "compression_breakout": {
            "required_regime": ["compression"],
            "required_bias": ["long", "short", "neutral"],
            "confirmation": ["volume_surge", "range_break", "direction_bias"],
            "invalidation": ["no volume on break", "immediate reversal below level", "range resumes"],
            "entry_type": "limit",
            "description": "Volatility contraction then expansion with volume"
        },
        "pullback_continuation": {
            "required_regime": ["trend_up", "trend_down"],
            "required_bias": ["long", "short"],
            "confirmation": ["higher_tf_trend_bullish", "lower_tf_pullback_to_ema", "structure_hold", "trigger_candle"],
            "invalidation": ["break of higher swing low", "loss of trend structure", "volume divergence on pullback"],
            "entry_type": "limit",
            "description": "MTF trend pullback to key level then continuation"
        },
        "session_breakout": {
            "required_regime": ["range", "compression"],
            "required_bias": ["long", "short", "neutral"],
            "confirmation": ["asia_range_defined", "session_high_break", "volume_confirmation"],
            "invalidation": ["session ends", "no volume on break", "immediate reversal below level"],
            "entry_type": "limit",
            "description": "Range from prior session broken with volume"
        },
        "liquidity_sweep_reversal": {
            "required_regime": ["range", "expansion"],
            "required_bias": ["long", "short", "neutral"],
            "confirmation": ["rejection_candle", "volume_divergence", "retest_below_level"],
            "invalidation": ["continuation beyond sweep", "no reversal within 3 candles", "increased volume on continuation"],
            "entry_type": "limit",
            "description": "Stop hunt above/below key level then reversal"
        },
        "vwap_reclaim": {
            "required_regime": ["range", "trend_up", "trend_down"],
            "required_bias": ["long", "short", "neutral"],
            "confirmation": ["price_reclaims_vwap", "structure_bullish_shift", "volume_on_reclaim", "momentum_confirmation"],
            "invalidation": ["price falls back under VWAP", "no volume on reclaim", "bearish structure intact"],
            "entry_type": "limit",
            "description": "Price reclaims VWAP after dip with structural shift"
        },
        "failed_breakout_reversal": {
            "required_regime": ["range", "expansion"],
            "required_bias": ["long", "short", "neutral"],
            "confirmation": ["breakout_attempt", "retest_below_level", "rejection_candle", "volume_divergence"],
            "invalidation": ["breakout holds for 3 candles", "no rejection wick", "volume confirms breakout"],
            "entry_type": "limit",
            "description": "Breakout fails and reverses back through level"
        }
    }

    @staticmethod
    def all() -> Dict[str, Any]:
        return PatternLibrary.PATTERNS

    @staticmethod
    def matching(regime: str, bias: str) -> Dict[str, Any]:
        results = {}
        for name, defn in PatternLibrary.PATTERNS.items():
            if regime in defn["required_regime"] and (bias in defn["required_bias"] or bias == "neutral"):
                results[name] = defn
        return results

    @staticmethod
    def names() -> List[str]:
        return list(PatternLibrary.PATTERNS.keys())

    @staticmethod
    def get(name: str) -> Optional[Dict[str, Any]]:
        return PatternLibrary.PATTERNS.get(name)
