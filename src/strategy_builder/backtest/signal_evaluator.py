from typing import List, Dict, Any, Optional, Callable
from .condition_registry import ConditionRegistry
from .evaluation_context import EvaluationContext
from ..features.volatility_profile import VolatilityProfile
from ..features.session_detector import SessionDetector
from ..configuration import Configuration
from ..domain import Candle

class SignalEvaluator:
    @staticmethod
    def warmup_bars() -> int:
        return Configuration().backtest_indicator_warmup

    @staticmethod
    def build(strategy: Dict[str, Any], mtf_candles: Optional[Dict[str, List[Candle]]] = None) -> Callable:
        entry = strategy.get("entry", {})
        conditions = entry.get("conditions", [])
        if not conditions:
            conditions = ["generic_breakout"]
        else:
            ConditionRegistry.validate_condition_names(conditions)

        filters = strategy.get("filters", {})
        sessions = strategy.get("session", [])

        def evaluate(candles: List[Candle], runtime_mtf: Optional[Dict[str, List[Candle]]] = None) -> Optional[Dict[str, Any]]:
            if len(candles) < SignalEvaluator.warmup_bars():
                return None

            mtf = runtime_mtf or mtf_candles
            ctx = EvaluationContext(candles, strategy, mtf_candles=mtf)

            passed = all(ConditionRegistry.evaluate(cond, ctx) for cond in conditions)

            if not passed:
                return None
            
            if not ctx.direction or not ctx.entry_price or not ctx.stop_distance:
                return None

            if not SignalEvaluator.passes_filters(ctx, filters):
                return None
            
            if not SignalEvaluator.passes_session_filter(ctx.current_candle, sessions):
                return None

            return {
                "direction": ctx.direction,
                "entry_price": ctx.entry_price,
                "stop_distance": ctx.stop_distance,
                "size": ctx.size
            }

        return evaluate

    @staticmethod
    def passes_filters(ctx: EvaluationContext, filters: Dict[str, Any]) -> bool:
        min_vol_z = filters.get("min_volume_zscore")
        if min_vol_z is not None and ctx.volume_zscore() < min_vol_z:
            return False

        min_atr_pct = filters.get("min_atr_percent")
        if min_atr_pct is not None:
            atr_pct_vals = VolatilityProfile.atr_percent(ctx.candles)
            compact_atr_pct = [v for v in atr_pct_vals if v is not None]
            if compact_atr_pct and compact_atr_pct[-1] < min_atr_pct:
                return False

        max_atr_pct = filters.get("max_atr_percent")
        if max_atr_pct is not None:
            atr_pct_vals = VolatilityProfile.atr_percent(ctx.candles)
            compact_atr_pct = [v for v in atr_pct_vals if v is not None]
            if compact_atr_pct and compact_atr_pct[-1] > max_atr_pct:
                return False

        required_regime = filters.get("required_regime")
        if required_regime:
            if ctx.regime() not in required_regime:
                return False

        required_structure = filters.get("required_structure")
        if required_structure:
            if ctx.structure() not in required_structure:
                return False

        return True

    @staticmethod
    def passes_session_filter(candle: Candle, sessions: List[str]) -> bool:
        if not sessions or "any" in sessions:
            return True

        tagged = SessionDetector.tag_candles([candle])[0]
        candle_sessions = tagged.get("sessions", [])

        return any(s in candle_sessions for s in sessions)
