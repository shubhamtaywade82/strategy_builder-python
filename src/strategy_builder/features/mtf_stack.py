from typing import List, Dict, Any, Optional
from .momentum_engine import MomentumEngine
from .structure_detector import StructureDetector
from .volatility_profile import VolatilityProfile
from ..domain import Candle
from ..configuration import Configuration

class MtfStack:
    @staticmethod
    def alignment(mtf_candles: Dict[str, List[Candle]], ema_period: int = 20) -> Dict[str, Any]:
        scores = {}

        for tf, candles in mtf_candles.items():
            if len(candles) < ema_period + 5:
                continue

            closes = [c.close for c in candles]
            ema_vals = MomentumEngine.ema(closes, period=ema_period)
            current_close = closes[-1]
            compact_ema = [e for e in ema_vals if e is not None]
            
            if not compact_ema:
                continue
            
            current_ema = compact_ema[-1]
            price_vs_ema = 1 if current_close > current_ema else -1

            ema_direction = 0
            if len(compact_ema) >= 3:
                slope = compact_ema[-1] - compact_ema[-3]
                ema_direction = 1 if slope > 0 else -1
            
            scores[tf] = (price_vs_ema + ema_direction) / 2.0

        if not scores:
            return {"alignment": 0.0, "scores": {}, "regime": "unknown"}

        alignment_score = sum(scores.values()) / len(scores)

        return {
            "alignment": round(alignment_score, 3),
            "scores": scores,
            "regime": MtfStack.classify_alignment(alignment_score),
            "aligned_bullish": all(s > 0 for s in scores.values()),
            "aligned_bearish": all(s < 0 for s in scores.values()),
            "conflicting": any(s > 0 for s in scores.values()) and any(s < 0 for s in scores.values())
        }

    @staticmethod
    def pullback_opportunities(mtf_candles: Dict[str, List[Candle]], higher_tf: str, lower_tf: str, ema_period: int = 20) -> Optional[Dict[str, Any]]:
        higher = mtf_candles.get(higher_tf)
        lower = mtf_candles.get(lower_tf)
        if not higher or not lower:
            return None

        higher_trend = MtfStack.trend_direction(higher, ema_period=ema_period)
        lower_trend = MtfStack.trend_direction(lower, ema_period=ema_period)

        if higher_trend == "bullish" and lower_trend == "bearish":
            return {"type": "bullish_pullback", "higher_tf": higher_tf, "lower_tf": lower_tf}
        elif higher_trend == "bearish" and lower_trend == "bullish":
            return {"type": "bearish_pullback", "higher_tf": higher_tf, "lower_tf": lower_tf}
        return None

    @staticmethod
    def trend_direction(candles: List[Candle], ema_period: int = 20) -> str:
        if len(candles) < ema_period + 5:
            return "unknown"

        closes = [c.close for c in candles]
        ema_vals = MomentumEngine.ema(closes, period=ema_period)
        current = closes[-1]
        compact_ema = [e for e in ema_vals if e is not None]

        if not compact_ema:
            return "unknown"

        return "bullish" if current > compact_ema[-1] else "bearish"

    @staticmethod
    def sorted_mtf_keys(keys: List[str]) -> List[str]:
        rank = {str(tf): i for i, tf in enumerate(Configuration.VALID_TIMEFRAMES)}
        return sorted(keys, key=lambda k: rank.get(str(k), 999), reverse=True)

    @staticmethod
    def profile(mtf_candles: Dict[str, List[Candle]]) -> Dict[str, Any]:
        align = MtfStack.alignment(mtf_candles)
        pullbacks = []
        sorted_keys = MtfStack.sorted_mtf_keys(list(mtf_candles.keys()))
        
        for i in range(len(sorted_keys) - 1):
            higher_tf = sorted_keys[i]
            lower_tf = sorted_keys[i+1]
            pb = MtfStack.pullback_opportunities(mtf_candles, higher_tf=higher_tf, lower_tf=lower_tf)
            if pb:
                pullbacks.append(pb)

        per_tf = {}
        for tf, candles in mtf_candles.items():
            if len(candles) < 25:
                per_tf[tf] = {"trend": "unknown"}
                continue
            
            rsi_vals = MomentumEngine.rsi(candles)
            compact_rsi = [r for r in rsi_vals if r is not None]
            
            per_tf[tf] = {
                "trend": MtfStack.trend_direction(candles),
                "structure": StructureDetector.structure(candles),
                "volatility_regime": VolatilityProfile.regime(candles),
                "rsi": round(compact_rsi[-1], 1) if compact_rsi else None
            }

        return {
            "alignment": align,
            "pullback_opportunities": pullbacks,
            "per_timeframe": per_tf
        }

    @staticmethod
    def classify_alignment(score: float) -> str:
        if score > 0.6: return "strong_bullish"
        if score > 0.2: return "bullish"
        if score < -0.6: return "strong_bearish"
        if score < -0.2: return "bearish"
        return "neutral"
