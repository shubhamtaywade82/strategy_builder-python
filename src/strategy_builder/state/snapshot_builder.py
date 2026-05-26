from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from .regime_classifier import RegimeClassifier
from .liquidity_map_builder import LiquidityMapBuilder
from ..domain import MarketState, Regime, VolatilityState, VolumeState, Bias, LiquidityInfo

class SnapshotBuilder:
    @staticmethod
    def build(instrument: str, features: Dict[str, Any]) -> MarketState:
        regime = RegimeClassifier.classify(features)
        session = SnapshotBuilder._detect_session(features)
        liquidity_data = LiquidityMapBuilder.build(features)
        htf_bias = SnapshotBuilder._higher_tf_bias(features)
        bias = SnapshotBuilder._derive_bias(regime, htf_bias, features)

        liquidity = LiquidityInfo(
            equal_highs=liquidity_data.get("equal_highs", []),
            equal_lows=liquidity_data.get("equal_lows", []),
            nearest_resist=liquidity_data.get("nearest_resist"),
            nearest_support=liquidity_data.get("nearest_support")
        )

        return MarketState(
            instrument=instrument,
            snapshot_at=datetime.now(timezone.utc),
            primary_timeframe=features.get("primary_timeframe", "unknown"),
            regime=regime,
            session=session,
            higher_tf_bias=str(htf_bias.value) if hasattr(htf_bias, "value") else str(htf_bias),
            mid_tf_structure=SnapshotBuilder._mid_tf_structure(features),
            lower_tf_state=SnapshotBuilder._lower_tf_state(features),
            volatility=SnapshotBuilder._volatility_label(features),
            volume=SnapshotBuilder._volume_label(features),
            liquidity=liquidity,
            bias=bias,
            raw_features=features
        )

    @staticmethod
    def _detect_session(features: Dict[str, Any]) -> str:
        sessions = features.get("sessions", [])
        if not sessions:
            return "closed"

        if "london_ny" in sessions: return "london_ny_overlap"
        if "asia_london" in sessions: return "asia_london_overlap"
        if "london" in sessions: return "london"
        if "new_york" in sessions: return "new_york"
        if "asia" in sessions: return "asia"

        return "closed"

    @staticmethod
    def _higher_tf_bias(features: Dict[str, Any]) -> Bias:
        alignment_profile = features.get("mtf_alignment", {}).get("alignment", {})
        if alignment_profile.get("aligned_bullish"):
            return Bias.LONG
        if alignment_profile.get("aligned_bearish"):
            return Bias.SHORT

        regime_label = alignment_profile.get("regime")
        if regime_label in ["strong_bullish", "bullish"]:
            return Bias.LONG
        if regime_label in ["strong_bearish", "bearish"]:
            return Bias.SHORT
        return Bias.NEUTRAL

    @staticmethod
    def _mid_tf_structure(features: Dict[str, Any]) -> str:
        per_tf = features.get("per_timeframe_summary", {})
        if not per_tf:
            return "unknown"

        preferred = ["15m", "30m", "1h", "5m"]
        mid_tf = next((tf for tf in preferred if tf in per_tf), next(iter(per_tf.keys())))
        
        structure = per_tf.get(mid_tf, {}).get("structure")
        if structure == "bullish": return "higher_high_higher_low"
        if structure == "bearish": return "lower_high_lower_low"
        if structure == "ranging": return "ranging"
        return "ranging"

    @staticmethod
    def _lower_tf_state(features: Dict[str, Any]) -> str:
        primary_tf = features.get("primary_timeframe")
        structure = features.get("structure", {}).get("structure")
        per_tf = features.get("per_timeframe_summary", {})

        lower_tf = SnapshotBuilder._lower_timeframe_for(primary_tf, list(per_tf.keys()))
        lower_structure = per_tf.get(lower_tf, {}).get("structure") if lower_tf else None

        if lower_structure in ["bullish", "bearish"]:
            return "trending"
        if lower_structure == "ranging":
            if structure == "bullish": return "pullback_into_support"
            if structure == "bearish": return "at_resistance"
            return "compressing"
        
        if structure in ["bullish", "bearish"]:
            return "trending"
        return "compressing"

    @staticmethod
    def _lower_timeframe_for(primary: str, available: List[str]) -> Optional[str]:
        order = ["1d", "4h", "1h", "30m", "15m", "5m", "3m", "1m"]
        try:
            idx = order.index(primary)
            for tf in order[idx+1:]:
                if tf in available:
                    return tf
        except ValueError:
            pass
        return None

    @staticmethod
    def _volatility_label(features: Dict[str, Any]) -> VolatilityState:
        regime = features.get("volatility", {}).get("regime")
        if regime == "compression": return VolatilityState.CONTRACTING
        if regime == "expansion": return VolatilityState.EXPANDING
        return VolatilityState.NORMAL

    @staticmethod
    def _volume_label(features: Dict[str, Any]) -> VolumeState:
        zscore = features.get("volume", {}).get("volume_zscore_current") or 0.0
        if zscore >= 1.0: return VolumeState.EXPANDING
        if zscore <= -0.5: return VolumeState.DECLINING
        return VolumeState.AVERAGE

    @staticmethod
    def _derive_bias(regime: Regime, htf_bias: Bias, features: Dict[str, Any]) -> Bias:
        structure = features.get("structure", {}).get("structure")

        if regime == Regime.TREND_UP:
            return Bias.LONG
        if regime == Regime.TREND_DOWN:
            return Bias.SHORT
        
        if regime in [Regime.COMPRESSION, Regime.RANGE, Regime.CHOP, Regime.EXPANSION]:
            if htf_bias == Bias.LONG: return Bias.LONG
            if htf_bias == Bias.SHORT: return Bias.SHORT
            
            if structure == "bullish": return Bias.LONG
            if structure == "bearish": return Bias.SHORT
            return Bias.NEUTRAL
        
        return Bias.NEUTRAL
