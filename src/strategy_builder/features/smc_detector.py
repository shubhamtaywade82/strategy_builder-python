"""
SMC Detector — Smart Money Concepts for strategy_builder-python
================================================================
Adds to: src/strategy_builder/features/smc_detector.py

Integrates with existing FeatureBuilder by adding:
- Fair Value Gaps (FVG)
- Order Blocks (OB)
- Liquidity Sweeps
- Premium/Discount zones

Usage in feature_builder.py:
    from .smc_detector import SmcDetector
    
    # Add to FeatureBuilder.build():
    "smc": SmcDetector.profile(primary_candles),
"""
from typing import List, Dict, Any
from ..domain import Candle


class SmcDetector:
    SWING_LEN = 5

    @staticmethod
    def profile(candles: List[Candle]) -> Dict[str, Any]:
        """Full SMC profile for a candle series."""
        if len(candles) < 10:
            return {"error": "insufficient_candles"}

        fvg = SmcDetector._fair_value_gaps(candles)
        ob = SmcDetector._order_blocks(candles)
        sweeps = SmcDetector._liquidity_sweeps(candles)
        pd_zone = SmcDetector._premium_discount(candles)

        return {
            "fair_value_gaps": fvg,
            "order_blocks": ob,
            "liquidity_sweeps": sweeps,
            "premium_discount": pd_zone,
            "summary": {
                "bullish_fvg_count": sum(1 for x in fvg if x["type"] == "bullish"),
                "bearish_fvg_count": sum(1 for x in fvg if x["type"] == "bearish"),
                "active_bull_ob": sum(1 for x in ob if x["type"] == "bullish" and x["active"]),
                "active_bear_ob": sum(1 for x in ob if x["type"] == "bearish" and x["active"]),
                "recent_sweep_bull": sum(1 for x in sweeps[-20:] if x["type"] == "bullish"),
                "recent_sweep_bear": sum(1 for x in sweeps[-20:] if x["type"] == "bearish"),
                "current_zone": pd_zone.get("zone", "unknown"),
            },
        }

    @staticmethod
    def _fair_value_gaps(candles: List[Candle]) -> List[Dict[str, Any]]:
        """Detect 3-candle FVG imbalances."""
        fvgs = []
        for i in range(2, len(candles)):
            # Bullish FVG: low[i] > high[i-2]
            if candles[i].low > candles[i - 2].high:
                fvgs.append({
                    "index": i,
                    "type": "bullish",
                    "top": candles[i].low,
                    "bottom": candles[i - 2].high,
                    "timestamp": candles[i].timestamp,
                })
            # Bearish FVG: high[i] < low[i-2]
            elif candles[i].high < candles[i - 2].low:
                fvgs.append({
                    "index": i,
                    "type": "bearish",
                    "top": candles[i - 2].low,
                    "bottom": candles[i].high,
                    "timestamp": candles[i].timestamp,
                })
        return fvgs

    @staticmethod
    def _order_blocks(candles: List[Candle]) -> List[Dict[str, Any]]:
        """Detect Order Blocks (last opposing candle before strong move)."""
        obs = []
        for i in range(3, len(candles) - 1):
            c0, c1, c2, c3 = candles[i - 3], candles[i - 2], candles[i - 1], candles[i]
            # Bullish OB: bearish candle before 3+ bullish candles
            if c0.close < c0.open:  # bearish
                if c1.close > c1.open and c2.close > c2.open and c3.close > c3.open:
                    obs.append({
                        "index": i - 3,
                        "type": "bullish",
                        "price": c0.high,
                        "timestamp": c0.timestamp,
                        "active": candles[-1].close < c0.high,
                    })
            # Bearish OB: bullish candle before 3+ bearish candles
            elif c0.close > c0.open:  # bullish
                if c1.close < c1.open and c2.close < c2.open and c3.close < c3.open:
                    obs.append({
                        "index": i - 3,
                        "type": "bearish",
                        "price": c0.low,
                        "timestamp": c0.timestamp,
                        "active": candles[-1].close > c0.low,
                    })
        return obs

    @staticmethod
    def _liquidity_sweeps(candles: List[Candle]) -> List[Dict[str, Any]]:
        """Detect wick sweeps beyond recent swing levels."""
        sweeps = []
        lookback = 20
        for i in range(lookback, len(candles)):
            recent_lows = [c.low for c in candles[i - lookback:i]]
            recent_highs = [c.high for c in candles[i - lookback:i]]
            swing_low = min(recent_lows)
            swing_high = max(recent_highs)

            c = candles[i]
            # Bullish sweep: wick below swing low, close above
            if c.low < swing_low and c.close > swing_low:
                sweeps.append({
                    "index": i,
                    "type": "bullish",
                    "swing_level": swing_low,
                    "wick_depth": (swing_low - c.low) / c.close,
                    "timestamp": c.timestamp,
                })
            # Bearish sweep: wick above swing high, close below
            elif c.high > swing_high and c.close < swing_high:
                sweeps.append({
                    "index": i,
                    "type": "bearish",
                    "swing_level": swing_high,
                    "wick_depth": (c.high - swing_high) / c.close,
                    "timestamp": c.timestamp,
                })
        return sweeps

    @staticmethod
    def _premium_discount(candles: List[Candle], lookback: int = 50) -> Dict[str, Any]:
        """Position within recent range."""
        if len(candles) < lookback:
            return {"zone": "unknown"}

        recent = candles[-lookback:]
        range_high = max(c.high for c in recent)
        range_low = min(c.low for c in recent)
        current = candles[-1].close

        if range_high == range_low:
            return {"zone": "equilibrium"}

        position = (current - range_low) / (range_high - range_low)

        if position > 0.7:
            zone = "premium"
        elif position < 0.3:
            zone = "discount"
        else:
            zone = "equilibrium"

        return {
            "zone": zone,
            "position": round(position, 3),
            "range_high": range_high,
            "range_low": range_low,
            "range_70": range_low + (range_high - range_low) * 0.7,
            "range_30": range_low + (range_high - range_low) * 0.3,
        }
