from typing import List, Dict, Any, Optional
import numpy as np

class LiquidityMapBuilder:
    EQUAL_LEVEL_PCT = 0.001

    @staticmethod
    def build(features: Dict[str, Any]) -> Dict[str, Any]:
        structure = features.get("structure", {})
        swing_highs = structure.get("swing_highs", [])
        swing_lows = structure.get("swing_lows", [])

        high_prices = [float(s["price"]) for s in swing_highs if s.get("price")]
        low_prices = [float(s["price"]) for s in swing_lows if s.get("price")]

        # Try to get current close from features if available
        current_close = features.get("current_close")
        if current_close is None:
            # Fallback to last candle in momentum or volume profiles if they had it
            # But the Ruby code returns nil here usually
            pass

        return {
            "equal_highs": LiquidityMapBuilder._cluster_levels(high_prices),
            "equal_lows": LiquidityMapBuilder._cluster_levels(low_prices),
            "buy_side_pool": sorted(low_prices)[:3],
            "sell_side_pool": sorted(high_prices, reverse=True)[:3],
            "nearest_support": LiquidityMapBuilder._nearest_level(low_prices, "below", current_close),
            "nearest_resist": LiquidityMapBuilder._nearest_level(high_prices, "above", current_close)
        }

    @staticmethod
    def _cluster_levels(prices: List[float]) -> List[Dict[str, Any]]:
        if not prices:
            return []

        sorted_prices = sorted(prices)
        groups = []
        current_group = [sorted_prices[0]]

        for price in sorted_prices[1:]:
            ref = current_group[0]
            if abs(price - ref) / ref <= LiquidityMapBuilder.EQUAL_LEVEL_PCT:
                current_group.append(price)
            else:
                groups.append({
                    "price": sum(current_group) / len(current_group),
                    "count": len(current_group)
                })
                current_group = [price]
        
        groups.append({
            "price": sum(current_group) / len(current_group),
            "count": len(current_group)
        })

        # Filter for "equal" levels (2 or more points)
        return sorted([g for g in groups if g["count"] >= 2], key=lambda x: x["count"], reverse=True)

    @staticmethod
    def _nearest_level(prices: List[float], side: str, current_close: Optional[float]) -> Optional[float]:
        if not prices or current_close is None or current_close == 0:
            return None

        if side == "below":
            candidates = [p for p in prices if p < current_close]
            return max(candidates) if candidates else None
        elif side == "above":
            candidates = [p for p in prices if p > current_close]
            return min(candidates) if candidates else None
        return None
