from typing import List, Dict, Any, Optional
from ..domain import Candle

class StructureDetector:
    SWING_LOOKBACK = 5

    @staticmethod
    def swing_points(candles: List[Candle], lookback: int = SWING_LOOKBACK) -> Dict[str, List[Dict[str, Any]]]:
        if len(candles) < (lookback * 2 + 1):
            return {"highs": [], "lows": []}

        highs = []
        lows = []

        for i in range(lookback, len(candles) - lookback):
            left_highs = [c.high for c in candles[i - lookback : i]]
            right_highs = [c.high for c in candles[i + 1 : i + 1 + lookback]]
            current_high = candles[i].high

            if all(h <= current_high for h in left_highs) and all(h <= current_high for h in right_highs):
                highs.append({"index": i, "price": current_high, "timestamp": candles[i].timestamp})

            left_lows = [c.low for c in candles[i - lookback : i]]
            right_lows = [c.low for c in candles[i + 1 : i + 1 + lookback]]
            current_low = candles[i].low

            if all(l >= current_low for l in left_lows) and all(l >= current_low for l in right_lows):
                lows.append({"index": i, "price": current_low, "timestamp": candles[i].timestamp})

        return {"highs": highs, "lows": lows}

    @staticmethod
    def structure(candles: List[Candle], lookback: int = SWING_LOOKBACK) -> str:
        sp = StructureDetector.swing_points(candles, lookback=lookback)
        if len(sp["highs"]) < 2 or len(sp["lows"]) < 2:
            return "unknown"

        last_highs = [h["price"] for h in sp["highs"][-3:]]
        last_lows = [l["price"] for l in sp["lows"][-3:]]

        def all_greater(arr):
            return all(arr[i+1] > arr[i] for i in range(len(arr)-1))

        def all_less(arr):
            return all(arr[i+1] < arr[i] for i in range(len(arr)-1))

        hh = all_greater(last_highs)
        hl = all_greater(last_lows)
        lh = all_less(last_highs)
        ll = all_less(last_lows)

        if hh and hl:
            return "bullish"
        elif lh and ll:
            return "bearish"
        else:
            return "ranging"

    @staticmethod
    def structure_shifts(candles: List[Candle], lookback: int = SWING_LOOKBACK) -> List[Dict[str, Any]]:
        sp = StructureDetector.swing_points(candles, lookback=lookback)
        shifts = []

        for i in range(len(sp["highs"]) - 1):
            prev_high = sp["highs"][i]
            curr_high = sp["highs"][i+1]
            if curr_high["price"] > prev_high["price"]:
                prior_highs = [h for h in sp["highs"] if h["index"] < prev_high["index"]]
                if len(prior_highs) >= 2 and prior_highs[-1]["price"] < prior_highs[-2]["price"]:
                    shifts.append({
                        "type": "bullish_mss",
                        "trigger_index": curr_high["index"],
                        "trigger_price": curr_high["price"],
                        "timestamp": curr_high["timestamp"]
                    })

        for i in range(len(sp["lows"]) - 1):
            prev_low = sp["lows"][i]
            curr_low = sp["lows"][i+1]
            if curr_low["price"] < prev_low["price"]:
                prior_lows = [l for l in sp["lows"] if l["index"] < prev_low["index"]]
                if len(prior_lows) >= 2 and prior_lows[-1]["price"] > prior_lows[-2]["price"]:
                    shifts.append({
                        "type": "bearish_mss",
                        "trigger_index": curr_low["index"],
                        "trigger_price": curr_low["price"],
                        "timestamp": curr_low["timestamp"]
                    })

        shifts.sort(key=lambda s: s["trigger_index"])
        return shifts

    @staticmethod
    def breakout_signals(candles: List[Candle], lookback: int = SWING_LOOKBACK) -> List[Dict[str, Any]]:
        sp = StructureDetector.swing_points(candles, lookback=lookback)
        signals = []

        last_high = sp["highs"][-1] if sp["highs"] else None
        last_low = sp["lows"][-1] if sp["lows"] else None
        current = candles[-1] if candles else None

        if last_high and current and current.close > last_high["price"]:
            signals.append({"type": "breakout_long", "level": last_high["price"], "close": current.close})

        if last_low and current and current.close < last_low["price"]:
            signals.append({"type": "breakout_short", "level": last_low["price"], "close": current.close})

        return signals

    @staticmethod
    def profile(candles: List[Candle], lookback: int = SWING_LOOKBACK) -> Dict[str, Any]:
        sp = StructureDetector.swing_points(candles, lookback=lookback)
        return {
            "structure": StructureDetector.structure(candles, lookback=lookback),
            "swing_highs": sp["highs"][-5:],
            "swing_lows": sp["lows"][-5:],
            "structure_shifts": StructureDetector.structure_shifts(candles, lookback=lookback)[-3:],
            "breakout_signals": StructureDetector.breakout_signals(candles, lookback=lookback)
        }
