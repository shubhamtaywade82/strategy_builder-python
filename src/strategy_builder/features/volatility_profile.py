import numpy as np
from typing import List, Dict, Any, Optional
from ..domain import Candle

class VolatilityProfile:
    DEFAULT_ATR_PERIOD = 14
    COMPRESSION_THRESHOLD = 0.6
    EXPANSION_THRESHOLD = 1.5

    @staticmethod
    def atr(candles: List[Candle], period: int = DEFAULT_ATR_PERIOD) -> List[Optional[float]]:
        if len(candles) < 2:
            return [None] * len(candles)

        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])

        tr = np.zeros(len(candles) - 1)
        for i in range(len(candles) - 1):
            tr[i] = max(
                highs[i+1] - lows[i+1],
                abs(highs[i+1] - closes[i]),
                abs(lows[i+1] - closes[i])
            )

        if len(tr) < period:
            return [None] * len(candles)

        atr_values = [None] * period
        first_atr = np.mean(tr[0:period])
        atr_values.append(float(first_atr))

        current_atr = first_atr
        for i in range(period, len(tr)):
            current_atr = (current_atr * (period - 1) + tr[i]) / period
            atr_values.append(float(current_atr))

        return atr_values[:len(candles)]

    @staticmethod
    def atr_percent(candles: List[Candle], period: int = DEFAULT_ATR_PERIOD) -> List[Optional[float]]:
        atr_vals = VolatilityProfile.atr(candles, period)
        results = []
        for i, c in enumerate(candles):
            if atr_vals[i] is None or c.close == 0:
                results.append(None)
            else:
                results.append((atr_vals[i] / c.close) * 100.0)
        return results

    @staticmethod
    def range_expansion(candles: List[Candle], period: int = DEFAULT_ATR_PERIOD) -> List[Optional[float]]:
        atr_vals = VolatilityProfile.atr(candles, period)
        results = []
        for i, c in enumerate(candles):
            if atr_vals[i] is None or atr_vals[i] == 0:
                results.append(None)
            else:
                results.append((c.high - c.low) / atr_vals[i])
        return results

    @staticmethod
    def regime(candles: List[Candle], period: int = DEFAULT_ATR_PERIOD, lookback: int = 50) -> str:
        atr_vals = [v for v in VolatilityProfile.atr(candles, period) if v is not None]
        if len(atr_vals) < lookback:
            return "unknown"

        current_atr = atr_vals[-1]
        rolling_mean = np.mean(atr_vals[-lookback:])

        ratio = current_atr / rolling_mean
        if ratio < VolatilityProfile.COMPRESSION_THRESHOLD:
            return "compression"
        elif ratio > VolatilityProfile.EXPANSION_THRESHOLD:
            return "expansion"
        else:
            return "normal"

    @staticmethod
    def profile(candles: List[Candle], period: int = DEFAULT_ATR_PERIOD) -> Dict[str, Any]:
        atr_vals = VolatilityProfile.atr(candles, period)
        atr_pct = VolatilityProfile.atr_percent(candles, period)

        current_atr = [v for v in atr_vals if v is not None]
        current_atr_val = current_atr[-1] if current_atr else None

        current_atr_pct = [v for v in atr_pct if v is not None]
        current_atr_pct_val = current_atr_pct[-1] if current_atr_pct else None

        re_vals = [v for v in VolatilityProfile.range_expansion(candles, period) if v is not None]

        return {
            "current_atr": current_atr_val,
            "current_atr_percent": current_atr_pct_val,
            "regime": VolatilityProfile.regime(candles, period),
            "range_expansion_last": re_vals[-1] if re_vals else None,
            "atr_series_size": len(current_atr)
        }
