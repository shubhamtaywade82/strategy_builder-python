import numpy as np
from typing import List, Dict, Any, Optional
from ..domain import Candle

class VolumeProfile:
    DEFAULT_LOOKBACK = 20

    @staticmethod
    def relative_volume(candles: List[Candle], lookback: int = DEFAULT_LOOKBACK) -> List[Optional[float]]:
        if not candles:
            return []

        volumes = np.array([c.volume for c in candles], dtype=float)
        results = [None] * lookback
        
        for i in range(lookback, len(volumes)):
            baseline = volumes[i - lookback : i]
            avg = np.mean(baseline)
            if avg == 0:
                results.append(0.0)
            else:
                results.append(float(volumes[i] / avg))
        
        return results

    @staticmethod
    def volume_zscore(candles: List[Candle], lookback: int = DEFAULT_LOOKBACK) -> List[Optional[float]]:
        if not candles:
            return []

        volumes = np.array([c.volume for c in candles], dtype=float)
        results = [None] * lookback
        
        for i in range(lookback, len(volumes)):
            window = volumes[i - lookback : i]
            mean = np.mean(window)
            stddev = np.std(window)

            if stddev == 0:
                results.append(0.0)
            else:
                results.append(float((volumes[i] - mean) / stddev))
        
        return results

    @staticmethod
    def burst_detection(candles: List[Candle], lookback: int = DEFAULT_LOOKBACK, threshold: float = 2.0) -> List[Dict[str, Any]]:
        zscores = VolumeProfile.volume_zscore(candles, lookback)
        bursts = []
        for i, candle in enumerate(candles):
            if zscores[i] is not None and zscores[i] >= threshold:
                bursts.append({
                    "index": i,
                    "timestamp": candle.timestamp,
                    "volume": candle.volume,
                    "zscore": zscores[i],
                    "direction": "bullish" if candle.close >= candle.open else "bearish"
                })
        return bursts

    @staticmethod
    def vwap(candles: List[Candle]) -> List[float]:
        if not candles:
            return []
            
        typical_prices = np.array([(c.high + c.low + c.close) / 3.0 for c in candles])
        volumes = np.array([c.volume for c in candles])
        
        tp_v = typical_prices * volumes
        cum_tp_v = np.cumsum(tp_v)
        cum_v = np.cumsum(volumes)
        
        results = []
        for i in range(len(candles)):
            if cum_v[i] == 0:
                results.append(typical_prices[i])
            else:
                results.append(float(cum_tp_v[i] / cum_v[i]))
        return results

    @staticmethod
    def profile(candles: List[Candle], lookback: int = DEFAULT_LOOKBACK) -> Dict[str, Any]:
        rvol = VolumeProfile.relative_volume(candles, lookback)
        zscores = VolumeProfile.volume_zscore(candles, lookback)
        bursts = VolumeProfile.burst_detection(candles, lookback)
        vwaps = VolumeProfile.vwap(candles)

        rvol_curr = [v for v in rvol if v is not None]
        zscore_curr = [v for v in zscores if v is not None]

        return {
            "relative_volume_current": rvol_curr[-1] if rvol_curr else None,
            "volume_zscore_current": zscore_curr[-1] if zscore_curr else None,
            "recent_bursts": bursts[-5:],
            "burst_count_last_20": sum(1 for b in bursts if b["index"] >= len(candles) - 20),
            "vwap_current": vwaps[-1] if vwaps else None
        }
