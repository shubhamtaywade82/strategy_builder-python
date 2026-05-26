import numpy as np
from typing import List, Dict, Any, Optional, Union
from ..domain import Candle

class MomentumEngine:
    @staticmethod
    def ema(values: Union[List[float], np.ndarray], period: int) -> List[Optional[float]]:
        if len(values) < period:
            return [None] * len(values)

        values = np.array(values, dtype=float)
        multiplier = 2.0 / (period + 1)
        result = [None] * (period - 1)
        
        # Initial SMA
        sma = np.mean(values[0:period])
        result.append(float(sma))

        current_ema = sma
        for val in values[period:]:
            current_ema = (val - current_ema) * multiplier + current_ema
            result.append(float(current_ema))

        return result

    @staticmethod
    def sma(values: Union[List[float], np.ndarray], period: int) -> List[Optional[float]]:
        if len(values) < period:
            return [None] * len(values)

        values = np.array(values, dtype=float)
        result = [None] * (period - 1)
        
        for i in range(len(values) - period + 1):
            window = values[i : i + period]
            result.append(float(np.mean(window)))
            
        return result

    @staticmethod
    def rsi(candles: List[Candle], period: int = 14) -> List[Optional[float]]:
        closes = [c.close for c in candles]
        if len(closes) < period + 1:
            return [None] * len(candles)

        closes = np.array(closes, dtype=float)
        changes = np.diff(closes)

        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, np.abs(changes), 0)

        avg_gain = np.mean(gains[0:period])
        avg_loss = np.mean(losses[0:period])

        result = [None] * (period + 1)

        if avg_loss == 0:
            rsi_val = 100.0 if avg_gain > 0 else 50.0 # Standard handling
            result.append(rsi_val)
        else:
            rs = avg_gain / avg_loss
            result.append(100.0 - (100.0 / (1.0 + rs)))

        for i in range(period, len(changes)):
            gain = gains[i]
            loss = losses[i]

            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

            if avg_loss == 0:
                result.append(100.0)
            else:
                rs = avg_gain / avg_loss
                result.append(100.0 - (100.0 / (1.0 + rs)))

        # Ensure return size matches input size (padding if necessary)
        if len(result) < len(candles):
            result = [None] * (len(candles) - len(result)) + result
        return result[:len(candles)]

    @staticmethod
    def macd(candles: List[Candle], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[Optional[float]]]:
        closes = [c.close for c in candles]
        fast_ema = MomentumEngine.ema(closes, fast)
        slow_ema = MomentumEngine.ema(closes, slow)

        macd_line = []
        for f, s in zip(fast_ema, slow_ema):
            if f is not None and s is not None:
                macd_line.append(f - s)
            else:
                macd_line.append(None)

        compact_macd = [m for m in macd_line if m is not None]
        signal_line_raw = MomentumEngine.ema(compact_macd, signal)

        # Realign signal line
        first_macd_idx = next((i for i, v in enumerate(macd_line) if v is not None), None)
        if first_macd_idx is None:
            return {"macd": macd_line, "signal": [None] * len(candles), "histogram": [None] * len(candles)}

        signal_line = [None] * (first_macd_idx + signal - 1)
        signal_line.extend([s for s in signal_line_raw if s is not None])
        
        # Pad or trim to match candles size
        if len(signal_line) < len(candles):
            signal_line.extend([None] * (len(candles) - len(signal_line)))
        signal_line = signal_line[:len(candles)]

        histogram = []
        for m, s in zip(macd_line, signal_line):
            if m is not None and s is not None:
                histogram.append(m - s)
            else:
                histogram.append(None)

        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }

    @staticmethod
    def roc(candles: List[Candle], period: int = 10) -> List[Optional[float]]:
        closes = [c.close for c in candles]
        if len(closes) < period + 1:
            return [None] * len(candles)

        result = [None] * period
        for i in range(len(closes) - period):
            prev = closes[i]
            curr = closes[i + period]
            if prev == 0:
                result.append(0.0)
            else:
                result.append(((curr - prev) / prev) * 100.0)
        
        return result[:len(candles)]

    @staticmethod
    def ema_slope(candles: List[Candle], period: int = 20, slope_lookback: int = 3) -> List[Optional[float]]:
        closes = [c.close for c in candles]
        ema_vals = MomentumEngine.ema(closes, period)

        result = [None] * (period + slope_lookback - 1)
        compact_ema = [e for e in ema_vals if e is not None]
        
        if len(compact_ema) < slope_lookback + 1:
            return [None] * len(candles)

        for i in range(len(compact_ema) - slope_lookback):
            prev = compact_ema[i]
            curr = compact_ema[i + slope_lookback]
            if prev == 0:
                result.append(0.0)
            else:
                result.append(((curr - prev) / prev) * 100.0)

        if len(result) < len(candles):
            result.extend([None] * (len(candles) - len(result)))
        return result[:len(candles)]

    @staticmethod
    def profile(candles: List[Candle]) -> Dict[str, Any]:
        rsi_vals = MomentumEngine.rsi(candles)
        macd_data = MomentumEngine.macd(candles)
        
        rsi_current = rsi_vals[-1] if rsi_vals else None
        macd_hist = macd_data["histogram"][-1] if macd_data["histogram"] else None
        
        return {
            "rsi_current": rsi_current,
            "rsi_zone": MomentumEngine.classify_rsi(rsi_current),
            "macd_histogram": macd_hist,
            "macd_crossover": MomentumEngine.detect_macd_crossover(macd_data),
            "roc_10": MomentumEngine.roc(candles, 10)[-1] if candles else None,
            "ema_20_slope": MomentumEngine.ema_slope(candles, 20)[-1] if candles else None,
            "ema_50_slope": MomentumEngine.ema_slope(candles, 50)[-1] if candles else None
        }

    @staticmethod
    def classify_rsi(value: Optional[float]) -> str:
        if value is None:
            return "unknown"
        if value >= 70: return "overbought"
        if value <= 30: return "oversold"
        if value >= 60: return "bullish"
        if value <= 40: return "bearish"
        return "neutral"

    @staticmethod
    def detect_macd_crossover(macd_data: Dict[str, List[Optional[float]]]) -> str:
        hist = [h for h in macd_data["histogram"] if h is not None]
        if len(hist) < 2:
            return "none"
        
        if hist[-2] < 0 and hist[-1] >= 0:
            return "bullish_crossover"
        elif hist[-2] > 0 and hist[-1] <= 0:
            return "bearish_crossover"
        else:
            return "none"
