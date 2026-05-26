from typing import List, Optional, Union
from ..domain import Candle
from ..configuration import Configuration
from ..features.volatility_profile import VolatilityProfile

class SlippageModel:
    def __init__(self, slippage_bps: Optional[float] = None, spread_bps: Optional[float] = None, volatility_scaled: Optional[bool] = None, latency_ms: int = 200):
        cfg = Configuration()
        self.slippage_bps = slippage_bps if slippage_bps is not None else cfg.backtest_slippage_bps
        self.spread_bps = spread_bps if spread_bps is not None else cfg.backtest_spread_bps
        self.volatility_scaled = volatility_scaled if volatility_scaled is not None else cfg.backtest_slippage_volatility_scale
        self.latency_ms = latency_ms

    def apply(self, price: float, direction: str, candle: Candle, candles_so_far: Optional[List[Candle]] = None) -> float:
        slip = self._adverse_slippage_amount(price, direction, candle, candles_so_far)
        
        dynamic_spread = self.spread_bps
        if self.volatility_scaled and candles_so_far and len(candles_so_far) > 25:
            atr_pct_vals = VolatilityProfile.atr_percent(candles_so_far)
            compact_atr_pct = [v for v in atr_pct_vals if v is not None]
            if compact_atr_pct and compact_atr_pct[-1] > 2.0:
                dynamic_spread *= 1.5
        
        half_spread = price * (dynamic_spread / 10000.0) / 2.0
        
        latency_penalty = 0.0
        if self.latency_ms > 0 and candle.high > candle.low:
            latency_penalty = (candle.high - candle.low) * 0.01 * (self.latency_ms / 100.0)

        if direction == "long":
            return price + slip + half_spread + latency_penalty
        elif direction == "short":
            return price - slip - half_spread - latency_penalty
        return price

    def _adverse_slippage_amount(self, price: float, direction: str, candle: Candle, candles_so_far: Optional[List[Candle]]) -> float:
        base = price * (self.slippage_bps / 10000.0)
        mult = 1.0
        if self.volatility_scaled and candles_so_far and len(candles_so_far) > 25:
            atr_pct_vals = VolatilityProfile.atr_percent(candles_so_far)
            compact_atr_pct = [v for v in atr_pct_vals if v is not None]
            if compact_atr_pct:
                mult += min(max(float(compact_atr_pct[-1]) / 2.0, 0.0), 2.5)
        return base * mult
