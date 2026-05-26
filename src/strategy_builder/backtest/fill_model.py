from typing import Dict, Any
from ..domain import Candle

class FillModel:
    FULL_FILL_RATIO = 0.05
    MAX_FILL_PENALTY = 0.70

    def fill(self, price: float, size: float, candle: Candle) -> Dict[str, Any]:
        volume = float(candle.volume)
        if volume <= 0:
            return {"filled_price": price, "filled_size": size, "partial": False}

        volume_ratio = size / volume
        if volume_ratio < self.FULL_FILL_RATIO:
            return {"filled_price": price, "filled_size": size, "partial": False}
        else:
            penalty = min(volume_ratio * 0.3, self.MAX_FILL_PENALTY)
            filled_size = size * (1.0 - penalty)
            filled_size = min(filled_size, size)
            return {"filled_price": price, "filled_size": filled_size, "partial": True}
