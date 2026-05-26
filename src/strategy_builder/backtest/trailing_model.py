import re
from typing import Any
from ..domain import Candle

class TrailingModel:
    def update(self, position: Any, candle: Candle) -> Any:
        trail_config = getattr(position, "trail_config", None)
        if not isinstance(trail_config, str) or not trail_config:
            return position

        atr_match = re.match(r"^atr_(\d+)_(\d+)", trail_config)
        if atr_match:
            atr_multiplier = float(f"{atr_match.group(1)}.{atr_match.group(2)}")
            return self._update_atr_trail(position, candle, atr_multiplier)
        
        pct_match = re.match(r"^percent_(\d+)", trail_config)
        if pct_match:
            percent = float(pct_match.group(1)) / 100.0
            return self._update_percent_trail(position, candle, percent)

        return position

    def _update_atr_trail(self, position: Any, candle: Candle, multiplier: float) -> Any:
        atr_proxy = candle.high - candle.low
        trail_distance = atr_proxy * multiplier

        if position.direction == "long":
            new_stop = max(position.current_trail_stop, candle.high - trail_distance)
        else:
            new_stop = min(position.current_trail_stop, candle.low + trail_distance)

        position.current_trail_stop = new_stop
        return position

    def _update_percent_trail(self, position: Any, candle: Candle, percent: float) -> Any:
        trail_distance = candle.close * percent

        if position.direction == "long":
            new_stop = max(position.current_trail_stop, candle.high - trail_distance)
        else:
            new_stop = min(position.current_trail_stop, candle.low + trail_distance)

        position.current_trail_stop = new_stop
        return position
