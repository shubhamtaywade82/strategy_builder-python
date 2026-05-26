from typing import Dict, Any, Optional
from ..domain import Candle

class PartialExitModel:
    def check(self, position: Any, candle: Candle) -> Optional[Dict[str, Any]]:
        targets = getattr(position, "targets", [])
        partials = getattr(position, "partial_exits", [1.0])

        if not targets:
            return None

        for i, target in enumerate(targets):
            fraction = partials[i] if i < len(partials) else (partials[-1] if partials else 1.0)
            
            hit = False
            if position.direction == "long":
                hit = candle.high >= target
            else:
                hit = candle.low <= target

            if not hit:
                continue

            exit_leg_size = position.size * fraction
            exit_leg_size = min(exit_leg_size, position.remaining_size)

            # Update position
            remaining_targets = targets[i + 1:]
            position.targets = remaining_targets
            
            remaining_partials = partials[i + 1:] if i + 1 < len(partials) else []
            position.partial_exits = remaining_partials

            is_full = not remaining_targets or (position.remaining_size - exit_leg_size) <= 1e-9

            if i == 0 and not getattr(position, "be_shifted", False):
                position.current_trail_stop = position.entry_price
                position.be_shifted = True

            return {
                "fraction": fraction,
                "price": target,
                "full_exit": is_full,
                "target_index": i
            }

        return None
