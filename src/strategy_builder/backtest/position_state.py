from typing import Any, Optional
from ..exceptions import BacktestError

class PositionState:
    ACTIVE = "position_open"
    CLOSED = "position_closed"

    @staticmethod
    def is_active(position: Any) -> bool:
        return position and getattr(position, "state", None) == PositionState.ACTIVE

    @staticmethod
    def ensure_active_for_exits(position: Any):
        if not PositionState.is_active(position):
            state = getattr(position, "state", None)
            raise BacktestError(f"Expected position.state == 'position_open' for exit handling, got {state}")
