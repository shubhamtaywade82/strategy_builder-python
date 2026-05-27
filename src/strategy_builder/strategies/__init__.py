from .supertrend import (
    REGIME_MULTIPLIER_SCALE,
    build_signal_generator as build_supertrend_signal_generator,
    calculate_supertrend,
    candles_to_dataframe,
)
from .supertrend2 import (
    calculate_supertrend2,
    build_signal_generator2 as build_supertrend2_signal_generator,
)

__all__ = [
    "REGIME_MULTIPLIER_SCALE",
    "build_supertrend_signal_generator",
    "calculate_supertrend",
    "candles_to_dataframe",
    "calculate_supertrend2",
    "build_supertrend2_signal_generator",
]
