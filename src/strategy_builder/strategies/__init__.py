from .supertrend import (
    REGIME_MULTIPLIER_SCALE,
    build_signal_generator as build_supertrend_signal_generator,
    calculate_supertrend,
    candles_to_dataframe,
)

__all__ = [
    "REGIME_MULTIPLIER_SCALE",
    "build_supertrend_signal_generator",
    "calculate_supertrend",
    "candles_to_dataframe",
]
