"""
Supertrend2 Strategy
====================

An alternative Supertrend implementation matching the custom Pine Script `supertrend2` logic.
Supports a "series float" ATR length, dynamic/series multipliers, and a "wicks" parameter 
to optionally use high/low prices instead of close prices for identifying trend reversals.

Mathematical foundation (Pine Script equivalent):
  source        = hl2
  atr           = atr2(atrLength) * factor
  highPrice     = wicks ? high : close
  lowPrice      = wicks ? low  : close
  lowerBand     = ratchet up unless previous lowPrice breached previous lowerBand
  upperBand     = ratchet down unless previous highPrice breached previous upperBand
  direction     = flip uptrend (-1 in Pine, mapped to 1.0 in Python) if highPrice crosses upperBand;
                  flip downtrend (1 in Pine, mapped to -1.0 in Python) if lowPrice crosses lowerBand.
"""

import logging
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..backtest.evaluation_context import EvaluationContext
from ..domain import Candle
from .supertrend import REGIME_MULTIPLIER_SCALE, candles_to_dataframe

logger = logging.getLogger(__name__)


def calculate_supertrend2(
    df: pd.DataFrame,
    length: Union[int, float, pd.Series, np.ndarray],
    multiplier: Union[int, float, pd.Series, np.ndarray],
    wicks: bool = False
) -> pd.DataFrame:
    """
    Compute the custom Supertrend2 bands and direction on a closed-candle DataFrame.

    Supports dynamic/series float length and multiplier.
    If `wicks` is True, uses `high` and `low` to identify trend reversals and reset bands.
    If `wicks` is False (default), uses `close` values for reversals/resets.

    Returns df with two new columns:
      st_value     - the active Supertrend band price level (stop reference)
      st_direction - 1.0 for uptrend (Long), -1.0 for downtrend (Short), NaN during warmup
    """
    df = df.copy()
    n = len(df)

    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)

    # Convert length to float array
    if isinstance(length, (int, float)):
        lengths = np.full(n, float(length))
    elif isinstance(length, pd.Series):
        lengths = length.values.astype(float)
    else:
        lengths = np.asarray(length, dtype=float)

    # Convert multiplier to float array
    if isinstance(multiplier, (int, float)):
        multipliers = np.full(n, float(multiplier))
    elif isinstance(multiplier, pd.Series):
        multipliers = multiplier.values.astype(float)
    else:
        multipliers = np.asarray(multiplier, dtype=float)

    # Calculate True Range
    prev_close = np.empty(n)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )

    # Calculate ATR (Wilder's smoothing/RMA equivalent supporting series float lengths)
    atr = np.full(n, np.nan)
    
    for i in range(n):
        L = lengths[i]
        if np.isnan(L) or L <= 0:
            continue
        
        # Warmup calculation
        idx_warmup = int(np.round(L))
        if idx_warmup < 1:
            idx_warmup = 1

        if i < idx_warmup - 1:
            continue
        elif i == idx_warmup - 1:
            # Initial base case: Simple Moving Average of True Range over L bars
            atr[i] = np.mean(tr[:idx_warmup])
        else:
            if np.isnan(atr[i - 1]):
                if i >= idx_warmup - 1:
                    atr[i] = np.mean(tr[i - idx_warmup + 1 : i + 1])
            else:
                # Standard RMA recursive equation with current length L
                alpha = 1.0 / L
                atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]

    # Calculate bands
    source = (high + low) / 2.0
    upper_band = source + atr * multipliers
    lower_band = source - atr * multipliers

    # High/low prices for wicks calculation
    high_price = high if wicks else close
    low_price = low if wicks else close

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    direction = np.full(n, np.nan)  # 1.0 = Uptrend (Long), -1.0 = Downtrend (Short)
    st_value = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(atr[i]):
            continue

        # Warmup initialization
        if i == 0 or np.isnan(atr[i - 1]) or np.isnan(st_value[i - 1]):
            final_upper[i] = upper_band[i]
            final_lower[i] = lower_band[i]
            direction[i] = 1.0  # Default to uptrend
            st_value[i] = final_lower[i]
            continue

        prev_upper = final_upper[i - 1]
        prev_lower = final_lower[i - 1]
        prev_st_value = st_value[i - 1]
        prev_direction = direction[i - 1]

        # Calculate final lower band
        # lowerBand := lowerBand > prevLowerBand or lowPrice[1] < prevLowerBand ? lowerBand : prevLowerBand
        if lower_band[i] > prev_lower or low_price[i - 1] < prev_lower:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = prev_lower

        # Calculate final upper band
        # upperBand := upperBand < prevUpperBand or highPrice[1] > prevUpperBand ? upperBand : prevUpperBand
        if upper_band[i] < prev_upper or high_price[i - 1] > prev_upper:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = prev_upper

        # Direction calculation:
        # If previous trend was Downtrend (Short, direction == -1.0, active band is upper band):
        # We flip to Uptrend (Long, direction == 1.0) if current high_price crosses upper_band
        if prev_direction == -1.0:
            if high_price[i] > final_upper[i]:
                direction[i] = 1.0
            else:
                direction[i] = -1.0
        # If previous trend was Uptrend (Long, direction == 1.0, active band is lower band):
        # We flip to Downtrend (Short, direction == -1.0) if current low_price drops below lower_band
        else:
            if low_price[i] < final_lower[i]:
                direction[i] = -1.0
            else:
                direction[i] = 1.0

        # Assign active Supertrend line value
        if direction[i] == 1.0:
            st_value[i] = final_lower[i]
        else:
            st_value[i] = final_upper[i]

    df["st_value"] = st_value
    df["st_direction"] = direction
    return df


def build_signal_generator2(
    length: Union[int, float] = 10,
    multiplier: float = 3.0,
    wicks: bool = False,
    adaptive_regime: bool = True,
    window_size: Optional[int] = None,
) -> Callable:
    """
    Return a signal_generator callable compatible with BacktestEngine.run() using Supertrend2 logic.
    """
    _window = window_size or max(int(np.round(length)) * 10, 100)

    def _generate(candles: List[Candle], runtime_mtf=None) -> Optional[Dict]:
        idx_len = int(np.round(length))
        if len(candles) < idx_len + 2:
            return None

        window = candles[-_window:]
        df = candles_to_dataframe(window)

        eff_multiplier = multiplier

        if adaptive_regime:
            try:
                ctx = EvaluationContext(candles, {}, mtf_candles=runtime_mtf)
                regime = ctx.regime()
                scale = REGIME_MULTIPLIER_SCALE.get(regime)
                if scale is None:
                    return None
                eff_multiplier = multiplier * scale
            except Exception:
                pass

        df = calculate_supertrend2(df, length, eff_multiplier, wicks=wicks)

        if len(df) < 2:
            return None

        prev_dir = df["st_direction"].iloc[-2]
        curr_dir = df["st_direction"].iloc[-1]

        if np.isnan(prev_dir) or np.isnan(curr_dir):
            return None

        if prev_dir == curr_dir:
            return None

        entry_price = df["close"].iloc[-1]
        st_band = df["st_value"].iloc[-1]
        stop_distance = abs(entry_price - st_band)

        if stop_distance < entry_price * 0.001:
            return None

        direction = "long" if curr_dir == 1.0 else "short"

        return {
            "direction": direction,
            "entry_price": entry_price,
            "stop_distance": stop_distance,
            "size": 1.0,
        }

    return _generate
