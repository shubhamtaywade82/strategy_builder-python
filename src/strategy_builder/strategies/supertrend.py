"""
Supertrend Strategy
===================

Vectorized Supertrend indicator implemented in pure pandas/numpy.
Integrates with BacktestEngine via a signal_generator callable.

Mathematical foundation:
  ATR(n)        = Wilder's RMA of True Range over n periods
  Basic Upper   = (H + L) / 2 + multiplier * ATR
  Basic Lower   = (H + L) / 2 - multiplier * ATR
  Final bands   = step-function that only moves in the trend direction
  Direction     = 1 (uptrend) when close > lower band; -1 (downtrend) otherwise

Regime adaptation: in choppy/ranging markets the multiplier is widened to
suppress noise; in trending markets it is tightened to stay responsive.
"""

import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..backtest.evaluation_context import EvaluationContext
from ..domain import Candle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime-based multiplier scaling
# None means "skip trading entirely in this regime"
# ---------------------------------------------------------------------------
REGIME_MULTIPLIER_SCALE: Dict[str, Optional[float]] = {
    "trend_expansion":       0.75,
    "trend_up":              0.80,
    "trend_down":            0.80,
    "breakout_environment":  0.85,
    "accumulation":          1.10,
    "compression":           1.20,
    "chop":                  1.30,
    "mean_reversion":        1.30,
    "range":                 1.35,
    "low_vol_chop":          1.40,
    "dead_market":           None,
    "liquidation_event":     None,
    "high_risk_chaos":       None,
}


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------

def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's Average True Range (RMA smoothing)."""
    n = len(close)
    prev_close = np.empty(n)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )

    atr = np.full(n, np.nan)
    if n < period:
        return atr

    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def calculate_supertrend(df: pd.DataFrame, length: int, multiplier: float) -> pd.DataFrame:
    """
    Compute Supertrend bands and direction on a closed-candle DataFrame.

    Returns df with two new columns:
      st_value     — the active Supertrend band price level (stop reference)
      st_direction — 1 for uptrend, -1 for downtrend, NaN during warmup
    """
    df = df.copy()
    n = len(df)

    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)

    atr = _wilder_atr(high, low, close, length)
    hl2 = (high + low) / 2.0

    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    direction = np.full(n, np.nan)
    st_value = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(atr[i]):
            continue

        if i == 0 or np.isnan(direction[i - 1]):
            final_upper[i] = basic_upper[i]
            final_lower[i] = basic_lower[i]
            direction[i] = 1.0
            st_value[i] = final_lower[i]
            continue

        # Upper band: only lower if price is already below it (ratchet up)
        final_upper[i] = (
            basic_upper[i]
            if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]
            else final_upper[i - 1]
        )
        # Lower band: only raise if price is already above it (ratchet down)
        final_lower[i] = (
            basic_lower[i]
            if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]
            else final_lower[i - 1]
        )

        prev_dir = direction[i - 1]
        if prev_dir == 1.0:
            if close[i] < final_lower[i]:
                direction[i] = -1.0
                st_value[i] = final_upper[i]
            else:
                direction[i] = 1.0
                st_value[i] = final_lower[i]
        else:
            if close[i] > final_upper[i]:
                direction[i] = 1.0
                st_value[i] = final_lower[i]
            else:
                direction[i] = -1.0
                st_value[i] = final_upper[i]

    df["st_value"] = st_value
    df["st_direction"] = direction
    return df


# ---------------------------------------------------------------------------
# Candle ↔ DataFrame conversion
# ---------------------------------------------------------------------------

def candles_to_dataframe(candles: List[Candle]) -> pd.DataFrame:
    """Convert a List[Candle] domain objects to a pandas DataFrame."""
    rows = [
        {
            "timestamp": c.get_timestamp_int(),
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
            "volume": float(c.volume),
        }
        for c in candles
    ]
    df = pd.DataFrame(rows)
    df.set_index("timestamp", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Signal generator factory (BacktestEngine-compatible)
# ---------------------------------------------------------------------------

def build_signal_generator(
    length: int = 10,
    multiplier: float = 3.0,
    adaptive_regime: bool = True,
    window_size: Optional[int] = None,
) -> Callable:
    """
    Return a signal_generator callable compatible with BacktestEngine.run().

    The callable processes only the last `window_size` candles to bound memory
    and recomputation cost.  On each call it:
      1. Converts List[Candle] → DataFrame
      2. Optionally adjusts the multiplier based on current market regime
      3. Computes Supertrend on the rolling window
      4. Returns a signal dict on a direction flip, else None

    No signal is ever generated on an unclosed candle — the candle list
    passed by BacktestEngine always contains fully-closed bars.
    """
    _window = window_size or max(length * 10, 100)

    def _generate(candles: List[Candle], runtime_mtf=None) -> Optional[Dict]:
        if len(candles) < length + 2:
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

        df = calculate_supertrend(df, length, eff_multiplier)

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
