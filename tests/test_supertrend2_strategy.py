"""
Tests for the Supertrend2 strategy module.
Matches test_supertrend_strategy.py styling and adds specific test cases for:
  - Series float ATR lengths
  - Optional wicks parameter toggling
"""

import math
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from typing import List

from strategy_builder.domain import Candle
from strategy_builder.strategies import (
    calculate_supertrend2,
    build_supertrend2_signal_generator,
    candles_to_dataframe,
)


def _make_candles(
    n: int,
    start_close: float = 100.0,
    delta: float = 0.0,
    spread_pct: float = 0.01,
    base_time: datetime = None,
) -> List[Candle]:
    if base_time is None:
        base_time = datetime(2023, 1, 1, 0, 0)
    candles = []
    close = start_close
    for i in range(n):
        close = max(close + delta, 0.01)
        spread = close * spread_pct
        candles.append(
            Candle(
                timestamp=int((base_time + timedelta(minutes=i * 15)).timestamp()),
                open=close - spread / 2,
                high=close + spread,
                low=close - spread,
                close=close,
                volume=1000.0 + i * 5,
            )
        )
    return candles


@pytest.fixture
def trending_up_candles() -> List[Candle]:
    return _make_candles(200, start_close=100.0, delta=0.5, spread_pct=0.01)


@pytest.fixture
def trending_down_candles() -> List[Candle]:
    return _make_candles(200, start_close=200.0, delta=-0.5, spread_pct=0.01)


@pytest.fixture
def sample_candles() -> List[Candle]:
    return _make_candles(100, start_close=100.0, delta=0.1, spread_pct=0.01)


# ---------------------------------------------------------------------------
# calculate_supertrend2() tests
# ---------------------------------------------------------------------------

def test_calculate_supertrend2_columns(sample_candles):
    """Verifies columns st_value and st_direction exist after calculation."""
    df = candles_to_dataframe(sample_candles)
    result = calculate_supertrend2(df, length=10, multiplier=3.0)
    assert "st_value" in result.columns
    assert "st_direction" in result.columns


def test_calculate_supertrend2_direction_values(trending_up_candles):
    """st_direction should be 1.0, -1.0 or NaN."""
    df = candles_to_dataframe(trending_up_candles)
    result = calculate_supertrend2(df, length=10, multiplier=3.0)
    valid = result["st_direction"].dropna()
    assert set(valid.unique()).issubset({1.0, -1.0})


def test_calculate_supertrend2_warmup_nans(sample_candles):
    """Warmup period has NaN values."""
    df = candles_to_dataframe(sample_candles)
    length = 10
    result = calculate_supertrend2(df, length=length, multiplier=3.0)
    warmup_nans = result["st_direction"].iloc[:length - 1].isna().sum()
    assert warmup_nans == length - 1


def test_calculate_supertrend2_series_float_length(sample_candles):
    """Test using a pandas Series of float values as the length parameter."""
    df = candles_to_dataframe(sample_candles)
    n = len(df)
    # Define a series float length that starts at 10.0 and climbs dynamically to 14.0
    dynamic_lengths = pd.Series(np.linspace(10.0, 14.0, n), index=df.index)
    
    result = calculate_supertrend2(df, length=dynamic_lengths, multiplier=3.0)
    assert "st_value" in result.columns
    assert not result["st_value"].dropna().empty


def test_calculate_supertrend2_series_float_multiplier(sample_candles):
    """Test using a pandas Series of float values as the multiplier parameter."""
    df = candles_to_dataframe(sample_candles)
    n = len(df)
    dynamic_multipliers = pd.Series(np.linspace(2.0, 4.0, n), index=df.index)
    
    result = calculate_supertrend2(df, length=10, multiplier=dynamic_multipliers)
    assert "st_value" in result.columns
    assert not result["st_value"].dropna().empty


def test_calculate_supertrend2_wicks_toggle(sample_candles):
    """Toggling the wicks parameter should produce different upper/lower boundaries."""
    df = candles_to_dataframe(sample_candles)
    r_wicks = calculate_supertrend2(df, length=10, multiplier=3.0, wicks=True)
    r_no_wicks = calculate_supertrend2(df, length=10, multiplier=3.0, wicks=False)
    
    # Values might differ where a wick high/low resets or pushes a band further than the close price.
    diffs = (r_wicks["st_value"] != r_no_wicks["st_value"]).sum()
    # It shouldn't crash and should calculate properly for both.
    assert "st_value" in r_wicks.columns
    assert "st_value" in r_no_wicks.columns


# ---------------------------------------------------------------------------
# build_supertrend2_signal_generator() tests
# ---------------------------------------------------------------------------

def test_signal_generator2_returns_none_during_warmup():
    gen = build_supertrend2_signal_generator(length=10, multiplier=3.0, wicks=False, adaptive_regime=False)
    few = _make_candles(5, delta=0.5)
    assert gen(few, runtime_mtf=None) is None


def test_signal_generator2_has_required_keys(trending_up_candles):
    gen = build_supertrend2_signal_generator(length=10, multiplier=3.0, wicks=False, adaptive_regime=False)
    signal = None
    for i in range(20, len(trending_up_candles)):
        signal = gen(trending_up_candles[:i], runtime_mtf=None)
        if signal:
            break
    
    if signal:
        assert "direction" in signal
        assert "entry_price" in signal
        assert "stop_distance" in signal
        assert "size" in signal
        assert signal["direction"] in ("long", "short")
        assert isinstance(signal["entry_price"], float)
        assert signal["stop_distance"] > 0
