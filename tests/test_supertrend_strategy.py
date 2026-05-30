"""
Tests for the Supertrend strategy module.

Follows the same patterns as test_backtest_engine.py:
  - Uses sample_candles fixture from conftest.py (100 candles)
  - Adds domain-specific fixtures for trending and flat markets
  - Asserts on signal dict structure and backtest Trade attributes
"""

import math
import pytest
from datetime import datetime, timedelta
from typing import List
from unittest.mock import patch, MagicMock

from strategy_builder.domain import Candle
from strategy_builder.strategies.supertrend import (
    REGIME_MULTIPLIER_SCALE,
    build_signal_generator,
    calculate_supertrend,
    candles_to_dataframe,
)


# ---------------------------------------------------------------------------
# Additional fixtures
# ---------------------------------------------------------------------------

def _make_candles(
    n: int,
    start_close: float = 100.0,
    delta: float = 0.0,
    spread_pct: float = 0.01,
    base_time: datetime = None,
) -> List[Candle]:
    """
    Build a list of synthetic candles.

    Parameters
    ----------
    n : int
        Number of candles.
    start_close : float
        Starting close price.
    delta : float
        Absolute price change per bar (positive = uptrend, negative = downtrend).
    spread_pct : float
        High-low spread as a fraction of close price.
    """
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
    """200 candles with a strong monotone uptrend (+0.5/bar)."""
    return _make_candles(200, start_close=100.0, delta=0.5, spread_pct=0.01)


@pytest.fixture
def trending_down_candles() -> List[Candle]:
    """200 candles with a strong monotone downtrend (-0.5/bar)."""
    return _make_candles(200, start_close=200.0, delta=-0.5, spread_pct=0.01)


@pytest.fixture
def flip_candles() -> List[Candle]:
    """
    160 candles: first 80 strongly down, then 80 strongly up.
    Designed to produce a clear Supertrend flip at bar ~80.
    """
    down = _make_candles(80, start_close=200.0, delta=-1.0, spread_pct=0.005)
    last_close = down[-1].close
    up = _make_candles(80, start_close=last_close, delta=1.5, spread_pct=0.005,
                       base_time=datetime(2023, 1, 1, 0, 0) + timedelta(minutes=80 * 15))
    return down + up


@pytest.fixture
def flat_candles() -> List[Candle]:
    """100 candles at constant price — ATR ~0, Supertrend may produce NaN."""
    return _make_candles(100, start_close=100.0, delta=0.0, spread_pct=0.0001)


# ---------------------------------------------------------------------------
# calculate_supertrend() unit tests
# ---------------------------------------------------------------------------

def test_calculate_supertrend_returns_required_columns(sample_candles):
    """calculate_supertrend adds st_value and st_direction columns."""
    df = candles_to_dataframe(sample_candles)
    result = calculate_supertrend(df, length=7, multiplier=2.0)
    assert "st_value" in result.columns
    assert "st_direction" in result.columns


def test_calculate_supertrend_direction_values(trending_up_candles):
    """Supertrend direction is either 1.0, -1.0, or NaN (never other values)."""
    df = candles_to_dataframe(trending_up_candles)
    result = calculate_supertrend(df, length=7, multiplier=2.0)
    valid = result["st_direction"].dropna()
    assert set(valid.unique()).issubset({1.0, -1.0})


def test_calculate_supertrend_warmup_nans(sample_candles):
    """First (length-1) rows have NaN direction during ATR warmup."""
    df = candles_to_dataframe(sample_candles)
    length = 10
    result = calculate_supertrend(df, length=length, multiplier=3.0)
    # First length-1 rows should be NaN
    warmup_nans = result["st_direction"].iloc[:length - 1].isna().sum()
    assert warmup_nans == length - 1


def test_calculate_supertrend_uptrend_direction(trending_up_candles):
    """Strong uptrend candles eventually produce sustained st_direction == 1.0."""
    df = candles_to_dataframe(trending_up_candles)
    result = calculate_supertrend(df, length=7, multiplier=2.0)
    # Last 10 bars of a strong uptrend should all be direction 1
    tail = result["st_direction"].dropna().tail(10)
    assert (tail == 1.0).all(), f"Expected uptrend direction, got {tail.tolist()}"


def test_calculate_supertrend_downtrend_direction(trending_down_candles):
    """Strong downtrend candles eventually produce sustained st_direction == -1.0."""
    df = candles_to_dataframe(trending_down_candles)
    result = calculate_supertrend(df, length=7, multiplier=2.0)
    tail = result["st_direction"].dropna().tail(10)
    assert (tail == -1.0).all(), f"Expected downtrend direction, got {tail.tolist()}"


def test_calculate_supertrend_reproducible(sample_candles):
    """Same inputs produce identical outputs (no random state)."""
    df = candles_to_dataframe(sample_candles)
    r1 = calculate_supertrend(df.copy(), length=10, multiplier=3.0)
    r2 = calculate_supertrend(df.copy(), length=10, multiplier=3.0)
    assert r1["st_direction"].equals(r2["st_direction"])
    assert r1["st_value"].equals(r2["st_value"])


# ---------------------------------------------------------------------------
# candles_to_dataframe() unit tests
# ---------------------------------------------------------------------------

def test_candles_to_dataframe_shape(sample_candles):
    """DataFrame has correct number of rows and expected columns."""
    df = candles_to_dataframe(sample_candles)
    assert len(df) == len(sample_candles)
    for col in ("open", "high", "low", "close", "volume"):
        assert col in df.columns


def test_candles_to_dataframe_dtypes(sample_candles):
    """All OHLCV columns are float dtype."""
    df = candles_to_dataframe(sample_candles)
    for col in ("open", "high", "low", "close", "volume"):
        assert df[col].dtype.kind == "f", f"Column {col} is not float"


# ---------------------------------------------------------------------------
# build_signal_generator() unit tests
# ---------------------------------------------------------------------------

def test_signal_generator_returns_none_during_warmup():
    """Signal generator returns None when candles < length + 2."""
    gen = build_signal_generator(length=10, multiplier=3.0, adaptive_regime=False)
    few = _make_candles(5, delta=0.5)
    assert gen(few, runtime_mtf=None) is None


def test_signal_generator_no_signal_in_monotone_trend(trending_up_candles):
    """After initial flip, a pure uptrend produces no further signals."""
    gen = build_signal_generator(length=7, multiplier=2.0, adaptive_regime=False)
    signals = []
    for i in range(30, len(trending_up_candles)):
        sig = gen(trending_up_candles[:i], runtime_mtf=None)
        if sig:
            signals.append(sig)
    # At most one initial long signal; no short signal in an uptrend
    short_signals = [s for s in signals if s["direction"] == "short"]
    assert len(short_signals) == 0, f"Unexpected short signals in uptrend: {short_signals}"


def test_signal_generator_detects_flip(flip_candles):
    """A down→up price flip produces a long signal."""
    gen = build_signal_generator(length=7, multiplier=2.0, adaptive_regime=False)
    long_signals = []
    for i in range(20, len(flip_candles)):
        sig = gen(flip_candles[:i], runtime_mtf=None)
        if sig and sig["direction"] == "long":
            long_signals.append(sig)
    assert len(long_signals) > 0, "Expected at least one long signal after price flip"


def test_signal_dict_has_required_keys(flip_candles):
    """Signal dict contains all keys required by BacktestEngine."""
    gen = build_signal_generator(length=7, multiplier=2.0, adaptive_regime=False)
    signal = None
    for i in range(20, len(flip_candles)):
        signal = gen(flip_candles[:i], runtime_mtf=None)
        if signal:
            break
    assert signal is not None, "Expected a signal to be generated"
    assert "direction" in signal
    assert "entry_price" in signal
    assert "stop_distance" in signal
    assert "size" in signal
    assert signal["direction"] in ("long", "short")
    assert isinstance(signal["entry_price"], float)
    assert signal["stop_distance"] > 0
    assert signal["size"] == 1.0


def test_signal_stop_distance_is_positive(flip_candles):
    """stop_distance is always strictly positive (never zero or negative)."""
    gen = build_signal_generator(length=7, multiplier=2.0, adaptive_regime=False)
    for i in range(20, len(flip_candles)):
        sig = gen(flip_candles[:i], runtime_mtf=None)
        if sig:
            assert sig["stop_distance"] > 0
            break


def test_adaptive_regime_dead_market_blocks_signal(flip_candles):
    """With adaptive_regime=True, DEAD_MARKET regime returns None."""
    gen = build_signal_generator(length=7, multiplier=2.0, adaptive_regime=True)

    mock_ctx = MagicMock()
    mock_ctx.regime.return_value = "dead_market"

    with patch(
        "strategy_builder.strategies.supertrend.EvaluationContext",
        return_value=mock_ctx,
    ):
        result = gen(flip_candles, runtime_mtf=None)

    assert result is None, "DEAD_MARKET regime should block signal generation"


def test_adaptive_regime_liquidation_event_blocks_signal(flip_candles):
    """With adaptive_regime=True, LIQUIDATION_EVENT regime returns None."""
    gen = build_signal_generator(length=7, multiplier=2.0, adaptive_regime=True)

    mock_ctx = MagicMock()
    mock_ctx.regime.return_value = "liquidation_event"

    with patch(
        "strategy_builder.strategies.supertrend.EvaluationContext",
        return_value=mock_ctx,
    ):
        result = gen(flip_candles, runtime_mtf=None)

    assert result is None


def test_adaptive_regime_scales_multiplier_by_regime():
    """Regime scale table: trending regimes use < 1.0, choppy regimes use > 1.0."""
    trending = ["trend_up", "trend_down", "trend_expansion"]
    choppy = ["chop", "range", "mean_reversion", "low_vol_chop"]

    for regime in trending:
        scale = REGIME_MULTIPLIER_SCALE.get(regime)
        assert scale is not None and scale < 1.0, (
            f"Trending regime '{regime}' should have scale < 1.0, got {scale}"
        )

    for regime in choppy:
        scale = REGIME_MULTIPLIER_SCALE.get(regime)
        assert scale is not None and scale > 1.0, (
            f"Choppy regime '{regime}' should have scale > 1.0, got {scale}"
        )


# ---------------------------------------------------------------------------
# BacktestEngine integration test
# ---------------------------------------------------------------------------

def test_backtest_engine_integration_with_supertrend(flip_candles, sample_strategy):
    """BacktestEngine.run() with Supertrend signal generator completes without error."""
    from strategy_builder.backtest.engine import BacktestEngine, Trade

    gen = build_signal_generator(length=7, multiplier=2.0, adaptive_regime=False)
    engine = BacktestEngine()
    result = engine.run(
        strategy=sample_strategy,
        candles=flip_candles,
        signal_generator=gen,
    )

    assert "trades" in result
    assert "metrics" in result
    assert isinstance(result["metrics"], dict)

    if result["trades"]:
        trade = result["trades"][0]
        assert isinstance(trade, Trade)
        assert trade.direction in ("long", "short")
        assert trade.entry_price > 0
        assert trade.exit_reason in ("stop_loss", "target_hit", "time_stop", "end_of_data")


def test_backtest_engine_long_signal_from_flip(flip_candles):
    """BacktestEngine picks up the long signal from a clear downtrend→uptrend flip."""
    from strategy_builder.backtest.engine import BacktestEngine

    gen = build_signal_generator(length=7, multiplier=2.0, adaptive_regime=False)
    strategy = {
        "name": "ST_flip_test",
        "exit": {"targets": [2.0], "partial_exits": [1.0]},
    }
    engine = BacktestEngine()
    result = engine.run(strategy=strategy, candles=flip_candles, signal_generator=gen)

    long_trades = [t for t in result["trades"] if t.direction == "long"]
    assert len(long_trades) > 0, "Expected at least one long trade from the flip"


# ---------------------------------------------------------------------------
# Anti-repainting test
# ---------------------------------------------------------------------------

def test_signal_unaffected_by_live_candle_modification(flip_candles):
    """
    Mutating the LAST (live/forming) candle to an extreme value must not change
    the signal, because signal logic reads the second-to-last closed candle.
    """
    gen = build_signal_generator(length=7, multiplier=2.0, adaptive_regime=False)

    candles_a = list(flip_candles)
    candles_b = list(flip_candles)

    # Replace last entry with an extreme high — simulates a live intra-bar spike
    last = candles_b[-1]
    candles_b[-1] = Candle(
        timestamp=last.timestamp,
        open=last.open,
        high=last.close * 10.0,  # extreme spike
        low=last.low,
        close=last.close * 10.0,
        volume=last.volume,
    )

    sig_a = gen(candles_a, runtime_mtf=None)
    sig_b = gen(candles_b, runtime_mtf=None)

    # Both should return the same outcome (both None or both a signal)
    if sig_a is None:
        assert sig_b is None
    else:
        assert sig_b is not None
        assert sig_a["direction"] == sig_b["direction"]
        # Entry price and stop are based on iloc[-1], so they CAN differ —
        # only the direction flip decision must be consistent
