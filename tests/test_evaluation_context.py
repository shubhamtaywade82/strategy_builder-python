import pytest
from strategy_builder.backtest.evaluation_context import EvaluationContext

def test_evaluation_context_init(sample_candles, sample_strategy):
    ctx = EvaluationContext(sample_candles, sample_strategy)
    assert ctx.index == 99
    assert ctx.current_candle == sample_candles[-1]
    assert ctx.previous_candle == sample_candles[-2]
    assert ctx.size == 1.0

def test_memoization(sample_candles, sample_strategy):
    ctx = EvaluationContext(sample_candles, sample_strategy)
    atr1 = ctx.atr()
    ctx._memo["atr_14"] = 99.9
    atr2 = ctx.atr()
    assert atr2 == 99.9

def test_mtf_series_up_to(sample_candles, sample_strategy):
    mtf_candles = {
        "1h": [sample_candles[0], sample_candles[50], sample_candles[99]]
    }
    ctx = EvaluationContext(sample_candles, sample_strategy, mtf_candles=mtf_candles)
    
    # current_candle is sample_candles[99]
    series = ctx._mtf_series_up_to("1h")
    assert len(series) == 3
    assert series[-1] == sample_candles[99]

def test_candle_ts(sample_candles, sample_strategy):
    ctx = EvaluationContext(sample_candles, sample_strategy)
    ts = ctx._candle_ts(sample_candles[0])
    assert isinstance(ts, int)
    assert ts == sample_candles[0].get_timestamp_int()
