import pytest
from strategy_builder.backtest.engine import BacktestEngine, Position, Trade
from strategy_builder.backtest.signal_evaluator import SignalEvaluator
from strategy_builder.backtest.condition_registry import ConditionRegistry

def test_backtest_engine_run(sample_candles, sample_strategy):
    # Simple signal generator that always goes long on the first available bar after warmup
    def mock_signal_gen(candles_prefix, runtime_mtf=None):
        if len(candles_prefix) == 51: # Warmup is 50
            return {
                "direction": "long",
                "entry_price": candles_prefix[-1].close,
                "stop_distance": 10.0,
                "size": 1.0
            }
        return None

    engine = BacktestEngine()
    result = engine.run(
        strategy=sample_strategy,
        candles=sample_candles,
        signal_generator=mock_signal_gen
    )
    
    assert "trades" in result
    assert "metrics" in result
    assert len(result["trades"]) >= 1
    trade = result["trades"][0]
    assert isinstance(trade, Trade)
    assert trade.direction == "long"

def test_stop_loss_hit(sample_candles, sample_strategy):
    # Signal with a very tight stop loss that should be hit immediately
    def mock_signal_gen(candles_prefix, runtime_mtf=None):
        if len(candles_prefix) == 51:
            return {
                "direction": "long",
                "entry_price": candles_prefix[-1].close,
                "stop_distance": 0.001,
                "size": 1.0
            }
        return None

    engine = BacktestEngine()
    result = engine.run(
        strategy=sample_strategy,
        candles=sample_candles,
        signal_generator=mock_signal_gen
    )
    
    assert len(result["trades"]) >= 1
    trade = result["trades"][0]
    assert trade.exit_reason == "stop_loss"

def test_target_hit(sample_candles, sample_strategy):
    # Signal with a target that should be hit
    def mock_signal_gen(candles_prefix, runtime_mtf=None):
        if len(candles_prefix) == 51:
            return {
                "direction": "long",
                "entry_price": 100.0,
                "stop_distance": 10.0,
                "size": 1.0
            }
        return None

    # Force a high price to hit target
    sample_candles[60].high = 200.0
    
    sample_strategy["exit"] = {"targets": [1.0], "partial_exits": [1.0]}

    engine = BacktestEngine()
    result = engine.run(
        strategy=sample_strategy,
        candles=sample_candles,
        signal_generator=mock_signal_gen
    )
    
    assert len(result["trades"]) >= 1
    trade = result["trades"][0]
    assert trade.exit_reason == "target_hit"
