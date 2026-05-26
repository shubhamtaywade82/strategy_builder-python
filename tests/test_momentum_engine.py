import pytest
import numpy as np
from strategy_builder.features.momentum_engine import MomentumEngine
from strategy_builder.domain import Candle

def test_ema(sample_candles):
    closes = [c.close for c in sample_candles]
    period = 10
    ema_vals = MomentumEngine.ema(closes, period)
    
    assert len(ema_vals) == len(closes)
    assert ema_vals[period-2] is None
    assert isinstance(ema_vals[period-1], float)
    
    # Simple check: EMA should follow price trend
    assert ema_vals[-1] > ema_vals[period] if closes[-1] > closes[0] else True

def test_rsi(sample_candles):
    rsi_vals = MomentumEngine.rsi(sample_candles, period=14)
    assert len(rsi_vals) == len(sample_candles)
    assert rsi_vals[14] is None # 14 + 1 candles needed for first RSI
    assert isinstance(rsi_vals[15], float)
    assert 0 <= rsi_vals[-1] <= 100

def test_macd(sample_candles):
    macd_data = MomentumEngine.macd(sample_candles)
    assert "macd" in macd_data
    assert "signal" in macd_data
    assert "histogram" in macd_data
    assert len(macd_data["macd"]) == len(sample_candles)

def test_classify_rsi():
    assert MomentumEngine.classify_rsi(75) == "overbought"
    assert MomentumEngine.classify_rsi(25) == "oversold"
    assert MomentumEngine.classify_rsi(65) == "bullish"
    assert MomentumEngine.classify_rsi(35) == "bearish"
    assert MomentumEngine.classify_rsi(50) == "neutral"
    assert MomentumEngine.classify_rsi(None) == "unknown"
