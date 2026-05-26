import pytest
from datetime import datetime, timedelta
from strategy_builder.domain import Candle

@pytest.fixture
def sample_candles():
    candles = []
    base_time = datetime(2023, 1, 1, 12, 0)
    for i in range(100):
        candles.append(Candle(
            timestamp=int((base_time + timedelta(minutes=i)).timestamp()),
            open=100.0 + i,
            high=105.0 + i,
            low=95.0 + i,
            close=102.0 + i,
            volume=1000.0 + i * 10
        ))
    return candles

@pytest.fixture
def sample_strategy():
    return {
        "name": "Test Strategy",
        "timeframes": ["1h", "15m"]
    }
