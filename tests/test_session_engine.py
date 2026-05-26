import pytest
from datetime import datetime, timezone
from strategy_builder.domain import Candle
from strategy_builder.analytics.session_engine import SessionAnalyticsEngine

def test_session_analytics_engine_basic():
    engine = SessionAnalyticsEngine()
    
    # Create 24 candles, one for each hour
    candles = []
    base_time = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)
    for i in range(24):
        candles.append(Candle(
            timestamp=base_time.replace(hour=i),
            open=100.0,
            high=102.0,
            low=98.0,
            close=101.0,
            volume=1000.0
        ))
    
    results = engine.analyze(candles)
    
    assert "current_session" in results
    assert "session_metrics" in results
    assert "rvol" in results
    assert results["rvol"] == 1.0 # Since all volumes are identical

def test_session_mapping():
    engine = SessionAnalyticsEngine()
    
    # London (10:00 UTC)
    ts_london = datetime(2023, 1, 1, 10, 0, tzinfo=timezone.utc)
    assert engine._get_current_session(ts_london).value == "london"
    
    # NY Open (15:00 UTC)
    ts_ny = datetime(2023, 1, 1, 15, 0, tzinfo=timezone.utc)
    assert engine._get_current_session(ts_ny).value == "ny_open"
    
    # Overlap (13:30 UTC)
    ts_overlap = datetime(2023, 1, 1, 13, 30, tzinfo=timezone.utc)
    assert engine._get_current_session(ts_overlap).value == "london_ny_overlap"
