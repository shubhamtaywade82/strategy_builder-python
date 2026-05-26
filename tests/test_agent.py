import pytest
from unittest.mock import MagicMock
from strategy_builder.agent.desk_pipeline import DeskPipeline
from strategy_builder.domain import MarketState, Regime, VolatilityState, VolumeState, Bias
from datetime import datetime, timezone

@pytest.fixture
def mock_ollama_client():
    client = MagicMock()
    client.config = MagicMock()
    client.config.ollama_model = "test-model"
    client.config.ollama_temperature = 0.3
    client.config.ollama_num_ctx = 8192
    client.config.ollama_base_url = "http://localhost:11434"
    client.config.ollama_timeout = 60
    return client

@pytest.fixture
def sample_market_state():
    return MarketState(
        instrument="B-BTC_USDT",
        snapshot_at=datetime.now(timezone.utc),
        primary_timeframe="5m",
        regime=Regime.TREND_UP,
        session="london",
        volatility=VolatilityState.NORMAL,
        volume=VolumeState.AVERAGE,
        bias=Bias.LONG
    )

def test_desk_pipeline_run(mock_ollama_client, sample_market_state):
    # Mock LLM responses for each role
    mock_ollama_client.call_chat_api.side_effect = [
        # Observer response
        '{"narrative": "Bullish trend confirmed", "session_context": "London open", "key_levels": ["100"], "no_trade_context": []}',
        # PatternAnalyst response
        '{"accepted_patterns": [{"name": "pullback_continuation", "confidence": 0.9, "trigger": "EMA touch", "continuation": true, "invalidation": []}], "rejected_patterns": []}',
        # TradeDesigner response
        '{"candidates": [{"name": "Trend Pullback", "family": "mtf_pullback", "timeframes": ["5m"], "session": ["any"], "entry": {"conditions": ["generic_breakout"], "direction": "long"}, "exit": {"targets": [1.0, 2.0], "partial_exits": [0.5, 0.5]}, "risk": {"stop": "below_low", "position_sizing": "fixed_risk_percent", "max_risk_percent": 1.0}}]}',
        # Skeptic response
        '{"accepted": true, "concerns": ["Late entry"]}'
    ]

    pipeline = DeskPipeline(mock_ollama_client)
    features = {
        "primary_timeframe": "5m",
        "volatility": {"regime": "normal"},
        "structure": {"structure": "bullish"},
        "mtf_alignment": {"alignment": {"aligned_bullish": True}}
    }
    
    results = pipeline.run(instrument="B-BTC_USDT", features=features)
    
    assert len(results) == 1
    assert results[0]["name"] == "Trend Pullback"
    assert "skeptic_notes" in results[0]
