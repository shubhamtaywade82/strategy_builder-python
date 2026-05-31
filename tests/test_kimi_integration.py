"""
Integration tests for the Kimi Agent v2 modules.

Verifies SMC detector, triple-barrier labeler, position sizer,
and walk-forward validator work correctly using the real installed package.
"""
import random
import pytest
import pathlib
import sys
import types

# Path to integration modules
_ROOT = pathlib.Path(__file__).parent.parent / "src" / "strategy_builder"


def _make_candles(n: int = 500):
    """Generate synthetic Candle objects using the real domain class."""
    from strategy_builder.domain import Candle
    price = 150.0
    candles = []
    for i in range(n):
        change = random.gauss(0, 0.5)
        o = price
        c = price + change
        h = max(o, c) + abs(random.gauss(0, 0.3))
        l = min(o, c) - abs(random.gauss(0, 0.3))
        candles.append(Candle(timestamp=i, open=o, high=h, low=l, close=c, volume=1000.0))
        price = c
    return candles


# ── SMC Detector ─────────────────────────────────────────────────────────────

def test_smc_detector_profile():
    from strategy_builder.features.smc_detector import SmcDetector
    candles = _make_candles(200)
    profile = SmcDetector.profile(candles)

    assert "fair_value_gaps" in profile
    assert "order_blocks" in profile
    assert "liquidity_sweeps" in profile
    assert "premium_discount" in profile
    s = profile["summary"]
    assert "bullish_fvg_count" in s
    assert "bearish_fvg_count" in s
    assert "current_zone" in s
    assert profile["premium_discount"]["zone"] in ("premium", "discount", "equilibrium")


def test_smc_detector_insufficient_data():
    from strategy_builder.features.smc_detector import SmcDetector
    from strategy_builder.domain import Candle
    tiny = [Candle(timestamp=i, open=100, high=101, low=99, close=100, volume=100) for i in range(5)]
    result = SmcDetector.profile(tiny)
    assert "error" in result


# ── Triple-Barrier Labeler ────────────────────────────────────────────────────

def test_triple_barrier_labels():
    from strategy_builder.backtest.triple_barrier import TripleBarrierLabeler, BarrierConfig
    cfg = BarrierConfig(up_pct=0.01, dn_pct=0.005, max_horizon=60)
    labeler = TripleBarrierLabeler(cfg)
    candles = _make_candles(300)
    labels = labeler.label_both_sides(candles)

    assert len(labels) > 0
    assert all("long_label" in r for r in labels)
    assert all("short_label" in r for r in labels)
    assert all("best_side" in r for r in labels)
    assert all(r["long_label"] in (0, 1) for r in labels)
    assert all(r["best_side"] in (-1, 0, 1) for r in labels)


def test_triple_barrier_stats():
    from strategy_builder.backtest.triple_barrier import TripleBarrierLabeler, BarrierConfig
    cfg = BarrierConfig(up_pct=0.01, dn_pct=0.005, max_horizon=60)
    labeler = TripleBarrierLabeler(cfg)
    labels = labeler.label_both_sides(_make_candles(300))
    stats = TripleBarrierLabeler.label_stats(labels)

    assert stats["total"] > 0
    assert 0 <= stats["long_win_pct"] <= 100
    assert 0 <= stats["short_win_pct"] <= 100


# ── Position Sizer ────────────────────────────────────────────────────────────

def test_kelly_criterion():
    from strategy_builder.analytics.position_sizing import PositionSizer
    sizer = PositionSizer(win_rate=0.576, avg_win=0.0095, avg_loss=0.0059)
    kelly = sizer.kelly()

    assert kelly.full_kelly >= 0
    assert kelly.half_kelly == kelly.full_kelly * 0.5
    assert kelly.quarter_kelly == kelly.full_kelly * 0.25
    assert kelly.edge > 0  # positive edge given our win_rate/avg_win


def test_fixed_fractional():
    from strategy_builder.analytics.position_sizing import PositionSizer
    result = PositionSizer.fixed_fractional(
        account_size=10_000, risk_pct=2.0,
        entry_price=150.0, stop_price=149.25, leverage=10.0
    )
    assert result["notional"] > 0
    assert result["margin_required"] > 0
    assert result["contracts"] > 0


def test_regime_adjusted():
    from strategy_builder.analytics.position_sizing import PositionSizer
    for regime, expected_multiplier in [("trending", 1.0), ("ranging", 0.6), ("volatile", 0.4), ("quiet", 1.3)]:
        r = PositionSizer.regime_adjusted(10_000, 0.02, regime)
        assert abs(r["multiplier"] - expected_multiplier) < 0.001
        assert r["adjusted_risk_pct"] == pytest.approx(0.02 * expected_multiplier * 100, rel=1e-4)


# ── Walk-Forward Validator (ML) ───────────────────────────────────────────────

def test_walk_forward_validator():
    import pandas as pd
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from strategy_builder.backtest.walk_forward import WalkForwardValidator

    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 5))
    y = pd.Series(np.random.randint(0, 2, 1000))

    validator = WalkForwardValidator(n_folds=3, embargo=50)
    result = validator.validate(X, y, lambda: DummyClassifier(strategy="stratified"))

    assert "folds" in result
    assert "is_valid" in result
    assert "avg_test_auc" in result
    assert result["folds"] > 0


# ── Feature Builder integration (SMC in build output) ────────────────────────

def test_feature_builder_includes_smc():
    from strategy_builder.features.feature_builder import FeatureBuilder
    from strategy_builder.domain import Candle
    import time

    now_s = int(time.time())
    candles = []
    price = 100.0
    for i in range(200):
        change = random.gauss(0, 0.3)
        o = price
        c = price + change
        candles.append(Candle(
            timestamp=now_s + i * 60,
            open=o, high=max(o, c) + 0.1,
            low=min(o, c) - 0.1, close=c, volume=500.0
        ))
        price = c

    features = FeatureBuilder.build("TEST", {"5m": candles})
    assert "smc" in features
    assert "fair_value_gaps" in features["smc"]
    assert "summary" in features["smc"]
