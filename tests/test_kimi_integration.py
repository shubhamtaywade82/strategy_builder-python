"""
Integration Test — Verifies all modules work together.
Run: python test_integration.py
"""
import sys
import random
import types

# Mock the strategy_builder package hierarchy
sb = types.ModuleType("strategy_builder")
sb_domain = types.ModuleType("strategy_builder.domain")
sb_exceptions = types.ModuleType("strategy_builder.exceptions")

# Mock domain classes
class Timeframe:
    M1 = "1m"; M5 = "5m"; M15 = "15m"; M30 = "30m"
    H1 = "1h"; H4 = "4h"; D1 = "1d"

class Candle:
    __slots__ = ["timestamp", "open", "high", "low", "close", "volume"]
    def __init__(self, ts, o, h, l, c, v=1000):
        self.timestamp = ts
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v

class DataError(Exception):
    pass

sb_domain.Candle = Candle
sb_domain.Timeframe = Timeframe
sb_exceptions.DataError = DataError

sys.modules["strategy_builder"] = sb
sys.modules["strategy_builder.domain"] = sb_domain
sys.modules["strategy_builder.exceptions"] = sb_exceptions

# Now we can import the modules
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

smc_mod = load_module("strategy_builder.features.smc_detector", "/mnt/agents/output/integration/smc_detector.py")
tb_mod = load_module("strategy_builder.backtest.triple_barrier", "/mnt/agents/output/integration/triple_barrier.py")
ps_mod = load_module("strategy_builder.analytics.position_sizing", "/mnt/agents/output/integration/position_sizing.py")
wf_mod = load_module("strategy_builder.backtest.walk_forward", "/mnt/agents/output/integration/walk_forward.py")


def generate_mock_candles(n=500):
    price = 150.0
    candles = []
    for i in range(n):
        change = random.gauss(0, 0.5)
        o = price
        c = price + change
        h = max(o, c) + abs(random.gauss(0, 0.3))
        l = min(o, c) - abs(random.gauss(0, 0.3))
        v = random.gauss(10000, 3000)
        candles.append(Candle(i, o, h, l, c, v))
        price = c
    return candles


def test_smc_detector():
    profile = smc_mod.SmcDetector.profile(generate_mock_candles(200))
    assert "fair_value_gaps" in profile
    assert "order_blocks" in profile
    s = profile["summary"]
    print(f"  SMC: {s['bullish_fvg_count']} bull FVGs, "
          f"{s['active_bull_ob']} active OBs, zone={profile['premium_discount']['zone']}")
    return True


def test_triple_barrier():
    cfg = tb_mod.BarrierConfig(up_pct=0.01, dn_pct=0.005, max_horizon=60)
    labeler = tb_mod.TripleBarrierLabeler(cfg)
    labels = labeler.label_both_sides(generate_mock_candles(300))
    stats = tb_mod.TripleBarrierLabeler.label_stats(labels)
    assert stats["total"] > 0
    print(f"  Barrier: n={stats['total']}, long_wr={stats['long_win_pct']}%, "
          f"best_side={stats['best_side_pct']}%")
    return True


def test_position_sizing():
    sizer = ps_mod.PositionSizer(win_rate=0.576, avg_win=0.0095, avg_loss=0.0059)
    kelly = sizer.kelly()
    size = sizer.fixed_fractional(10000, 2.0, 150, 149.25, 10)
    regime = sizer.regime_adjusted(10000, 0.02, "trending")
    assert size["notional"] > 0
    print(f"  Sizing: Kelly={kelly.half_kelly:.3f} (half), "
          f"position=${size['notional']:.0f}")
    return True


def test_walk_forward():
    import pandas as pd
    import numpy as np
    from sklearn.dummy import DummyClassifier
    
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 5))
    y = pd.Series(np.random.randint(0, 2, 1000))
    
    validator = wf_mod.WalkForwardValidator(n_folds=3, embargo=50)
    result = validator.validate(X, y, lambda: DummyClassifier(strategy="stratified"))
    assert "folds" in result
    print(f"  WalkForward: {result['folds']} folds, valid={result['is_valid']}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("INTEGRATION TEST SUITE")
    print("=" * 60)

    tests = [
        ("SMC Detector", test_smc_detector),
        ("Triple Barrier", test_triple_barrier),
        ("Position Sizing", test_position_sizing),
        ("Walk-Forward CV", test_walk_forward),
    ]

    passed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            if fn():
                print(f"  PASS")
                passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print(f"{'=' * 60}")
    sys.exit(0 if passed == len(tests) else 1)
