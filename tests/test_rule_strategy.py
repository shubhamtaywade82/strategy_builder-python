"""Unit tests for the rule-strategy backtest (no network)."""
import numpy as np
import pandas as pd
import pytest

from strategy_research._ui_engine import rule_strategy as rs


def _frame(n, tf_ms, start_ms=0, seed=0):
    rng = np.random.default_rng(seed)
    ot = pd.to_datetime(start_ms + np.arange(n) * tf_ms, unit="ms", utc=True)
    ct = pd.to_datetime(start_ms + (np.arange(n) + 1) * tf_ms - 1, unit="ms", utc=True)
    price = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high = price + np.abs(rng.normal(0, 0.3, n))
    low = price - np.abs(rng.normal(0, 0.3, n))
    return pd.DataFrame({
        "open_time": ot, "close_time": ct, "open": price, "high": high,
        "low": low, "close": price, "volume": np.abs(rng.normal(100, 30, n)),
    })


def _frames(n1m=4000):
    return {
        "1m": _frame(n1m, 60_000, seed=1),
        "15m": _frame(n1m // 15 + 5, 900_000, seed=2),
        "1h": _frame(n1m // 60 + 5, 3_600_000, seed=3),
        "4h": _frame(n1m // 240 + 60, 14_400_000, start_ms=-60 * 14_400_000, seed=4),
    }


def test_apply_cooldown_spacing():
    assert rs.apply_cooldown([1, 2, 3, 40, 41, 80], bars=30) == [1, 40, 80]
    assert rs.apply_cooldown([], bars=30) == []


def test_bootstrap_pvalue_bounds():
    # all-positive pnl -> mean almost never <= 0 -> tiny p
    assert rs.bootstrap_pvalue(np.full(100, 0.05)) < 0.05
    # all-negative -> p ~ 1
    assert rs.bootstrap_pvalue(np.full(100, -0.05)) > 0.95
    # too few -> 1.0
    assert rs.bootstrap_pvalue(np.array([0.1, 0.2])) == 1.0


def test_walk_forward_folds_stability():
    trades = [{"entry_idx": i, "net_pnl": 0.01} for i in range(0, 1000, 10)]
    wf = rs.walk_forward_folds(trades, 1000, n_folds=5)
    assert wf["stability"] == 1.0  # all folds positive
    assert len(wf["folds"]) == 5


def test_build_signal_mask_columns_and_leakage():
    mask = rs.build_signal_mask(_frames())
    for col in ("sig_long", "sig_short", "c4", "c5"):
        assert col in mask.columns
    # percentile thresholds need a full trailing window -> early bars cannot fire
    assert not mask["c4"].iloc[:50].any()
    assert mask["sig_long"].dtype == bool


def test_backtest_rule_structure():
    out = rs.backtest_rule(_frames(), {"up_pct": 0.01, "dn_pct": 0.005, "label": "2:1"},
                           cost=0.0004, leverage=10, horizon=120)
    assert set(("long", "short", "best", "has_edge", "conditions")).issubset(out)
    for side in ("long", "short"):
        for mode in ("momentum", "reversion"):
            cell = out[side][mode]
            assert cell["side"] == side and cell["mode"] == mode
            assert "is_robust" in cell and "robustness_score" in cell
    # random-walk synthetic data has no real edge -> nothing deployable
    assert out["has_edge"] is False


def test_backtest_rule_reversion_trades_opposite():
    out = rs.backtest_rule(_frames(), {"up_pct": 0.01, "dn_pct": 0.005, "label": "2:1"},
                           cost=0.0004, leverage=10, horizon=120)
    long_mom = out["long"]["momentum"]
    long_rev = out["long"]["reversion"]
    if long_mom.get("metrics") and long_rev.get("metrics"):
        assert long_mom["trade_direction"] == "long"
        assert long_rev["trade_direction"] == "short"
