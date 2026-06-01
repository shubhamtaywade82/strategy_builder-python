"""Deterministic unit tests for the shared live risk guard."""
from risk_guard import RiskGuard


def test_fixed_fractional_sizing():
    g = RiskGuard(account=10000.0, max_risk_pct=0.02, base_leverage=10.0)
    s = g.position_size(entry=150.0, stop=149.25)  # 0.5% stop
    assert s["stop_distance_pct"] == 0.005
    assert s["risk_amount"] == 200.0           # 2% of 10k
    assert s["notional"] == 40000.0            # 200 / 0.005
    assert s["margin_required"] == 4000.0      # notional / 10x
    assert s["leverage"] == 10.0


def test_leverage_steps_down_after_consecutive_losses():
    g = RiskGuard(base_leverage=10.0, reduced_leverage=5.0, step_down_after=3)
    assert g.current_leverage() == 10.0
    g.register_trade(-1.0); g.register_trade(-1.0)
    assert g.current_leverage() == 10.0
    g.register_trade(-1.0)                      # 3rd loss
    assert g.current_leverage() == 5.0
    g.register_trade(+5.0)                      # a win resets the streak
    assert g.consecutive_losses == 0
    assert g.current_leverage() == 10.0


def test_daily_loss_limit_halts():
    g = RiskGuard(account=100.0, daily_loss_limit_pct=0.05)
    ok, _ = g.can_trade()
    assert ok
    g.register_trade(-5.0)                      # -5% of 100
    ok, reason = g.can_trade()
    assert not ok and "daily loss" in reason


def test_max_consecutive_losses_halts():
    g = RiskGuard(account=10_000.0, max_consecutive_losses=5, daily_loss_limit_pct=1.0)
    for _ in range(5):
        g.register_trade(-1.0)
    ok, reason = g.can_trade()
    assert not ok and "consecutive" in reason


def test_cooldown_blocks_rapid_reentry():
    g = RiskGuard(cooldown_bars=30)
    g.mark_entry(100)
    ok, reason = g.can_trade(bar_idx=120)
    assert not ok and "cooldown" in reason
    ok, _ = g.can_trade(bar_idx=131)
    assert ok


def test_quarter_kelly_caps_risk():
    # Strong edge -> quarter-Kelly is large, so the 2% cap binds.
    g = RiskGuard(account=100.0, max_risk_pct=0.02, win_rate=0.6, payoff_ratio=2.0)
    assert g.position_size(100.0, 99.5)["risk_pct"] == 0.02
    # Weak edge -> quarter-Kelly below 2%, so it binds instead.
    # w=0.52, r=1.0 -> Kelly=0.04, quarter-Kelly=0.01 < 0.02.
    g2 = RiskGuard(account=100.0, max_risk_pct=0.02, win_rate=0.52, payoff_ratio=1.0)
    assert g2.position_size(100.0, 99.5)["risk_pct"] < 0.02
