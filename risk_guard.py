"""
Risk Guard — shared live-trading guardrails
===========================================
Stateful position sizing + circuit breakers the report (§6.3) prescribes, kept
separate from any execution so it can be unit-tested deterministically and reused
by trading_bot.py.

It does NOT place orders. It answers two questions:
  - ``can_trade()``  — are we allowed to open a new position right now?
  - ``position_size(...)`` — how big should it be?
and tracks streaks/daily-PnL via ``register_trade(pnl_pct)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Tuple


@dataclass
class RiskGuard:
    account: float = 100.0
    max_risk_pct: float = 0.02          # fixed-fractional risk per trade (2%)
    daily_loss_limit_pct: float = 0.05  # halt for the day after -5%
    max_consecutive_losses: int = 5     # halt after N losers in a row
    base_leverage: float = 10.0
    reduced_leverage: float = 5.0
    step_down_after: int = 3            # drop leverage after N consecutive losses
    cooldown_bars: int = 30            # min bars between entries
    # kelly clamp inputs (from the validated rule); quarter-Kelly cap
    win_rate: float = 0.5
    payoff_ratio: float = 2.0          # avg_win / avg_loss

    # --- mutable session state ---
    consecutive_losses: int = field(default=0, init=False)
    daily_pnl: float = field(default=0.0, init=False)
    _day: Optional[date] = field(default=None, init=False)
    _last_entry_bar: int = field(default=-10**9, init=False)

    # ---------------------------------------------------------------- sizing
    def current_leverage(self) -> float:
        return self.reduced_leverage if self.consecutive_losses >= self.step_down_after else self.base_leverage

    def _quarter_kelly_cap(self) -> float:
        """Quarter-Kelly fraction (>=0), used only as an UPPER bound on risk."""
        w, r = self.win_rate, max(self.payoff_ratio, 1e-9)
        kelly = w - (1 - w) / r
        return max(kelly, 0.0) * 0.25

    def position_size(self, entry: float, stop: float, bar_idx: Optional[int] = None) -> dict:
        """Fixed-fractional risk sizing, clamped to quarter-Kelly. No order placed."""
        self._roll_day()
        stop_dist = abs(entry - stop) / entry if entry > 0 else 0.0
        # cap risk fraction at min(configured, quarter-Kelly)
        risk_frac = min(self.max_risk_pct, self._quarter_kelly_cap()) if self._quarter_kelly_cap() > 0 else self.max_risk_pct
        risk_amount = self.account * risk_frac
        notional = (risk_amount / stop_dist) if stop_dist > 0 else 0.0
        lev = self.current_leverage()
        margin = notional / lev if lev > 0 else 0.0
        qty = notional / entry if entry > 0 else 0.0
        return {
            "risk_pct": risk_frac,
            "risk_amount": round(risk_amount, 4),
            "notional": round(notional, 4),
            "qty": round(qty, 6),
            "leverage": lev,
            "margin_required": round(margin, 4),
            "stop_distance_pct": round(stop_dist, 5),
        }

    # ----------------------------------------------------------- can we trade
    def can_trade(self, bar_idx: Optional[int] = None) -> Tuple[bool, str]:
        self._roll_day()
        if self.daily_pnl <= -abs(self.daily_loss_limit_pct) * self.account:
            return False, f"daily loss limit hit ({self.daily_pnl:.2f})"
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"max consecutive losses ({self.consecutive_losses})"
        if bar_idx is not None and (bar_idx - self._last_entry_bar) < self.cooldown_bars:
            return False, "cooldown active"
        return True, "ok"

    def mark_entry(self, bar_idx: int) -> None:
        self._last_entry_bar = bar_idx

    # ------------------------------------------------------- trade accounting
    def register_trade(self, pnl_quote: float) -> None:
        """Record a closed trade's PnL (account currency) and update streak/daily."""
        self._roll_day()
        self.daily_pnl += pnl_quote
        self.account += pnl_quote
        if pnl_quote < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def _roll_day(self) -> None:
        today = date.today()
        if self._day != today:
            self._day = today
            self.daily_pnl = 0.0
