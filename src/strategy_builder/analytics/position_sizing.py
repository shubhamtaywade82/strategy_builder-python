"""
Position Sizing for strategy_builder-python
============================================
Adds to: src/strategy_builder/analytics/position_sizing.py

Usage:
    from strategy_builder.analytics.position_sizing import PositionSizer, KellyResult
    
    # From your backtest results
    sizer = PositionSizer(win_rate=0.576, avg_win=0.0095, avg_loss=0.0059)
    
    # Kelly
    kelly = sizer.kelly()  # -> KellyResult with full/half/quarter kelly
    
    # Fixed fractional for a specific trade
    size = sizer.fixed_fractional(
        account_size=10000,
        risk_pct=2.0,  # 2% of account
        entry_price=150,
        stop_price=149.25,
        leverage=10,
    )
    
    # Regime-adjusted
    size = sizer.regime_adjusted(
        account_size=10000,
        base_risk=0.02,
        regime="trending",  # or "ranging", "volatile", "quiet"
    )
"""
from dataclasses import dataclass
from typing import Dict, Any, Literal
import numpy as np


@dataclass
class KellyResult:
    full_kelly: float
    half_kelly: float
    quarter_kelly: float
    win_rate: float
    avg_win: float
    avg_loss: float
    edge: float


class PositionSizer:
    """Position sizing using Kelly Criterion and fixed fractional methods."""

    def __init__(self, win_rate: float, avg_win: float, avg_loss: float):
        """
        Args:
            win_rate: 0-1 probability of win
            avg_win: average win as decimal (e.g. 0.01 = 1%)
            avg_loss: average loss as decimal (e.g. 0.005 = 0.5%)
        """
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = abs(avg_loss)

    def kelly(self) -> KellyResult:
        """Calculate Kelly Criterion fractions."""
        if self.avg_loss == 0:
            return KellyResult(0, 0, 0, self.win_rate, self.avg_win, self.avg_loss, 0)

        r = self.avg_win / self.avg_loss
        kelly = self.win_rate - ((1 - self.win_rate) / r)
        edge = self.win_rate * self.avg_win - (1 - self.win_rate) * self.avg_loss

        return KellyResult(
            full_kelly=max(0, kelly),
            half_kelly=max(0, kelly * 0.5),
            quarter_kelly=max(0, kelly * 0.25),
            win_rate=self.win_rate,
            avg_win=self.avg_win,
            avg_loss=self.avg_loss,
            edge=edge,
        )

    @staticmethod
    def fixed_fractional(
        account_size: float,
        risk_pct: float,
        entry_price: float,
        stop_price: float,
        leverage: float = 10.0,
    ) -> Dict[str, Any]:
        """Fixed fractional position sizing."""
        price_risk = abs(entry_price - stop_price) / entry_price
        if price_risk == 0:
            return {"error": "zero_price_risk"}

        risk_amount = account_size * (risk_pct / 100)
        notional = risk_amount / price_risk
        margin = notional / leverage

        return {
            "risk_amount": risk_amount,
            "notional": notional,
            "margin_required": margin,
            "margin_pct_of_account": 100 * margin / account_size,
            "contracts": notional / entry_price,
        }

    @staticmethod
    def regime_adjusted(
        account_size: float,
        base_risk: float,
        regime: Literal["trending", "ranging", "volatile", "quiet"],
    ) -> Dict[str, Any]:
        """Adjust risk % based on market regime."""
        multipliers = {
            "trending": 1.0,
            "ranging": 0.6,
            "volatile": 0.4,
            "quiet": 1.3,
        }
        m = multipliers.get(regime, 0.5)
        adjusted = base_risk * m

        return {
            "regime": regime,
            "base_risk_pct": base_risk * 100,
            "adjusted_risk_pct": adjusted * 100,
            "multiplier": m,
            "risk_per_10k": adjusted * 10000,
        }

    def monte_carlo(
        self,
        account_size: float = 10000,
        n_trades: int = 1000,
        n_sims: int = 1000,
    ) -> Dict[str, Any]:
        """Run Monte Carlo to find optimal risk fraction."""
        risk_fracs = np.arange(0.01, 0.51, 0.01)
        results = []

        for rf in risk_fracs:
            finals = []
            for _ in range(n_sims):
                eq = 1.0
                for _ in range(n_trades):
                    if np.random.random() < self.win_rate:
                        eq *= (1 + rf * self.avg_win)
                    else:
                        eq *= (1 - rf * self.avg_loss)
                finals.append(eq)

            finals = np.array(finals)
            results.append({
                "risk": rf,
                "mean_final": np.mean(finals),
                "ruin_prob": np.mean(finals < 0.5),
                "score": np.mean(finals) * (1 - np.mean(finals < 0.5)),
            })

        viable = [r for r in results if r["ruin_prob"] < 0.1]
        if not viable:
            viable = results
        best = max(viable, key=lambda x: x["score"])

        return {
            "optimal_risk_pct": round(best["risk"] * 100, 2),
            "expected_return": round(best["mean_final"], 2),
            "ruin_probability": round(best["ruin_prob"] * 100, 1),
        }
