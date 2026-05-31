"""
Position Sizing & Risk Management
==================================
Kelly Criterion, Fixed Fractional, and Volatility-adjusted sizing.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class KellyResult:
    kelly_fraction: float      # Full Kelly fraction
    half_kelly: float          # Conservative half-Kelly
    quarter_kelly: float       # Very conservative
    win_rate: float
    avg_win: float
    avg_loss: float
    edge: float
    optimal_f: float            # Optimal fixed fraction


def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> KellyResult:
    """Calculate Kelly Criterion for position sizing.

    Kelly% = W - ((1-W) / (AvgWin/AvgLoss))
    where W = win rate, R = win/loss ratio
    """
    if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
        return KellyResult(0, 0, 0, win_rate, avg_win, avg_loss, 0, 0)

    win_loss_ratio = abs(avg_win / avg_loss)
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
    edge = win_rate * avg_win - (1 - win_rate) * abs(avg_loss)

    # Clamp
    kelly = max(-1, min(1, kelly))

    # Optimal fixed fraction (simplified)
    optimal_f = kelly / (abs(avg_win) + abs(avg_loss)) if (avg_win + avg_loss) != 0 else 0
    optimal_f = max(0, min(0.5, optimal_f))

    return KellyResult(
        kelly_fraction=kelly,
        half_kelly=kelly * 0.5,
        quarter_kelly=kelly * 0.25,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        edge=edge,
        optimal_f=optimal_f,
    )


def fixed_fractional_size(
    account_size: float,
    risk_per_trade: float,      # % of account to risk (e.g. 0.02 = 2%)
    entry_price: float,
    stop_price: float,
    leverage: float = 10.0,
) -> dict:
    """Calculate position size using fixed fractional method.

    Position = (Account * Risk%) / (|Entry - Stop| * Leverage)
    """
    price_risk = abs(entry_price - stop_price) / entry_price
    if price_risk == 0:
        return {"position_size": 0, "margin_required": 0, "notional": 0}

    risk_amount = account_size * risk_per_trade
    position_notional = risk_amount / price_risk
    margin_required = position_notional / leverage

    return {
        "position_size": position_notional,
        "margin_required": margin_required,
        "notional": position_notional,
        "risk_amount": risk_amount,
        "risk_pct": risk_per_trade * 100,
        "leverage": leverage,
    }


def volatility_adjusted_size(
    account_size: float,
    base_risk: float,
    current_atr: float,
    median_atr: float,
    entry_price: float,
    leverage: float = 10.0,
) -> dict:
    """Adjust position size based on current vs median volatility.

    When volatility is high, reduce size. When low, increase.
    """
    vol_ratio = current_atr / median_atr if median_atr > 0 else 1.0
    adjusted_risk = base_risk / max(vol_ratio, 0.5)  # Floor at 0.5x
    adjusted_risk = min(adjusted_risk, base_risk * 2)  # Cap at 2x

    stop_distance = current_atr * 1.5  # 1.5x ATR stop
    stop_pct = stop_distance / entry_price

    risk_amount = account_size * adjusted_risk
    position_notional = risk_amount / stop_pct
    margin_required = position_notional / leverage

    return {
        "position_size": position_notional,
        "margin_required": margin_required,
        "notional": position_notional,
        "risk_amount": risk_amount,
        "risk_pct": adjusted_risk * 100,
        "vol_adjustment": 1 / vol_ratio,
        "current_atr": current_atr,
        "median_atr": median_atr,
        "leverage": leverage,
    }


def monte_carlo_sizing(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    n_trades: int = 1000,
    n_sims: int = 1000,
    max_ruin: float = 0.5,  # Max acceptable ruin probability
) -> dict:
    """Run Monte Carlo to find optimal risk fraction.

    Tests multiple risk fractions, returns the one with best
    growth rate that keeps ruin probability below threshold.
    """
    risk_fractions = np.arange(0.01, 0.51, 0.01)
    results = []
    # seeded so the Monte Carlo sizing recommendation is reproducible run-to-run
    rng = np.random.default_rng(2024)

    for risk_frac in risk_fractions:
        final_equities = []
        max_drawdowns = []

        for _ in range(n_sims):
            equity = 1.0
            peak = equity
            max_dd = 0

            for _ in range(n_trades):
                if rng.random() < win_rate:
                    equity *= (1 + risk_frac * avg_win)
                else:
                    equity *= (1 - risk_frac * abs(avg_loss))

                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd

            final_equities.append(equity)
            max_drawdowns.append(max_dd)

        mean_final = np.mean(final_equities)
        ruin_prob = np.mean(np.array(final_equities) < max_ruin)
        mean_dd = np.mean(max_drawdowns)

        results.append({
            "risk_fraction": risk_frac,
            "mean_final": mean_final,
            "ruin_prob": ruin_prob,
            "mean_dd": mean_dd,
            "score": mean_final * (1 - ruin_prob),  # Growth * survival
        })

    # Best by score with ruin prob < 10%
    viable = [r for r in results if r["ruin_prob"] < 0.1]
    if not viable:
        viable = results

    best = max(viable, key=lambda x: x["score"])

    return {
        "optimal_risk_fraction": best["risk_fraction"],
        "optimal_risk_pct": best["risk_fraction"] * 100,
        "expected_growth": best["mean_final"],
        "ruin_probability": best["ruin_prob"] * 100,
        "expected_max_dd": best["mean_dd"] * 100,
        "all_results": results,
    }


def regime_adjusted_size(
    account_size: float,
    base_risk: float,
    regime: str,  # 'trending', 'ranging', 'volatile', 'quiet'
    entry_price: float,
    stop_price: float,
    leverage: float = 10.0,
) -> dict:
    """Adjust position size based on market regime.

    Trending: Normal size
    Ranging: Reduce (false signals)
    Volatile: Reduce size, widen stop
    Quiet: Increase size (clean moves)
    """
    regime_multipliers = {
        "trending": 1.0,
        "ranging": 0.6,
        "volatile": 0.4,
        "quiet": 1.3,
    }

    multiplier = regime_multipliers.get(regime, 0.5)
    adjusted_risk = base_risk * multiplier

    price_risk = abs(entry_price - stop_price) / entry_price
    risk_amount = account_size * adjusted_risk
    position_notional = risk_amount / price_risk if price_risk > 0 else 0
    margin_required = position_notional / leverage

    return {
        "position_size": position_notional,
        "margin_required": margin_required,
        "risk_pct": adjusted_risk * 100,
        "regime": regime,
        "regime_multiplier": multiplier,
        "leverage": leverage,
    }
