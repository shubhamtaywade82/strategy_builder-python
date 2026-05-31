"""
Backtest Engine
===============
Fee-and-slippage-adjusted backtest for 1m strategies.

Key features:
- Entry at next bar's open (realistic fill)
- Path-dependent exit checking on 1m highs/lows
- Fee-adjusted PnL
- Margin PnL for leverage

Target: +1% price move = +10% on 10x margin
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("backtest")


@dataclass
class Trade:
    """A single trade record."""
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    side: int  # +1 long, -1 short
    exit_idx: int
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str  # 'target', 'stop', 'timeout'
    price_return: float
    gross_pnl: float
    fees: float
    net_pnl: float
    margin_pnl: float
    bars_held: int


@dataclass
class BacktestResult:
    """Complete backtest results."""
    trades: List[Trade]
    equity_curve: pd.Series
    metrics: Dict[str, float]


def backtest_1m_strategy(
    ohlcv: pd.DataFrame,
    signals: List,
    up_pct: float = 0.01,
    dn_pct: float = 0.005,
    leverage: float = 10.0,
    fee_per_side: float = 0.0005,
    slippage: float = 0.0002,
    max_horizon: int = 120,
    initial_capital: float = 1.0,
) -> BacktestResult:
    """Backtest a strategy on 1m data.

    Args:
        ohlcv: 1m OHLCV DataFrame with open_time, open, high, low, close
        signals: List of entry signals (with bar_idx, side, confidence)
        up_pct: Target move (price space)
        dn_pct: Stop loss (price space)
        leverage: Leverage multiplier
        fee_per_side: Fee per trade side (0.0005 = 0.05%)
        slippage: Slippage per entry/exit (0.0002 = 0.02%)
        max_horizon: Max bars to hold
        initial_capital: Starting capital (normalized to 1.0)

    Returns:
        BacktestResult with trades and metrics
    """
    if ohlcv.empty or not signals:
        return BacktestResult(trades=[], equity_curve=pd.Series(), metrics={})

    o = ohlcv["open"].to_numpy(float)
    hi = ohlcv["high"].to_numpy(float)
    lo = ohlcv["low"].to_numpy(float)
    cl = ohlcv["close"].to_numpy(float)
    times = ohlcv["open_time"].to_numpy()
    n = len(ohlcv)

    gross_up = up_pct + 2 * fee_per_side + slippage
    total_cost = 2 * fee_per_side + slippage

    trades = []
    equity = [initial_capital]

    for sig in signals:
        i = sig.bar_idx
        if i >= n - 1:
            continue

        side = sig.side
        entry_price = o[min(i + 1, n - 1)]

        if side == 1:
            up_b = entry_price * (1 + gross_up)
            dn_b = entry_price * (1 - dn_pct)
        else:
            up_b = entry_price * (1 - gross_up)
            dn_b = entry_price * (1 + dn_pct)

        end = min(i + 1 + max_horizon, n)
        label, bte, pret = 0, max_horizon, 0.0
        exit_price = cl[end - 1]
        exit_reason = "timeout"

        for j in range(i + 1, end):
            bh, bl = hi[j], lo[j]

            if side == 1:
                hit_dn = bl <= dn_b
                hit_up = bh >= up_b
            else:
                hit_dn = bh >= dn_b
                hit_up = bl <= up_b

            if hit_dn and hit_up:
                label, bte = 0, j - (i + 1)
                pret = -dn_pct
                exit_price = dn_b
                exit_reason = "stop"
                break

            if hit_dn:
                label, bte = 0, j - (i + 1)
                pret = -dn_pct
                exit_price = dn_b
                exit_reason = "stop"
                break

            if hit_up:
                label, bte = 1, j - (i + 1)
                pret = up_pct
                exit_price = up_b
                exit_reason = "target"
                break
        else:
            raw = (cl[end - 1] - entry_price) / entry_price * side
            pret = raw
            exit_price = cl[end - 1]
            exit_reason = "timeout"

        # Fee-adjusted
        fees = total_cost
        net_pret = pret - fees
        margin_pnl = net_pret * leverage

        trade = Trade(
            entry_idx=i,
            entry_time=pd.Timestamp(times[i]),
            entry_price=entry_price,
            side=side,
            exit_idx=i + 1 + bte,
            exit_time=pd.Timestamp(times[min(i + 1 + bte, n - 1)]),
            exit_price=exit_price,
            exit_reason=exit_reason,
            price_return=pret,
            gross_pnl=pret * leverage,
            fees=fees * leverage,
            net_pnl=net_pret * leverage,
            margin_pnl=margin_pnl,
            bars_held=bte,
        )
        trades.append(trade)
        equity.append(equity[-1] + margin_pnl * initial_capital)

    if not trades:
        return BacktestResult(trades=[], equity_curve=pd.Series([initial_capital]), metrics={})

    equity_curve = pd.Series(equity)
    metrics = compute_metrics(trades, equity_curve)

    return BacktestResult(trades=trades, equity_curve=equity_curve, metrics=metrics)


def compute_metrics(trades: List[Trade], equity: pd.Series) -> Dict[str, float]:
    """Compute performance metrics from trades."""
    if not trades:
        return {}

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]

    n = len(trades)
    n_wins = len(wins)
    n_losses = len(losses)

    win_rate = n_wins / n if n > 0 else 0

    gross_profit = sum(t.net_pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.net_pnl for t in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    net_pnl = sum(t.net_pnl for t in trades)
    expectancy = net_pnl / n if n > 0 else 0

    margin_pnls = [t.margin_pnl for t in trades]

    # Drawdown
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()

    # Sharpe (simplified)
    returns = pd.Series(margin_pnls)
    sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24 * 60) if returns.std() > 0 else 0

    # By exit reason
    target_hits = sum(1 for t in trades if t.exit_reason == "target")
    stop_hits = sum(1 for t in trades if t.exit_reason == "stop")

    return {
        "trade_count": n,
        "win_count": n_wins,
        "loss_count": n_losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "net_pnl": net_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "avg_win": gross_profit / n_wins if n_wins > 0 else 0,
        "avg_loss": gross_loss / n_losses if n_losses > 0 else 0,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "avg_bars_held": np.mean([t.bars_held for t in trades]),
        "target_hit_rate": target_hits / n if n > 0 else 0,
        "stop_hit_rate": stop_hits / n if n > 0 else 0,
        "margin_pnl_mean": np.mean(margin_pnls),
        "margin_pnl_std": np.std(margin_pnls),
    }


def evaluate_expectancy(
    result: BacktestResult,
    min_trades: int = 30,
    min_pf: float = 1.1,
    min_wr: float = 0.30,
) -> Tuple[bool, str]:
    """Evaluate if a strategy meets minimum viability criteria.

    Returns:
        (is_viable, reason)
    """
    m = result.metrics
    if m.get("trade_count", 0) < min_trades:
        return False, f"trades={m.get('trade_count', 0)} < {min_trades}"

    if m.get("profit_factor", 0) < min_pf:
        return False, f"PF={m.get('profit_factor', 0):.2f} < {min_pf}"

    if m.get("win_rate", 0) < min_wr:
        return False, f"WR={m.get('win_rate', 0):.1%} < {min_wr:.0%}"

    if m.get("expectancy", 0) <= 0:
        return False, f"expectancy={m.get('expectancy', 0):.4f} <= 0"

    return True, "PASS"
