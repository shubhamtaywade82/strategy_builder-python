"""
1m Backtest Engine - Path-dependent, fee-adjusted.
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def backtest(ohlcv: pd.DataFrame, signals: list, up_pct: float, dn_pct: float,
             leverage: float = 10.0, fee_cost: float = 0.0009,
             max_horizon: int = 120) -> BacktestResult:
    if not signals or ohlcv.empty:
        return BacktestResult()

    o = ohlcv["open"].to_numpy(float)
    hi = ohlcv["high"].to_numpy(float)
    lo = ohlcv["low"].to_numpy(float)
    cl = ohlcv["close"].to_numpy(float)
    times = ohlcv["open_time"].to_numpy()
    n = len(ohlcv)

    trades = []

    for sig in signals:
        i = sig["bar_idx"]
        if i >= n - 1:
            continue
        side = sig["side"]
        entry = o[min(i + 1, n - 1)]

        # barriers at the literal target/stop distances (fee handled once, below)
        if side == 1:
            up_b, dn_b = entry * (1 + up_pct), entry * (1 - dn_pct)
        else:
            up_b, dn_b = entry * (1 - up_pct), entry * (1 + dn_pct)

        end = min(i + 1 + max_horizon, n)
        # pr is the RAW signed price move; the round-trip fee is subtracted ONCE
        # in net_pnl. Charging it here too would double-count it on losses/timeouts.
        label, bte, pr, exit_reason = 0, max_horizon, 0.0, "timeout"

        for j in range(i + 1, end):
            if side == 1:
                hit_dn, hit_up = lo[j] <= dn_b, hi[j] >= up_b
            else:
                hit_dn, hit_up = hi[j] >= dn_b, lo[j] <= up_b

            # conservative: if both barriers touch in the same bar, assume stop first
            if hit_dn:
                label, bte, pr, exit_reason = 0, j - (i + 1), -dn_pct, "stop"
                break
            if hit_up:
                label, bte, pr, exit_reason = 1, j - (i + 1), up_pct, "target"
                break
        else:
            pr = (cl[end - 1] - entry) / entry * side

        net_pnl = (pr - fee_cost) * leverage
        trades.append({
            "entry_idx": i, "entry_price": entry, "side": side,
            "exit_idx": i + 1 + bte, "exit_price": cl[min(i + 1 + bte, n - 1)],
            "exit_reason": exit_reason, "net_pnl": net_pnl,
            "margin_pnl": pr * leverage, "bars_held": bte,
            "label": label,
        })

    if not trades:
        return BacktestResult()

    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] <= 0]
    n_trades = len(trades)
    n_wins = len(wins)
    gross_profit = sum(t["net_pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["net_pnl"] for t in losses)) if losses else 0

    metrics = {
        "trade_count": n_trades,
        "win_count": n_wins,
        "loss_count": len(losses),
        "win_rate": n_wins / n_trades if n_trades else 0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "expectancy": sum(t["net_pnl"] for t in trades) / n_trades if n_trades else 0,
        "net_pnl": sum(t["net_pnl"] for t in trades),
        "avg_win": gross_profit / n_wins if n_wins else 0,
        "avg_loss": gross_loss / len(losses) if losses else 0,
        "target_hit_rate": sum(1 for t in trades if t["exit_reason"] == "target") / n_trades if n_trades else 0,
        "stop_hit_rate": sum(1 for t in trades if t["exit_reason"] == "stop") / n_trades if n_trades else 0,
        "avg_bars_held": np.mean([t["bars_held"] for t in trades]),
    }
    return BacktestResult(trades=trades, metrics=metrics)
