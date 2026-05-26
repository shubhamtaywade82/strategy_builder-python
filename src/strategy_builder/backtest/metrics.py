import math
from typing import List, Dict, Any, Optional

class Metrics:
    @staticmethod
    def compute(trades: List[Any]) -> Dict[str, Any]:
        if not trades:
            return Metrics.empty_metrics()

        pnls = [t.pnl for t in trades]
        r_multiples = [t.pnl_r for t in trades if t.pnl_r is not None]
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]

        return {
            "trade_count": len(trades),
            "net_pnl": round(sum(pnls), 4),
            "gross_profit": round(sum(t.pnl for t in winners), 4),
            "gross_loss": round(sum(t.pnl for t in losers), 4),
            "win_rate": round(len(winners) / len(trades), 4),
            "loss_rate": round(len(losers) / len(trades), 4),
            "avg_win": round(sum(t.pnl for t in winners) / len(winners), 4) if winners else 0.0,
            "avg_loss": round(sum(t.pnl for t in losers) / len(losers), 4) if losers else 0.0,
            "profit_factor": Metrics.compute_profit_factor(winners, losers),
            "expectancy": Metrics.compute_expectancy(trades),
            "avg_r": round(sum(r_multiples) / len(r_multiples), 4) if r_multiples else 0.0,
            "max_r": max(r_multiples) if r_multiples else 0.0,
            "min_r": min(r_multiples) if r_multiples else 0.0,
            "max_drawdown": Metrics.compute_max_drawdown(pnls),
            "max_consecutive_losses": Metrics.max_consecutive(losers, trades),
            "max_consecutive_wins": Metrics.max_consecutive(winners, trades),
            "avg_hold_candles": round(sum(t.hold_candles for t in trades) / len(trades), 1),
            "total_fees": round(sum(t.fees for t in trades), 4),
            "sharpe_ratio": Metrics.compute_sharpe(pnls),
            "sortino_ratio": Metrics.compute_sortino(pnls),
            "calmar_ratio": Metrics.compute_calmar(pnls),
            "exit_reason_distribution": Metrics.exit_reason_dist(trades),
            "direction_distribution": Metrics.direction_dist(trades),
            "session_performance": {},
            "regime_performance": {},
            "instrument_performance": {}
        }

    @staticmethod
    def empty_metrics() -> Dict[str, Any]:
        return {
            "trade_count": 0, "net_pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
            "win_rate": 0.0, "loss_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "profit_factor": 0.0, "expectancy": 0.0, "avg_r": 0.0, "max_r": 0.0, "min_r": 0.0,
            "max_drawdown": 0.0, "max_consecutive_losses": 0, "max_consecutive_wins": 0,
            "avg_hold_candles": 0.0, "total_fees": 0.0, "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0, "calmar_ratio": 0.0,
            "exit_reason_distribution": {}, "direction_distribution": {},
            "session_performance": {}, "regime_performance": {}, "instrument_performance": {}
        }

    @staticmethod
    def compute_profit_factor(winners: List[Any], losers: List[Any]) -> float:
        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        if gross_loss == 0:
            return 0.0
        return round(gross_profit / gross_loss, 4)

    @staticmethod
    def compute_expectancy(trades: List[Any]) -> float:
        if not trades:
            return 0.0
        return round(sum(t.pnl for t in trades) / len(trades), 4)

    @staticmethod
    def compute_max_drawdown(pnls: List[float]) -> float:
        if not pnls:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        return round(max_dd, 4)

    @staticmethod
    def max_consecutive(subset: List[Any], all_trades: List[Any]) -> int:
        if not subset:
            return 0
        subset_ids = {t.position_id for t in subset}
        max_streak = 0
        current_streak = 0
        for t in all_trades:
            if t.position_id in subset_ids:
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
            else:
                current_streak = 0
        return max_streak

    @staticmethod
    def compute_sharpe(pnls: List[float], risk_free: float = 0.0) -> float:
        if len(pnls) < 2:
            return 0.0
        mean = sum(pnls) / len(pnls)
        variance = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
        stddev = math.sqrt(variance)
        if stddev == 0:
            return 0.0
        return round((mean - risk_free) / stddev, 4)

    @staticmethod
    def compute_sortino(pnls: List[float], risk_free: float = 0.0) -> float:
        if len(pnls) < 2:
            return 0.0
        mean = sum(pnls) / len(pnls)
        downside = [p for p in pnls if p < risk_free]
        if not downside:
            return 0.0
        downside_variance = sum((p - risk_free) ** 2 for p in downside) / len(downside)
        downside_dev = math.sqrt(downside_variance)
        if downside_dev == 0:
            return 0.0
        return round((mean - risk_free) / downside_dev, 4)

    @staticmethod
    def compute_calmar(pnls: List[float]) -> float:
        if not pnls:
            return 0.0
        total_return = sum(pnls)
        max_dd = Metrics.compute_max_drawdown(pnls)
        if max_dd == 0:
            return 0.0
        return round(total_return / max_dd, 4)

    @staticmethod
    def exit_reason_dist(trades: List[Any]) -> Dict[str, int]:
        dist = {}
        for t in trades:
            reason = str(t.exit_reason)
            dist[reason] = dist.get(reason, 0) + 1
        return dist

    @staticmethod
    def direction_dist(trades: List[Any]) -> Dict[str, int]:
        dist = {}
        for t in trades:
            direction = str(t.direction)
            dist[direction] = dist.get(direction, 0) + 1
        return dist
