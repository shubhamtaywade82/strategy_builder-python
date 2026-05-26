from typing import Dict, Any, List

class Gatekeeper:
    GATES = {
        "min_trades": 20,
        "max_drawdown_multiple": 5.0,
        "min_oos_positive_folds": 0.6,
        "min_profit_factor": 1.1,
        "max_avg_degradation": 0.7,
        "min_win_rate": 0.25
    }

    @staticmethod
    def evaluate(walk_forward_result: Dict[str, Any], metrics: Any = None) -> Dict[str, Any]:
        agg = walk_forward_result.get("aggregate", {})
        failures = []

        if agg.get("oos_trade_count", 0) < Gatekeeper.GATES["min_trades"]:
            failures.append(f"Insufficient trades: {agg.get('oos_trade_count')} < {Gatekeeper.GATES['min_trades']}")

        if agg.get("oos_expectancy", 0) <= 0:
            failures.append(f"Negative OOS expectancy: {agg.get('oos_expectancy')}")

        if agg.get("oos_profit_factor", 0) < Gatekeeper.GATES["min_profit_factor"]:
            failures.append(f"Low profit factor: {agg.get('oos_profit_factor')} < {Gatekeeper.GATES['min_profit_factor']}")

        if agg.get("avg_degradation", 0) > Gatekeeper.GATES["max_avg_degradation"]:
            failures.append(f"High IS-to-OOS degradation: {agg.get('avg_degradation')} > {Gatekeeper.GATES['max_avg_degradation']}")

        if agg.get("oos_win_rate", 0) < Gatekeeper.GATES["min_win_rate"]:
            failures.append(f"Low win rate: {agg.get('oos_win_rate')} < {Gatekeeper.GATES['min_win_rate']}")

        stability = walk_forward_result.get("stability_score", 0.0)
        if stability < Gatekeeper.GATES["min_oos_positive_folds"]:
            failures.append(f"Unstable across folds: stability {stability} < {Gatekeeper.GATES['min_oos_positive_folds']}")

        oos_exp = agg.get("oos_expectancy", 0)
        oos_dd = agg.get("oos_max_drawdown", 0)
        if oos_exp > 0 and oos_dd > 0:
            dd_ratio = oos_dd / oos_exp
            if dd_ratio > Gatekeeper.GATES["max_drawdown_multiple"]:
                failures.append(f"Excessive drawdown ratio: {round(dd_ratio, 2)} > {Gatekeeper.GATES['max_drawdown_multiple']}")

        if not failures:
            status = "pass"
        elif len(failures) <= 2:
            status = "watchlist"
        else:
            status = "reject"

        return {
            "status": status,
            "failures": failures,
            "gate_count": len(Gatekeeper.GATES),
            "failures_count": len(failures)
        }
