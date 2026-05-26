from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

class StrategyCard:
    @staticmethod
    def build(catalog_entry: Dict[str, Any]) -> Dict[str, Any]:
        strategy = catalog_entry["strategy"]
        br = catalog_entry.get("backtest_results") or {}
        metrics = br.get("metrics") or {}
        ranking = catalog_entry.get("ranking") or {}
        doc = catalog_entry.get("documentation") or {}

        return {
            "id": catalog_entry["id"],
            "name": strategy.get("name"),
            "family": strategy.get("family"),
            "status": catalog_entry["status"],
            "summary": doc.get("summary") or StrategyCard._auto_summary(strategy, metrics),
            "edge_explanation": doc.get("edge_explanation") or StrategyCard._auto_edge(strategy),
            "best_conditions": {
                "timeframes": strategy.get("timeframes"),
                "sessions": strategy.get("session") or ["any"],
                "instruments": strategy.get("filters", {}).get("instruments") or ["all"],
                "regimes": strategy.get("filters", {}).get("required_regime") or ["any"]
            },
            "entry_checklist": strategy.get("entry", {}).get("conditions", []),
            "exit_plan": {
                "targets_r": strategy.get("exit", {}).get("targets"),
                "partial_exits": strategy.get("exit", {}).get("partial_exits"),
                "trail": strategy.get("exit", {}).get("trail"),
                "time_stop": strategy.get("exit", {}).get("time_stop_candles")
            },
            "risk_model": {
                "stop": strategy.get("risk", {}).get("stop"),
                "sizing": strategy.get("risk", {}).get("position_sizing"),
                "max_risk": strategy.get("risk", {}).get("max_risk_percent")
            },
            "invalidation": strategy.get("invalidation", []),
            "performance": {
                "expectancy": metrics.get("expectancy"),
                "win_rate": metrics.get("win_rate"),
                "profit_factor": metrics.get("profit_factor"),
                "max_drawdown": metrics.get("max_drawdown"),
                "avg_r": metrics.get("avg_r"),
                "trade_count": metrics.get("trade_count"),
                "sharpe": metrics.get("sharpe_ratio")
            },
            "ranking_score": ranking.get("final_score"),
            "component_scores": ranking.get("component_scores"),
            "failure_modes": doc.get("failure_modes") or StrategyCard._auto_failure_modes(strategy),
            "parameter_bounds": strategy.get("parameter_ranges", {}),
            "created_at": catalog_entry.get("created_at"),
            "updated_at": catalog_entry.get("updated_at")
        }

    @staticmethod
    def _auto_summary(strategy: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        parts = []
        name = strategy.get("name")
        family = strategy.get("family")
        parts.append(f"{name} ({family})")
        
        tfs = strategy.get("timeframes")
        if tfs:
            parts.append(f"trades on {'/'.join(tfs)}")
            
        sessions = strategy.get("session")
        if sessions:
            parts.append(f"during {'/'.join(sessions)} sessions")

        exp = metrics.get("expectancy")
        if exp is not None:
            parts.append(f"with {round(exp, 4)} expectancy")
            wr = metrics.get("win_rate")
            if wr is not None:
                parts.append(f"and {round(wr * 100, 1)}% win rate")

        return " ".join(parts)

    @staticmethod
    def _auto_edge(strategy: Dict[str, Any]) -> str:
        conds = strategy.get("entry", {}).get("conditions", [])
        stop = strategy.get("risk", {}).get("stop")
        sizing = strategy.get("risk", {}).get("position_sizing")
        return f"Enters on confluence of: {', '.join(conds)}. Risk managed with {stop} stop and {sizing} sizing."

    @staticmethod
    def _auto_failure_modes(strategy: Dict[str, Any]) -> List[str]:
        modes = list(strategy.get("invalidation", []))
        sessions = strategy.get("session", [])
        if sessions and "any" not in sessions:
            modes.append("low_liquidity_sessions")
        if strategy.get("filters", {}).get("required_regime"):
            modes.append("regime_mismatch")
        return modes
