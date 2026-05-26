from typing import Dict, Any, Optional

class Scorer:
    WEIGHTS = {
        "expectancy": 0.25,
        "profit_factor": 0.20,
        "oos_stability": 0.15,
        "drawdown_resilience": 0.15,
        "session_consistency": 0.10,
        "parameter_robustness": 0.10,
        "trade_frequency": 0.02,
        "regime_robustness": 0.03
    }

    @staticmethod
    def score(walk_forward_result: Dict[str, Any], session_results: Optional[Dict[str, Any]] = None, robustness_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        agg = walk_forward_result.get("aggregate", {})

        components = {
            "expectancy": Scorer._normalize_expectancy(agg.get("oos_expectancy")),
            "profit_factor": Scorer._normalize_profit_factor(agg.get("oos_profit_factor")),
            "oos_stability": walk_forward_result.get("stability_score", 0.0),
            "drawdown_resilience": Scorer._normalize_drawdown(agg.get("oos_max_drawdown"), agg.get("oos_expectancy")),
            "session_consistency": Scorer._score_session_consistency(session_results) if session_results else 0.5,
            "parameter_robustness": robustness_result.get("robustness_score", 0.5) if robustness_result else 0.5,
            "trade_frequency": Scorer._normalize_trade_frequency(agg.get("oos_trade_count")),
            "regime_robustness": Scorer._normalize_regime_robustness(walk_forward_result.get("regime_slices"))
        }

        final = sum(components[key] * Scorer.WEIGHTS[key] for key in Scorer.WEIGHTS)

        return {
            "final_score": round(final, 4),
            "component_scores": components
        }

    @staticmethod
    def _normalize_expectancy(exp: Optional[float]) -> float:
        if exp is None or exp <= 0:
            return 0.0
        return min(exp / 0.5, 1.0)

    @staticmethod
    def _normalize_profit_factor(pf: Optional[float]) -> float:
        if pf is None or pf <= 1.0:
            return 0.0
        return min((pf - 1.0) / 2.0, 1.0)

    @staticmethod
    def _normalize_drawdown(max_dd: Optional[float], expectancy: Optional[float]) -> float:
        if max_dd is None or expectancy is None or expectancy == 0:
            return 0.0
        
        ratio = 10.0 if max_dd == 0 else abs(expectancy) / max_dd
        return min(ratio / 2.0, 1.0)

    @staticmethod
    def _normalize_trade_frequency(count: Optional[int]) -> float:
        if count is None or count < 10:
            return 0.0
        
        if count < 30:
            return count / 30.0
        elif count <= 200:
            return 1.0
        else:
            return max(200.0 / count, 0.5)

    @staticmethod
    def _score_session_consistency(session_results: Dict[str, Any]) -> float:
        if not session_results:
            return 0.5
        
        profitable_sessions = sum(1 for s, m in session_results.items() if isinstance(m, dict) and float(m.get("expectancy") or 0) > 0)
        total_sessions = len(session_results)
        if total_sessions == 0:
            return 0.0
        return round(profitable_sessions / total_sessions, 4)

    @staticmethod
    def _normalize_regime_robustness(regime_slices: Any) -> float:
        if not isinstance(regime_slices, dict) or "fraction_positive_expectancy" not in regime_slices:
            return 0.5
        
        val = float(regime_slices["fraction_positive_expectancy"])
        return max(0.0, min(1.0, val))
