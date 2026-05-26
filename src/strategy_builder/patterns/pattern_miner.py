from typing import List, Dict, Any, Optional
from .pattern_library import PatternLibrary
from ..domain import MarketState

class PatternMiner:
    MIN_SCORE = 0.3

    @staticmethod
    def mine(market_state: MarketState) -> List[Dict[str, Any]]:
        candidates = PatternLibrary.matching(
            regime=market_state.regime.value,
            bias=market_state.bias.value
        )

        results = []
        for name, defn in candidates.items():
            score = PatternMiner._score_pattern(name, defn, market_state)
            if score < PatternMiner.MIN_SCORE:
                continue

            results.append({
                "name": name,
                "score": round(score, 3),
                "evidence": PatternMiner._collect_evidence(defn, market_state),
                "description": defn["description"],
                "entry_type": defn["entry_type"],
                "confirmation": defn["confirmation"],
                "invalidation": defn["invalidation"]
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    @staticmethod
    def _score_pattern(name: str, defn: Dict[str, Any], state: MarketState) -> float:
        score = 0.0

        # Base: regime match
        if state.regime.value in defn["required_regime"]:
            score += 0.30

        # Volatility-specific bonuses
        if state.volatility.value == "contracting" and name == "compression_breakout":
            score += 0.20
        if state.volatility.value == "expanding" and name in ["pullback_continuation", "session_breakout"]:
            score += 0.15

        # Volume confirmation
        if state.volume.value == "expanding":
            score += 0.20

        # Higher-TF alignment
        if state.higher_tf_bias and state.higher_tf_bias != "neutral":
            score += 0.15

        # Liquidity context
        liquidity = state.liquidity
        if liquidity and (liquidity.equal_highs or liquidity.equal_lows):
            score += 0.10

        # Session bonus
        if state.session in ["london", "london_ny_overlap"]:
            score += 0.05

        return max(0.0, min(1.0, score))

    @staticmethod
    def _collect_evidence(defn: Dict[str, Any], state: MarketState) -> List[str]:
        evidence = []
        if state.regime.value in defn["required_regime"]:
            evidence.append(f"Regime {state.regime.value} matches pattern requirement")
        if state.higher_tf_bias and state.higher_tf_bias != "neutral":
            evidence.append(f"Higher-TF bias: {state.higher_tf_bias}")
        if state.volatility.value != "normal":
            evidence.append(f"Volatility: {state.volatility.value}")
        if state.volume.value == "expanding":
            evidence.append(f"Volume: {state.volume.value}")
        if state.session in ["london", "london_ny_overlap"]:
            evidence.append(f"Session: {state.session}")
        return evidence
