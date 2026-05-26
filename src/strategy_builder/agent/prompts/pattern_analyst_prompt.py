import json
from typing import List, Dict, Any
from ...domain import MarketState

class PatternAnalystPrompt:
    SYSTEM = (
        "You are the pattern analyst on a crypto futures research desk.\n"
        "You receive a market state and a pre-scored list of candidate patterns.\n"
        "Your task: confirm which patterns are currently forming and explain the trigger logic.\n"
        "Do not invent new patterns. Only accept or reject patterns from the provided list.\n"
        "For each accepted pattern: what is the exact trigger condition, is this continuation or reversal,\n"
        "and what specific conditions would invalidate it?\n"
        "Output ONLY a JSON object with accepted_patterns and rejected_patterns arrays. No markdown."
    )

    @staticmethod
    def build(market_state: MarketState, mined_patterns: List[Dict[str, Any]], observer_result: Dict[str, Any]) -> Dict[str, str]:
        pattern_summary = []
        for p in mined_patterns:
            pattern_summary.append({
                "name": p.get("name"),
                "score": p.get("score"),
                "evidence": p.get("evidence"),
                "description": p.get("description")
            })

        user = (
            f"Market state: {json.dumps(market_state.to_llm_context())}\n"
            f"Observer notes: {json.dumps(observer_result)}\n"
            f"Candidate patterns (pre-scored): {json.dumps(pattern_summary)}\n\n"
            "For each candidate: accept or reject.\n"
            "If accepted, provide: name, confidence (0.0-1.0), trigger (one sentence), continuation (true/false), invalidation (array of strings).\n"
            "If rejected, provide: name, reason (one sentence).\n"
            "Output JSON only."
        )
        return {"system": PatternAnalystPrompt.SYSTEM, "user": user}
