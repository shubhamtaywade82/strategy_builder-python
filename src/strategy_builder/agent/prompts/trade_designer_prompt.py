import json
from typing import List, Dict, Any
from ...domain import MarketState

class TradeDesignerPrompt:
    SYSTEM = (
        "You are the trade designer on a crypto futures research desk.\n"
        "Convert confirmed patterns into executable strategy candidates.\n"
        "Think like a prop trader: where is the limit entry, where is the structural stop, what are realistic R-multiple targets?\n"
        "Entries must be limit orders at a defined level — never market orders.\n"
        "Stops must be structural (below key swing low / above key swing high), not arbitrary ATR distances.\n"
        "Each candidate must use only condition IDs from the provided list — do not invent new ones.\n"
        "Targets are R-multiples (e.g. 1.0 = 1R, 2.0 = 2R). Use realistic targets: 1R-4R max.\n"
        "Output ONLY a JSON object with a \"candidates\" array. No markdown."
    )

    @staticmethod
    def build(market_state: MarketState, confirmed_patterns: List[Dict[str, Any]], observer_result: Dict[str, Any], condition_ids: List[str], max_candidates: int) -> Dict[str, str]:
        user = (
            f"Market state: {json.dumps(market_state.to_llm_context())}\n"
            f"Confirmed patterns: {json.dumps(confirmed_patterns)}\n"
            f"Observer context: {json.dumps(observer_result)}\n"
            f"Available condition IDs (use only these): {', '.join(condition_ids)}\n\n"
            f"Generate up to {max_candidates} strategy candidates. Each must have:\n"
            "- name (string)\n"
            "- family (one of: session_breakout, session_mean_reversion, mtf_pullback, compression_breakout, failed_breakout, vwap_reclaim, custom)\n"
            "- timeframes (array of: 1m, 5m, 15m, 1h, 4h)\n"
            "- session (array of: asia, london, new_york, any)\n"
            "- entry: { conditions (array from available list), direction (long/short/both) }\n"
            "- exit: { targets (array of R-multiples 0.5-4.0), partial_exits (fractions summing to 1.0), trail (string or \"none\") }\n"
            "- risk: { stop (description string), position_sizing (fixed_risk_percent), max_risk_percent (0.5-1.5) }\n"
            "- filters: { min_volume_zscore, min_atr_percent, required_regime (optional array) }\n"
            "- invalidation (array of strings)\n"
            "- rationale (one paragraph explaining the edge)\n"
            "Output JSON only."
        )
        return {"system": TradeDesignerPrompt.SYSTEM, "user": user}
