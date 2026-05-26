import json
from typing import Dict, Any
from ...domain import MarketState

class SkepticPrompt:
    SYSTEM = (
        "You are the skeptical desk reviewer on a crypto futures research desk.\n"
        "Your job is to find every reason to reject a strategy candidate.\n"
        "Be ruthless. Challenge: Is the entry already late? Is the stop too tight or too wide?\n"
        "Is the RR realistic for this type of setup? Is this regime-appropriate?\n"
        "Is the pattern actually present or just noise? Is this hindsight fitting?\n"
        "Does this setup have genuine edge or is it random?\n"
        "Accept ONLY if the setup is genuinely clean, well-defined, and executable.\n"
        "Output ONLY a JSON object. No markdown.\n"
        "Fields: accepted (boolean), rejection_reason (string if rejected), concerns (array of strings), required_changes (array of strings)."
    )

    @staticmethod
    def build(candidate: Dict[str, Any], market_state: MarketState) -> Dict[str, str]:
        user = (
            f"Market state: {json.dumps(market_state.to_llm_context())}\n"
            f"Strategy candidate: {json.dumps(candidate)}\n\n"
            "Challenge this setup thoroughly. Is it truly tradeable with defined edge, or is it noise?\n"
            "Output JSON only."
        )
        return {"system": SkepticPrompt.SYSTEM, "user": user}
