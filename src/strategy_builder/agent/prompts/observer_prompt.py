import json
from typing import Dict
from ...domain import MarketState

class ObserverPrompt:
    SYSTEM = (
        "You are the market observer on a crypto futures research desk.\n"
        "Your role is to describe what you see — not to propose trades.\n"
        "Given multi-timeframe market state data, output ONLY valid JSON.\n"
        "Describe: session context, narrative summary, key price levels, and any no-trade conditions.\n"
        "Do not predict price direction. Do not mention indicator names unless they appear in the input.\n"
        "Do not say \"buy\" or \"sell\". Output the JSON object directly with no markdown."
    )

    @staticmethod
    def build(market_state: MarketState) -> Dict[str, str]:
        context = json.dumps(market_state.to_llm_context(), indent=2)
        return {
            "system": ObserverPrompt.SYSTEM,
            "user": f"Market state:\n{context}\n\nOutput JSON only."
        }
