import logging
from typing import Dict, Any, List
from ..prompts.observer_prompt import ObserverPrompt
from ...generation.ollama_generate_planner import OllamaGeneratePlanner
from ...domain import MarketState

class Observer:
    SCHEMA = {
        "type": "object",
        "required": ["narrative", "session_context", "key_levels", "no_trade_context"],
        "properties": {
            "narrative": {"type": "string"},
            "session_context": {"type": "string"},
            "key_levels": {"type": "array", "items": {"type": "string"}},
            "no_trade_context": {"type": "array", "items": {"type": "string"}}
        }
    }

    def __init__(self, client: Any):
        self.planner = OllamaGeneratePlanner(client)
        self.logger = logging.getLogger(__name__)

    def classify(self, market_state: MarketState) -> Dict[str, Any]:
        prompt = ObserverPrompt.build(market_state)
        try:
            raw = self.invoke(prompt, self.SCHEMA)

            return {
                "confirmed_regime": market_state.regime,
                "narrative": raw.get("narrative", ""),
                "session_context": raw.get("session_context", ""),
                "key_levels": list(raw.get("key_levels", [])),
                "no_trade_context": list(raw.get("no_trade_context", []))
            }
        except Exception as e:
            self.logger.warning(f"Observer failed: {e}")
            return {
                "confirmed_regime": market_state.regime,
                "narrative": "",
                "session_context": "",
                "key_levels": [],
                "no_trade_context": []
            }

    def invoke(self, prompt_dict: Dict[str, str], schema: Dict[str, Any]) -> Dict[str, Any]:
        raw = self.planner.run(
            prompt=prompt_dict["user"],
            system_prompt=prompt_dict["system"],
            schema=schema
        )
        if isinstance(raw, dict):
            return raw
        # If it returned a string, it might be raw JSON or something else
        try:
            import json
            return json.loads(str(raw))
        except:
            return {}
