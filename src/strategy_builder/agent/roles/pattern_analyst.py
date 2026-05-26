import logging
from typing import List, Dict, Any
from ..prompts.pattern_analyst_prompt import PatternAnalystPrompt
from ...generation.ollama_generate_planner import OllamaGeneratePlanner
from ...domain import MarketState

class PatternAnalyst:
    SCHEMA = {
        "type": "object",
        "required": ["accepted_patterns", "rejected_patterns"],
        "properties": {
            "accepted_patterns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "confidence", "trigger", "continuation", "invalidation"],
                    "properties": {
                        "name": {"type": "string"},
                        "confidence": {"type": "number"},
                        "trigger": {"type": "string"},
                        "continuation": {"type": "boolean"},
                        "invalidation": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "rejected_patterns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "reason"],
                    "properties": {
                        "name": {"type": "string"},
                        "reason": {"type": "string"}
                    }
                }
            }
        }
    }

    def __init__(self, client: Any):
        self.planner = OllamaGeneratePlanner(client)
        self.logger = logging.getLogger(__name__)

    def analyze(self, market_state: MarketState, mined_patterns: List[Dict[str, Any]], observer_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not mined_patterns:
            return []

        prompt = PatternAnalystPrompt.build(
            market_state=market_state,
            mined_patterns=mined_patterns,
            observer_result=observer_result
        )

        try:
            raw = self.invoke(prompt, self.SCHEMA)
            llm_accepted = raw.get("accepted_patterns", [])

            results = []
            for p in llm_accepted:
                base = next((m for m in mined_patterns if str(m.get("name")) == str(p.get("name"))), None)
                if not base:
                    continue
                
                # Merge base and LLM analysis
                merged = base.copy()
                merged.update({
                    "llm_confidence": min(float(p.get("confidence", 0.0)), 1.0),
                    "trigger": str(p.get("trigger", "")),
                    "continuation": bool(p.get("continuation")),
                    "invalidation": list(p.get("invalidation", [])) if p.get("invalidation") else base.get("invalidation", [])
                })
                results.append(merged)
            return results

        except Exception as e:
            self.logger.warning(f"PatternAnalyst failed: {e}")
            return []

    def invoke(self, prompt_dict: Dict[str, str], schema: Dict[str, Any]) -> Dict[str, Any]:
        raw = self.planner.run(
            prompt=prompt_dict["user"],
            system_prompt=prompt_dict["system"],
            schema=schema
        )
        if isinstance(raw, dict):
            return raw
        try:
            import json
            return json.loads(str(raw))
        except:
            return {}
