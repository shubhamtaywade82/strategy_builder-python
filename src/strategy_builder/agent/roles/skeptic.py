import logging
from typing import Dict, Any, Optional
from ..prompts.skeptic_prompt import SkepticPrompt
from ...generation.ollama_generate_planner import OllamaGeneratePlanner
from ...domain import MarketState, Regime

class Skeptic:
    SCHEMA = {
        "type": "object",
        "required": ["accepted"],
        "properties": {
            "accepted": {"type": "boolean"},
            "rejection_reason": {"type": "string"},
            "concerns": {"type": "array", "items": {"type": "string"}},
            "required_changes": {"type": "array", "items": {"type": "string"}}
        }
    }

    def __init__(self, client: Any):
        self.planner = OllamaGeneratePlanner(client)
        self.logger = logging.getLogger(__name__)

    def review(self, candidate: Dict[str, Any], market_state: MarketState) -> Optional[Dict[str, Any]]:
        hard_reason = self._hard_reject_reason(candidate, market_state)
        if hard_reason:
            self.logger.info(f"Skeptic hard-rejected '{candidate.get('name')}': {hard_reason}")
            return None

        prompt = SkepticPrompt.build(
            candidate=candidate,
            market_state=market_state
        )

        try:
            raw = self.invoke(prompt, self.SCHEMA)
            if raw.get("accepted"):
                merged = candidate.copy()
                merged["skeptic_notes"] = list(raw.get("concerns", []))
                return merged
            else:
                self.logger.info(f"Skeptic rejected '{candidate.get('name')}': {raw.get('rejection_reason')}")
                return None

        except Exception as e:
            self.logger.warning(f"Skeptic failed for '{candidate.get('name')}': {e} — passing candidate through")
            merged = candidate.copy()
            merged["skeptic_notes"] = ["Skeptic unavailable"]
            return merged

    def _hard_reject_reason(self, candidate: Dict[str, Any], market_state: MarketState) -> Optional[str]:
        # 1. RR check
        targets = candidate.get("exit", {}).get("targets", [])
        if not targets or max(float(t) for t in targets) < 1.0:
            return "RR < 1.0"
        
        # 2. Conditions check
        if not candidate.get("entry", {}).get("conditions"):
            return "no entry conditions"
            
        # 3. Chop regime check
        if market_state.regime == Regime.CHOP:
            if candidate.get("family") not in ["session_mean_reversion", "vwap_reclaim"]:
                return "chop regime + wrong family"
                
        return None

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
