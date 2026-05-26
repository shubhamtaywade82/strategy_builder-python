import logging
from typing import List, Dict, Any
from ..prompts.trade_designer_prompt import TradeDesignerPrompt
from ...generation.ollama_generate_planner import OllamaGeneratePlanner
from ...generation.candidate_validator import CandidateValidator
from ...generation.strategy_templates import StrategyTemplates
from ...backtest.condition_registry import ConditionRegistry
from ...domain import MarketState

class TradeDesigner:
    MAX_CANDIDATES = 3

    SCHEMA = {
        "type": "object",
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "maxItems": MAX_CANDIDATES,
                "items": {"type": "object", "additionalProperties": True}
            }
        }
    }

    def __init__(self, client: Any):
        self.planner = OllamaGeneratePlanner(client)
        self.validator = CandidateValidator()
        self.logger = logging.getLogger(__name__)

    def synthesize(self, market_state: MarketState, confirmed_patterns: List[Dict[str, Any]], observer_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not confirmed_patterns:
            self.logger.info("TradeDesigner: no confirmed patterns — using template fallback")
            return self.template_fallback(market_state)

        prompt = TradeDesignerPrompt.build(
            market_state=market_state,
            confirmed_patterns=confirmed_patterns,
            observer_result=observer_result,
            condition_ids=ConditionRegistry.condition_ids(),
            max_candidates=self.MAX_CANDIDATES
        )

        try:
            raw = self.invoke(prompt, self.SCHEMA)
            raw_list = raw.get("candidates", [])

            valid_candidates = []
            for c in raw_list:
                result = self.validator.validate(c)
                if result["valid"]:
                    valid_candidates.append(result["candidate"])
                else:
                    name = c.get("name", "unknown")
                    self.logger.warning(f"TradeDesigner: invalid candidate '{name}': {', '.join(result['errors'][:2])}")
            
            return valid_candidates if valid_candidates else self.template_fallback(market_state)

        except Exception as e:
            self.logger.warning(f"TradeDesigner failed: {e}")
            return self.template_fallback(market_state)

    def template_fallback(self, market_state: MarketState) -> List[Dict[str, Any]]:
        return StrategyTemplates.for_regime(market_state.regime)[:2]

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
