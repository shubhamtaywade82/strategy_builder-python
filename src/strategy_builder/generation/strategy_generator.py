import logging
import json
import random
from typing import List, Dict, Any, Optional
from .ollama_generate_planner import OllamaGeneratePlanner
from .candidate_validator import CandidateValidator
from .llm_invoker import LlmInvoker
from .prompt_builder import PromptBuilder
from .strategy_templates import StrategyTemplates

class StrategyGenerator:
    def __init__(self, client: Any):
        self.client = client
        self.planner = OllamaGeneratePlanner(client)
        self.validator = CandidateValidator()
        self.logger = logging.getLogger(__name__)
        self.llm_invoker = LlmInvoker(planner=self.planner, logger=self.logger)

    def generate(self, features: Dict[str, Any], count: int = 3) -> List[Dict[str, Any]]:
        templates = StrategyTemplates.all()
        compact = self._compact_features(features)
        prompt = PromptBuilder.generation_prompt(
            features=compact,
            templates=templates,
            mode="generate"
        )

        candidates = self.llm_invoker.run(prompt, response_schema=PromptBuilder.strategy_candidates_generate_schema())
        if not candidates:
            self.logger.warning("LLM returned no candidates. Using template fallback.")
            candidates = self._template_fallback_candidates(features=features, count=count)

        valid_candidates = []
        for c in candidates:
            res = self.validator.validate(c)
            if res["valid"]:
                valid_candidates.append(res["candidate"])
        
        if not valid_candidates and candidates:
            self.logger.warn("All LLM proposals failed validation; seeding from built-in templates.")
            candidates = self._template_fallback_candidates(
                features=features,
                count=count,
                offline_reason="Seeded from built-in template after LLM proposals failed validation."
            )
            for c in candidates:
                res = self.validator.validate(c)
                if res["valid"]:
                    valid_candidates.append(res["candidate"])

        self.logger.info(f"Generated {len(candidates)} candidates, {len(valid_candidates)} passed validation")
        return valid_candidates

    def mutate(self, features: Dict[str, Any], template: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        templates = [template] if template else [random.choice(StrategyTemplates.all())]
        compact = self._compact_features(features)
        prompt = PromptBuilder.generation_prompt(
            features=compact,
            templates=templates,
            mode="mutate"
        )

        candidates = self.llm_invoker.run(prompt, response_schema=PromptBuilder.strategy_candidates_generate_schema())
        if not candidates:
            candidates = [self._decorate_fallback_template(t, features) for t in templates]

        valid_candidates = []
        for c in candidates:
            res = self.validator.validate(c)
            if res["valid"]:
                valid_candidates.append(res["candidate"])
        
        return valid_candidates

    def document(self, strategy: Dict[str, Any], backtest_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = PromptBuilder.documentation_prompt(
            strategy=strategy,
            backtest_results=backtest_results
        )

        result = self.llm_invoker.run(prompt)
        return result[0] if isinstance(result, list) and result else (result if isinstance(result, dict) else None)

    def _template_fallback_candidates(self, features: Dict[str, Any], count: int = 3, offline_reason: Optional[str] = None) -> List[Dict[str, Any]]:
        templates = StrategyTemplates.all()
        n = min(count, len(templates))
        return [self._decorate_fallback_template(t, features, offline_reason) for t in templates[:n]]

    def _decorate_fallback_template(self, template: Dict[str, Any], features: Dict[str, Any], offline_reason: Optional[str] = None) -> Dict[str, Any]:
        cand = json.loads(json.dumps(template)) # Deep copy
        instrument = features.get("instrument", "unknown")
        cand["name"] = f"{cand['name']} (offline template)"
        cand["rationale"] = offline_reason or f"Seeded from built-in template while LLM was unavailable (instrument: {instrument})."
        return cand

    def _compact_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        compacted = features.copy()
        if "candles" in compacted: del compacted["candles"]
        if "raw_candles" in compacted: del compacted["raw_candles"]

        if "per_timeframe_summary" in compacted:
            new_summary = {}
            for tf, v in compacted["per_timeframe_summary"].items():
                if isinstance(v, dict):
                    v_copy = v.copy()
                    if "candles" in v_copy: del v_copy["candles"]
                    new_summary[tf] = v_copy
                else:
                    new_summary[tf] = v
            compacted["per_timeframe_summary"] = new_summary

        if "session_ranges" in compacted:
            new_ranges = {}
            for k, v in compacted["session_ranges"].items():
                if isinstance(v, dict):
                    new_ranges[k] = {ik: iv for ik, iv in v.items() if ik in ["session", "date", "high", "low", "open", "close"]}
                else:
                    new_ranges[k] = v
            compacted["session_ranges"] = new_ranges

        return compacted
