import time
import json
import logging
from typing import Any, Optional, Dict, List, Union
from .candidate_parser import CandidateParser
from .prompt_builder import PromptBuilder
from ..configuration import Configuration
from ..exceptions import ConfigurationError

class LlmInvoker:
    def __init__(self, planner: Any, logger: Optional[logging.Logger] = None):
        self.planner = planner
        self.logger = logger or logging.getLogger(__name__)

    def run(self, prompt: str, response_schema: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        cfg = Configuration()
        max_attempts = max(cfg.ollama_llm_max_attempts, 1)
        attempts = 0
        schema_for_llm = response_schema or self.planner.ANY_JSON_SCHEMA

        while attempts < max_attempts:
            attempts += 1
            try:
                self._log_llm_inputs(
                    prompt=prompt,
                    system_prompt=PromptBuilder.system_prompt(),
                    context={},
                    response_schema=response_schema
                )

                raw = self.planner.run(
                    prompt=prompt,
                    context={},
                    system_prompt=PromptBuilder.system_prompt(),
                    schema=schema_for_llm
                )

                self._log_llm_raw_output(raw)

                if isinstance(raw, (dict, list)):
                    return self._normalize_llm_candidate_list(raw)

                parsed = CandidateParser.parse(str(raw))
                if parsed:
                    return parsed

                raise ValueError("Empty parse result")

            except Exception as e:
                if attempts >= max_attempts:
                    self.logger.error(f"LLM error after {attempts} attempts: {e}")
                    return []
                
                sleep_sec = min(cfg.ollama_llm_retry_base_seconds * (2 ** (attempts - 1)), 16.0)
                self.logger.warning(f"LLM call failed (attempt {attempts}/{max_attempts}): {e}. Retrying in {sleep_sec}s...")
                time.sleep(sleep_sec)
        
        return []

    def _log_llm_inputs(self, prompt: str, system_prompt: str, context: Dict[str, Any], response_schema: Optional[Dict[str, Any]]):
        cfg = Configuration()
        if not cfg.llm_io_log:
            return

        sys = str(system_prompt)
        usr = str(prompt)
        ctx = json.dumps(context or {}, indent=2)
        schema_note = "schema=default_any_json" if response_schema is None else f"schema=custom"

        self.logger.info(f"LLM IO — system_prompt ({len(sys)} chars): {self._truncate_payload(sys)}")
        self.logger.info(f"LLM IO — user_prompt ({len(usr)} chars): {self._truncate_payload(usr)}")
        self.logger.info(f"LLM IO — context ({len(ctx)} chars): {self._truncate_payload(ctx)}")
        self.logger.info(f"LLM IO — {schema_note}")

    def _log_llm_raw_output(self, raw: Any):
        cfg = Configuration()
        if not cfg.llm_io_log:
            return

        if isinstance(raw, (dict, list)):
            preview = json.dumps(raw, indent=2)
        else:
            preview = str(raw)

        self.logger.info(f"LLM IO — response: {self._truncate_payload(preview)}")

    def _truncate_payload(self, text: str) -> str:
        cfg = Configuration()
        max_chars = max(cfg.llm_io_log_max_chars, 256)
        if len(text) <= max_chars:
            return text
        
        suffix = f"...(truncated, {len(text)} chars total)"
        head = max_chars - len(suffix)
        head = max(head, 256)
        return f"{text[:head]}{suffix}"

    def _normalize_llm_candidate_list(self, raw: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return raw if isinstance(raw, list) else [raw]
