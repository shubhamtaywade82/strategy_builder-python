import time
import json
import logging
from typing import Any, Optional, Dict

class OllamaGeneratePlanner:
    ANY_JSON_SCHEMA = {
        "anyOf": [
            {"type": "object", "additionalProperties": True},
            {"type": "array"},
            {"type": "string"},
            {"type": "number"},
            {"type": "integer"},
            {"type": "boolean"},
            {"type": "null"}
        ]
    }

    def __init__(self, client: Any):
        self.client = client
        self.logger = logging.getLogger(__name__)

    def run(self, prompt: str, context: Optional[Dict[str, Any]] = None, schema: Optional[Dict[str, Any]] = None, system_prompt: Optional[str] = None) -> Any:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{full_prompt}"
        
        if context:
            full_prompt = f"{full_prompt}\n\nContext (JSON):\n{json.dumps(context, indent=2)}"

        max_attempts = 5
        attempts = 0
        model_name = getattr(self.client.config, "ollama_model", "unknown")

        while attempts < max_attempts:
            attempts += 1
            self.logger.info(f"Ollama: Sending chat request to model '{model_name}' (attempt {attempts}/{max_attempts})...")
            start_time = time.monotonic()
            
            try:
                # Use call_chat_api for better structured support
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                
                user_content = prompt
                if context:
                    user_content = f"{user_content}\n\nContext (JSON):\n{json.dumps(context, indent=2)}"
                messages.append({"role": "user", "content": user_content})

                res_content = self.client.call_chat_api(
                    model=model_name,
                    messages=messages,
                    format=schema or self.ANY_JSON_SCHEMA
                )
                
                elapsed = round(time.monotonic() - start_time, 2)
                self.logger.info(f"Ollama: Received response from model '{model_name}' in {elapsed}s.")
                
                # Attempt to parse JSON if schema was provided
                if schema:
                    try:
                        return json.loads(res_content)
                    except ValueError:
                        self.logger.warning("Ollama: Failed to parse JSON response. Returning raw content.")
                        return res_content
                return res_content

            except Exception as e:
                if attempts < max_attempts:
                    sleep_sec = min(0.5 * (2 ** (attempts - 1)), 10.0)
                    self.logger.warning(f"Ollama: Request failed ({type(e).__name__}: {e}). Retrying in {sleep_sec}s...")
                    time.sleep(sleep_sec)
                else:
                    raise e
