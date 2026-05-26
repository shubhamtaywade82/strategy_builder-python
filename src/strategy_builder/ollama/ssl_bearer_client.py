import time
import json
import requests
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urljoin

class OllamaError(Exception):
    pass

class OllamaTimeoutError(OllamaError):
    pass

class OllamaInvalidJSONError(OllamaError):
    pass

class OllamaSslBearerClient:
    def __init__(self, config: Any, bearer_token: Optional[str] = None):
        self.config = config
        self.bearer_token = bearer_token.strip() if bearer_token else None
        self.base_url = config.ollama_base_url
        self.timeout = config.ollama_timeout

    def health(self, return_meta: bool = False) -> Union[bool, Dict[str, Any]]:
        url = urljoin(self.base_url.rstrip('/') + '/', "api/ping")
        start_time = time.monotonic()
        
        try:
            headers = self._auth_headers()
            response = requests.get(url, headers=headers, timeout=self.timeout)
            ok = response.status_code == 200
            
            if not return_meta:
                return ok

            return {
                "ok": ok,
                "meta": {
                    "endpoint": "/api/ping",
                    "status_code": response.status_code,
                    "latency_ms": int((time.monotonic() - start_time) * 1000)
                }
            }
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if not return_meta:
                return False
            return {
                "ok": False,
                "meta": {
                    "endpoint": "/api/ping",
                    "error": str(e),
                    "latency_ms": int((time.monotonic() - start_time) * 1000)
                }
            }

    def list_models(self) -> List[str]:
        url = urljoin(self.base_url.rstrip('/') + '/', "api/tags")
        headers = self._auth_headers()
        
        try:
            response = requests.get(url, headers=headers, timeout=self.timeout)
            if response.status_code != 200:
                raise OllamaError(f"Failed to fetch models: HTTP {response.status_code}")
            
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except ValueError as e:
            raise OllamaInvalidJSONError(f"Failed to parse models response: {e}")
        except requests.exceptions.Timeout:
            raise OllamaTimeoutError(f"Request timed out after {self.timeout}s")
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"Connection failed: {e}")

    def call_chat_api(self, model: str, messages: List[Dict[str, str]], format: Optional[Dict[str, Any]] = None, options: Optional[Dict[str, Any]] = None) -> str:
        url = urljoin(self.base_url.rstrip('/') + '/', "api/chat")
        headers = self._auth_headers()
        headers["Content-Type"] = "application/json"

        body = {
            "model": model or self.config.ollama_model,
            "messages": messages,
            "stream": False
        }

        if format:
            body["format"] = format
        
        # Merge options
        opt = {
            "temperature": self.config.ollama_temperature,
            "num_ctx": self.config.ollama_num_ctx
        }
        if options:
            opt.update(options)
        body["options"] = opt

        try:
            response = requests.post(url, headers=headers, json=body, timeout=self.timeout)
            if response.status_code != 200:
                raise OllamaError(f"Ollama API returned HTTP {response.status_code}: {response.text}")
            
            data = response.json()
            return data["message"]["content"]
        except ValueError as e:
            raise OllamaInvalidJSONError(f"Failed to parse API response: {e}")
        except requests.exceptions.Timeout:
            raise OllamaTimeoutError(f"Request timed out after {self.timeout}s")
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"Connection failed: {e}")

    def _auth_headers(self) -> Dict[str, str]:
        headers = {}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return headers
