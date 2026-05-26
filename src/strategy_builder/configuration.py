import os
import logging
from typing import List, Optional
from urllib.parse import urlparse

from .exceptions import ConfigurationError

class Configuration:
    VALID_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
    DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"

    def __init__(self):
        self.coindcx_api_key = os.getenv("COINDCX_API_KEY")
        self.coindcx_api_secret = os.getenv("COINDCX_API_SECRET")

        self.ollama_model = self._ollama_model_from_env()
        self.ollama_temperature = 0.3
        self.ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "240"))
        self.ollama_cloud = self._truthy_env("STRATEGY_BUILDER_OLLAMA_CLOUD")
        self.ollama_api_key = os.getenv("OLLAMA_API_KEY", "").strip() or None
        self.ollama_base_url = self._default_ollama_base_url(self.ollama_cloud)
        self.ollama_num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
        self.ollama_retries = int(os.getenv("OLLAMA_CLIENT_RETRIES", "2"))
        self.ollama_llm_max_attempts = int(os.getenv("STRATEGY_BUILDER_OLLAMA_LLM_ATTEMPTS", "5"))
        self.ollama_llm_retry_base_seconds = float(os.getenv("STRATEGY_BUILDER_OLLAMA_RETRY_BASE", "0.75"))
        
        self.llm_io_log = not self._falsey_env("STRATEGY_BUILDER_LLM_IO_LOG")
        self.llm_io_log_max_chars = int(os.getenv("STRATEGY_BUILDER_LLM_IO_LOG_MAX_CHARS", "16000"))

        self.default_instruments = ["B-BTC_USDT", "B-ETH_USDT", "B-SOL_USDT"]
        self.default_timeframes = ["1m", "5m", "15m", "1h", "4h"]

        self.backtest_fee_rate = 0.0005
        self.backtest_slippage_bps = 2.0
        self.backtest_spread_bps = float(os.getenv("STRATEGY_BUILDER_SPREAD_BPS", "1.0"))
        self.backtest_slippage_volatility_scale = self._truthy_env("STRATEGY_BUILDER_VOL_SLIPPAGE")
        self.walk_forward_in_sample_ratio = 0.7

        self.max_strategy_candidates = 50
        self.max_agent_iterations = 20

        self.parallel_instrument_max = int(os.getenv("STRATEGY_BUILDER_PARALLEL_INSTRUMENTS", "6"))
        self.backtest_indicator_warmup = int(os.getenv("STRATEGY_BUILDER_BACKTEST_WARMUP", "50"))
        self.backtest_default_stop_price_fraction = float(os.getenv("STRATEGY_BUILDER_DEFAULT_STOP_FRAC", "0.01"))

        self.market_data_source = os.getenv("STRATEGY_BUILDER_MARKET_DATA_SOURCE", "auto")

        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output"))
        self.logger = logging.getLogger("strategy_builder")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _ollama_model_from_env(self) -> str:
        for key in ["OLLAMA_AGENT_MODEL", "OLLAMA_MODEL"]:
            val = os.getenv(key, "").strip()
            if val:
                return val
        return self.DEFAULT_OLLAMA_MODEL

    def _truthy_env(self, name: str) -> bool:
        val = os.getenv(name, "").strip().lower()
        return val in ["1", "true", "yes", "on"]

    def _falsey_env(self, name: str) -> bool:
        val = os.getenv(name, "").strip().lower()
        return val in ["0", "false", "no", "off"]

    def _default_ollama_base_url(self, cloud: bool) -> str:
        if cloud:
            raw = os.getenv("OLLAMA_BASE_URL", "").strip()
            return raw if raw else "https://ollama.com"
        else:
            return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    @property
    def ollama_cloud(self) -> bool:
        return self._ollama_cloud

    @ollama_cloud.setter
    def ollama_cloud(self, value: bool):
        self._ollama_cloud = value

    def is_ollama_cloud(self) -> bool:
        return self.ollama_cloud

    def is_ollama_public_website_host(self) -> bool:
        url = str(self.ollama_base_url)
        if not url.strip():
            return False
        try:
            parsed = urlparse(url)
            host = parsed.hostname
            return host and host.lower().endswith("ollama.com")
        except Exception:
            return False

    def is_ollama_bearer_transport(self) -> bool:
        if self.is_ollama_cloud():
            return True
        return self.is_ollama_public_website_host() and bool(self.ollama_api_key and self.ollama_api_key.strip())

    def validate(self):
        if not self.coindcx_api_key:
            raise ConfigurationError("COINDCX_API_KEY required")
        if not self.coindcx_api_secret:
            raise ConfigurationError("COINDCX_API_SECRET required")

        invalid_tf = [tf for tf in self.default_timeframes if tf not in self.VALID_TIMEFRAMES]
        if invalid_tf:
            raise ConfigurationError(f"Invalid timeframes: {invalid_tf}")
        return True
