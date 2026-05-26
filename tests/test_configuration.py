import os
import pytest
from strategy_builder.configuration import Configuration, ConfigurationError

@pytest.fixture
def clean_env():
    saved_env = os.environ.copy()
    keys_to_delete = [
        "OLLAMA_AGENT_MODEL", "OLLAMA_MODEL", "STRATEGY_BUILDER_OLLAMA_CLOUD",
        "OLLAMA_API_KEY", "OLLAMA_BASE_URL", "STRATEGY_BUILDER_LLM_IO_LOG",
        "STRATEGY_BUILDER_LLM_IO_LOG_MAX_CHARS", "COINDCX_API_KEY", "COINDCX_API_SECRET"
    ]
    for key in keys_to_delete:
        if key in os.environ:
            del os.environ[key]
    yield
    os.environ.clear()
    os.environ.update(saved_env)

def test_ollama_model_from_env(clean_env):
    os.environ["OLLAMA_MODEL"] = "from-model"
    os.environ["OLLAMA_AGENT_MODEL"] = "from-agent"
    cfg = Configuration()
    assert cfg.ollama_model == "from-agent"

def test_ollama_model_fallback(clean_env):
    os.environ["OLLAMA_MODEL"] = "fallback"
    cfg = Configuration()
    assert cfg.ollama_model == "fallback"

def test_ollama_model_default(clean_env):
    cfg = Configuration()
    assert cfg.ollama_model == Configuration.DEFAULT_OLLAMA_MODEL

def test_ollama_model_ignores_blank(clean_env):
    os.environ["OLLAMA_AGENT_MODEL"] = "  "
    os.environ["OLLAMA_MODEL"] = "valid"
    cfg = Configuration()
    assert cfg.ollama_model == "valid"

def test_truthy_env(clean_env):
    for v in ["1", "true", "yes", "on"]:
        os.environ["SB_TEST_FLAG"] = v
        cfg = Configuration()
        assert cfg._truthy_env("SB_TEST_FLAG") is True

def test_truthy_env_false(clean_env):
    os.environ["SB_TEST_FLAG"] = "0"
    cfg = Configuration()
    assert cfg._truthy_env("SB_TEST_FLAG") is False
    assert cfg._truthy_env("NON_EXISTENT") is False

def test_falsey_env(clean_env):
    for v in ["0", "false", "no", "off"]:
        os.environ["SB_TEST_FALSEY"] = v
        cfg = Configuration()
        assert cfg._falsey_env("SB_TEST_FALSEY") is True

def test_ollama_bearer_transport(clean_env):
    os.environ["STRATEGY_BUILDER_OLLAMA_CLOUD"] = "1"
    os.environ["OLLAMA_API_KEY"] = "k"
    cfg = Configuration()
    cfg.ollama_base_url = "https://ollama.com"
    assert cfg.is_ollama_bearer_transport() is True

def test_ollama_bearer_transport_public_host(clean_env):
    os.environ["OLLAMA_API_KEY"] = "secret"
    cfg = Configuration()
    cfg.ollama_base_url = "https://ollama.com"
    assert cfg.is_ollama_bearer_transport() is True

def test_ollama_bearer_transport_no_key(clean_env):
    if "OLLAMA_API_KEY" in os.environ:
        del os.environ["OLLAMA_API_KEY"]
    cfg = Configuration()
    cfg.ollama_base_url = "https://ollama.com"
    cfg.ollama_api_key = None
    assert cfg.is_ollama_bearer_transport() is False

def test_llm_io_logging_default(clean_env):
    cfg = Configuration()
    assert cfg.llm_io_log is True

def test_llm_io_logging_disabled(clean_env):
    os.environ["STRATEGY_BUILDER_LLM_IO_LOG"] = "0"
    cfg = Configuration()
    assert cfg.llm_io_log is False

def test_default_ollama_base_url_cloud(clean_env):
    cfg = Configuration()
    assert cfg._default_ollama_base_url(True) == "https://ollama.com"

def test_default_ollama_base_url_custom(clean_env):
    os.environ["OLLAMA_BASE_URL"] = "https://custom.example"
    cfg = Configuration()
    assert cfg._default_ollama_base_url(True) == "https://custom.example"

def test_default_ollama_base_url_local(clean_env):
    cfg = Configuration()
    assert cfg._default_ollama_base_url(False) == "http://127.0.0.1:11434"
