"""
AI Strategy Generator for strategy_builder-python Ollama
==========================================================
Adds to: src/strategy_builder/ollama/strategy_generator.py

Usage:
    from strategy_builder.ollama.strategy_generator import StrategyGenerator
    from strategy_builder.ollama.ssl_bearer_client import OllamaClient  # existing
    
    client = OllamaClient()  # your existing Ollama client
    gen = StrategyGenerator(client)
    
    # Generate strategy from backtest results
    strategy = gen.generate(
        symbol="SOLUSDT",
        features=feature_dict,      # from FeatureBuilder.build()
        metrics=backtest_metrics,   # from your backtest
        model="llama3.1",
    )
    
    # Get market analysis
    analysis = gen.analyze_market(
        symbol="SOLUSDT",
        current_price=150.0,
        regime="trending",
        model="llama3.1",
    )
"""
from typing import Dict, Any, Optional


class StrategyGenerator:
    """Generates trading strategies using LLM + backtest data."""

    SYSTEM_PROMPT = """You are an expert quantitative trading strategist specializing in crypto futures.
You analyze backtest results and feature importance to generate human-readable strategy rules.
You only output strategies that pass statistical validation.
Be concise, specific, and ground all advice in provided data."""

    def __init__(self, ollama_client):
        """Args:
            ollama_client: Instance of strategy_builder.ollama.ssl_bearer_client.OllamaClient
        """
        self.client = ollama_client

    def generate(
        self,
        symbol: str,
        features: Dict[str, Any],
        metrics: Dict[str, float],
        model: str = "llama3.1",
    ) -> Dict[str, str]:
        """Generate strategy rule from features + metrics."""
        prompt = self._build_strategy_prompt(symbol, features, metrics)
        response = self._chat(prompt, model)
        return {"strategy": response, "model": model, "symbol": symbol}

    def analyze_market(
        self,
        symbol: str,
        current_price: float,
        regime: str,
        model: str = "llama3.1",
    ) -> Dict[str, str]:
        """Get AI market analysis."""
        prompt = f"""Analyze {symbol} current conditions:
- Price: ${current_price:.2f}
- Regime: {regime}

Provide:
1. MARKET REGIME interpretation
2. BEST RR SETUP for current conditions
3. KEY LEVELS to watch
4. RISKS that could invalidate the setup"""
        response = self._chat(prompt, model)
        return {"analysis": response, "model": model, "symbol": symbol}

    def optimize_strategy(
        self,
        symbol: str,
        current_strategy: str,
        backtest_metrics: Dict[str, float],
        model: str = "llama3.1",
    ) -> Dict[str, str]:
        """Optimize an existing strategy based on metrics."""
        prompt = f"""Optimize this strategy for {symbol}:

CURRENT STRATEGY:
{current_strategy}

BACKTEST METRICS:
{self._format_metrics(backtest_metrics)}

Suggest specific improvements to:
1. Increase win rate
2. Reduce max drawdown
3. Improve Sharpe ratio

Output only actionable changes with expected impact."""
        response = self._chat(prompt, model)
        return {"optimization": response, "model": model}

    def _build_strategy_prompt(self, symbol: str, features: Dict, metrics: Dict) -> str:
        """Build the strategy generation prompt."""
        # Extract key feature info
        structure = features.get("structure", {})
        volatility = features.get("volatility", {})
        volume = features.get("volume", {})
        smc = features.get("smc", {})
        mtf = features.get("mtf_alignment", {})

        return f"""Generate a trading strategy for {symbol} based on this data:

FEATURES:
- Structure: {structure}
- Volatility: {volatility}
- Volume: {volume}
- SMC: {smc}
- MTF Alignment: {mtf}

BACKTEST METRICS:
{self._format_metrics(metrics)}

Output format:
STRATEGY NAME: [name]
SIDE: [LONG/SHORT]
ENTRY CONDITIONS:
1. [feature] [operator] [threshold]
2. [feature] [operator] [threshold]
...
EXIT RULES:
- Target: [value]
- Stop: [value]
RATIONALE: [explanation]
RISK WARNING: [specific risk]"""

    def _format_metrics(self, metrics: Dict) -> str:
        lines = []
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"- {k}: {v:.4f}")
            else:
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    def _chat(self, prompt: str, model: str) -> str:
        """Send chat request via OllamaSslBearerClient.call_chat_api()."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return self.client.call_chat_api(model=model, messages=messages)
