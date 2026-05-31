from __future__ import annotations
from typing import Dict, List, Optional
import httpx
from strategy_api.config import settings


async def chat_completion(
    messages: List[dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
) -> dict:
    model = model or settings.ollama_agent_model
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as c:
        r = await c.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        if r.status_code != 200:
            raise RuntimeError(f"Ollama error ({r.status_code}): {r.text}")
        return r.json()


async def generate_strategy(
    symbol: str,
    rr: str,
    features: List[str],
    metrics: Dict[str, float],
    model: Optional[str] = None,
) -> str:
    system_prompt = (
        "You are an expert quantitative trading strategist specializing in crypto "
        "futures. You analyze backtest results and feature importance to generate "
        "human-readable strategy rules. You only output strategies that pass "
        "statistical validation. You are concise and specific."
    )

    metric_lines = "\n".join(
        f"- {k}: {v:.4f}" if isinstance(v, (int, float)) else f"- {k}: {v}"
        for k, v in metrics.items()
    )

    user_prompt = (
        f"Analyze this strategy research data for {symbol} with R:R {rr}:\n"
        "\n"
        "BACKTEST METRICS:\n"
        f"{metric_lines}\n"
        "\n"
        "TOP PREDICTIVE FEATURES:\n"
        f"{chr(10).join(features)}\n"
        "\n"
        "Generate a human-readable trading rule in this exact format:\n"
        "\n"
        "STRATEGY NAME: [descriptive name]\n"
        "SIDE: [LONG or SHORT]\n"
        "ENTRY CONDITIONS:\n"
        "1. [feature] [operator] [threshold]\n"
        "2. [feature] [operator] [threshold]\n"
        "...\n"
        "\n"
        "EXIT RULES:\n"
        "- Target: [price target]\n"
        "- Stop: [stop loss]\n"
        "- Max Hold: [time limit]\n"
        "\n"
        f"RATIONALE: [1-2 sentence explanation of why these conditions work for {symbol}]\n"
        "\n"
        "RISK WARNING: [specific risk for this setup]"
    )

    res = await chat_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model,
        temperature=0.4,
    )
    return res["message"]["content"]


async def analyze_market(
    symbol: str,
    price_data: dict,
    model: Optional[str] = None,
) -> str:
    system_prompt = (
        "You are a crypto market analyst. Provide brief, actionable analysis. Focus "
        "on what matters for short-term futures trading (1-4 hour holds)."
    )

    chg = price_data["change24h"]
    sign = "+" if chg > 0 else ""
    user_prompt = (
        f"Analyze {symbol} market conditions:\n"
        f"- Price: ${price_data['currentPrice']}\n"
        f"- 24h Change: {sign}{chg * 100:.2f}%\n"
        f"- 24h High: ${price_data['high24h']}\n"
        f"- 24h Low: ${price_data['low24h']}\n"
        f"- 24h Volume: ${price_data['volume24h'] / 1e6:.1f}M\n"
        "\n"
        "Provide:\n"
        "1. MARKET REGIME: (trending up / ranging / trending down / volatile)\n"
        "2. KEY LEVELS: support and resistance\n"
        "3. BEST RR SETUP: which risk:reward ratio is likely optimal right now\n"
        "4. CAUTION: what could invalidate the setup"
    )

    res = await chat_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model,
        temperature=0.5,
    )
    return res["message"]["content"]


async def list_models() -> List[str]:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as c:
            r = await c.get(f"{settings.ollama_base_url}/api/tags")
            if r.status_code != 200:
                return []
            return [m.get("name") or m.get("model") for m in r.json().get("models", [])]
    except Exception:
        return []


async def health_check() -> bool:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as c:
            r = await c.get(f"{settings.ollama_base_url}/api/tags")
            return r.status_code == 200
    except Exception:
        return False
