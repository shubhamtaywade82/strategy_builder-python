/**
 * Ollama AI Service
 * Supports local Ollama (localhost:11434) and cloud endpoints.
 */

const OLLAMA_BASE = process.env.OLLAMA_URL || "http://localhost:11434";
const DEFAULT_MODEL = process.env.OLLAMA_MODEL || "qwen3.5:4b";

export interface OllamaMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface OllamaResponse {
  message: OllamaMessage;
  done: boolean;
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  eval_count?: number;
}

export async function chatCompletion(
  messages: OllamaMessage[],
  model: string = DEFAULT_MODEL,
  options: { temperature?: number; stream?: boolean } = {}
): Promise<OllamaResponse> {
  const res = await fetch(`${OLLAMA_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages,
      stream: options.stream ?? false,
      options: {
        temperature: options.temperature ?? 0.7,
      },
    }),
  });

  if (!res.ok) {
    const err = await res.text().catch(() => res.statusText);
    throw new Error(`Ollama error (${res.status}): ${err}`);
  }

  return res.json();
}

export async function generateStrategy(
  symbol: string,
  rr: string,
  features: string[],
  metrics: Record<string, number>,
  model: string = DEFAULT_MODEL
): Promise<string> {
  const systemPrompt = `You are an expert quantitative trading strategist specializing in crypto futures. You analyze backtest results and feature importance to generate human-readable strategy rules. You only output strategies that pass statistical validation. You are concise and specific.`;

  const userPrompt = `Analyze this strategy research data for ${symbol} with R:R ${rr}:

BACKTEST METRICS:
${Object.entries(metrics)
  .map(([k, v]) => `- ${k}: ${typeof v === "number" ? v.toFixed(4) : v}`)
  .join("\n")}

TOP PREDICTIVE FEATURES:
${features.join("\n")}

Generate a human-readable trading rule in this exact format:

STRATEGY NAME: [descriptive name]
SIDE: [LONG or SHORT]
ENTRY CONDITIONS:
1. [feature] [operator] [threshold]
2. [feature] [operator] [threshold]
...

EXIT RULES:
- Target: [price target]
- Stop: [stop loss]
- Max Hold: [time limit]

RATIONALE: [1-2 sentence explanation of why these conditions work for ${symbol}]

RISK WARNING: [specific risk for this setup]`;

  const res = await chatCompletion(
    [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ],
    model,
    { temperature: 0.4 }
  );

  return res.message.content;
}

export async function analyzeMarket(
  symbol: string,
  priceData: { currentPrice: number; change24h: number; high24h: number; low24h: number; volume24h: number },
  model: string = DEFAULT_MODEL
): Promise<string> {
  const systemPrompt = `You are a crypto market analyst. Provide brief, actionable analysis. Focus on what matters for short-term futures trading (1-4 hour holds).`;

  const userPrompt = `Analyze ${symbol} market conditions:
- Price: $${priceData.currentPrice}
- 24h Change: ${priceData.change24h > 0 ? "+" : ""}${(priceData.change24h * 100).toFixed(2)}%
- 24h High: $${priceData.high24h}
- 24h Low: $${priceData.low24h}
- 24h Volume: $${(priceData.volume24h / 1e6).toFixed(1)}M

Provide:
1. MARKET REGIME: (trending up / ranging / trending down / volatile)
2. KEY LEVELS: support and resistance
3. BEST RR SETUP: which risk:reward ratio is likely optimal right now
4. CAUTION: what could invalidate the setup`;

  const res = await chatCompletion(
    [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ],
    model,
    { temperature: 0.5 }
  );

  return res.message.content;
}

export async function listModels(): Promise<string[]> {
  try {
    const res = await fetch(`${OLLAMA_BASE}/api/tags`, { method: "GET" });
    if (!res.ok) return [];
    const data = await res.json();
    return (data.models || []).map((m: any) => m.name || m.model);
  } catch {
    return [];
  }
}

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${OLLAMA_BASE}/api/tags`, { method: "GET", signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}
