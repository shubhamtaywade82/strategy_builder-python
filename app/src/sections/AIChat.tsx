import { useState, useRef, useEffect } from "react";
import { useAiHealth, useAiChat } from "@/hooks/api";
import {
  Send,
  Bot,
  User,
  Sparkles,
  Loader,
  Trash2,
  Cpu,
  AlertCircle,
  ChevronDown,
  Zap,
} from "lucide-react";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const SYSTEM_PROMPT = `You are an expert quantitative trading strategist and market analyst specializing in Binance USD-M perpetual futures. You help traders:

1. INTERPRET strategy research results (backtest metrics, win rates, expectancy)
2. SUGGEST which R:R ratios work best for different market conditions
3. EXPLAIN why certain features (trend, volatility, volume) predict price moves
4. OPTIMIZE entry/exit rules based on feature importance
5. WARN about risks (liquidation, funding rates, slippage)

You are concise, specific, and always ground your advice in the data provided. You never give generic advice - always reference specific metrics, features, or market conditions.

When analyzing a strategy, always consider:
- Win rate vs profit factor tradeoff
- Fee-adjusted expectancy (round-trip 0.09%)
- Walk-forward validation stability
- Feature lookahead risks`;

const QUICK_PROMPTS = [
  "Analyze the best strategy from results",
  "Which R:R gives highest win rate?",
  "Explain the top 3 predictive features",
  "Optimize entry conditions for higher WR",
  "Risk assessment for 10x leverage",
  "Compare long vs short strategies",
];

export default function AIChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hello! I'm your AI strategy research assistant. I can help you interpret backtest results, optimize trading rules, and analyze market conditions. What would you like to explore?",
    },
  ]);
  const [input, setInput] = useState("");
  const [selectedModel, setSelectedModel] = useState("qwen3.5:4b");
  const [isExpanded, setIsExpanded] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const sessionId = useRef(`session_${Date.now()}`);

  const healthQuery = useAiHealth();
  const chatMutation = useAiChat({
    onSuccess: (data) => {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.content },
      ]);
    },
    onError: (err) => {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: ${err.message}. Make sure Ollama is running (${import.meta.env?.VITE_OLLAMA_URL || "localhost:11434"}).`,
        },
      ]);
    },
  });

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessage = (text: string) => {
    if (!text.trim()) return;

    const userMsg: Message = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");

    const allMessages = [
      { role: "system" as const, content: SYSTEM_PROMPT },
      ...messages.map((m) => ({ role: m.role, content: m.content })),
      { role: "user" as const, content: text },
    ];

    chatMutation.mutate({
      messages: allMessages,
      model: selectedModel,
      temperature: 0.7,
      sessionId: sessionId.current,
    });
  };

  const clearChat = () => {
    setMessages([
      {
        role: "assistant",
        content: "Chat cleared. How can I help you with your strategy research?",
      },
    ]);
  };

  return (
    <div className="metric-card" style={{ border: "1px solid var(--border)" }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center gap-2"
        >
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{
              background:
                "linear-gradient(135deg, rgba(168,85,247,0.2), rgba(59,130,246,0.15))",
            }}
          >
            <Sparkles size={16} style={{ color: "var(--accent-purple)" }} />
          </div>
          <div className="text-left">
            <span
              className="text-sm font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              AI Strategy Assistant
            </span>
            <div className="flex items-center gap-1.5">
              <div
                className="w-1.5 h-1.5 rounded-full"
                style={{
                  background: healthQuery.data?.ok
                    ? "var(--accent-green)"
                    : "var(--accent-red)",
                }}
              />
              <span
                className="text-[10px]"
                style={{ color: "var(--text-muted)" }}
              >
                {healthQuery.data?.ok
                  ? `${healthQuery.data.models[0] || "qwen3.5:4b"} ready`
                  : "Ollama offline"}
              </span>
            </div>
          </div>
          <ChevronDown
            size={14}
            style={{
              color: "var(--text-muted)",
              transform: isExpanded ? "rotate(180deg)" : "rotate(0)",
              transition: "transform 0.2s",
            }}
          />
        </button>
        <div className="flex items-center gap-2">
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="input-dark text-[10px] py-1 px-2"
          >
            <option value="qwen3.5:4b">Llama 3.1</option>
            <option value="codellama">CodeLlama</option>
            <option value="mistral">Mistral</option>
            <option value="qwen2.5">Qwen 2.5</option>
          </select>
          <button
            onClick={clearChat}
            className="p-1.5 rounded-md transition-all hover:opacity-70"
            style={{
              color: "var(--text-muted)",
              background: "var(--bg-secondary)",
            }}
            title="Clear chat"
          >
            <Trash2 size={12} />
          </button>
        </div>
      </div>

      {isExpanded && (
        <>
          {/* Messages */}
          <div
            ref={scrollRef}
            className="space-y-3 mb-3 pr-1"
            style={{
              maxHeight: "320px",
              overflowY: "auto",
              background: "var(--bg-primary)",
              borderRadius: "8px",
              padding: "12px",
            }}
          >
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex gap-2 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
              >
                <div
                  className="w-6 h-6 rounded-full flex-shrink-0 flex items-center justify-center"
                  style={{
                    background:
                      msg.role === "assistant"
                        ? "rgba(168,85,247,0.15)"
                        : "rgba(59,130,246,0.15)",
                  }}
                >
                  {msg.role === "assistant" ? (
                    <Bot
                      size={12}
                      style={{ color: "var(--accent-purple)" }}
                    />
                  ) : (
                    <User
                      size={12}
                      style={{ color: "var(--accent-blue)" }}
                    />
                  )}
                </div>
                <div
                  className="max-w-[85%] text-xs leading-relaxed p-2.5 rounded-lg"
                  style={{
                    background:
                      msg.role === "assistant"
                        ? "var(--bg-card)"
                        : "rgba(59,130,246,0.1)",
                    color: "var(--text-secondary)",
                    border:
                      msg.role === "assistant"
                        ? "1px solid var(--border)"
                        : "1px solid rgba(59,130,246,0.2)",
                  }}
                >
                  {msg.content.split("\n").map((line, j) => (
                    <p key={j} className={j > 0 ? "mt-1" : ""}>
                      {line}
                    </p>
                  ))}
                </div>
              </div>
            ))}
            {chatMutation.isPending && (
              <div className="flex gap-2">
                <div
                  className="w-6 h-6 rounded-full flex items-center justify-center"
                  style={{ background: "rgba(168,85,247,0.15)" }}
                >
                  <Loader
                    size={12}
                    className="animate-spin"
                    style={{ color: "var(--accent-purple)" }}
                  />
                </div>
                <div
                  className="text-xs p-2.5 rounded-lg"
                  style={{
                    background: "var(--bg-card)",
                    border: "1px solid var(--border)",
                  }}
                >
                  <span
                    className="animate-pulse"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Analyzing...
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Quick Prompts */}
          {messages.length < 3 && (
            <div className="flex gap-1.5 flex-wrap mb-3">
              {QUICK_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => sendMessage(prompt)}
                  className="text-[10px] px-2.5 py-1.5 rounded-md flex items-center gap-1 transition-all hover:opacity-80"
                  style={{
                    background: "var(--bg-secondary)",
                    color: "var(--text-muted)",
                    border: "1px solid var(--border)",
                  }}
                >
                  <Zap size={8} />
                  {prompt}
                </button>
              ))}
            </div>
          )}

          {/* Input */}
          <div className="flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendMessage(input)}
              placeholder="Ask about strategy analysis, feature importance, market conditions..."
              className="input-dark flex-1 text-xs"
              disabled={chatMutation.isPending || !healthQuery.data?.ok}
            />
            <button
              onClick={() => sendMessage(input)}
              disabled={
                chatMutation.isPending ||
                !input.trim() ||
                !healthQuery.data?.ok
              }
              className="btn-primary flex items-center gap-1.5 px-3"
              style={{
                opacity:
                  chatMutation.isPending || !healthQuery.data?.ok ? 0.5 : 1,
              }}
            >
              {chatMutation.isPending ? (
                <Loader size={12} className="animate-spin" />
              ) : (
                <Send size={12} />
              )}
            </button>
          </div>

          {!healthQuery.data?.ok && (
            <div
              className="mt-2 flex items-center gap-2 text-[10px] px-2 py-1.5 rounded"
              style={{
                background: "rgba(239,68,68,0.1)",
                color: "var(--accent-red)",
                border: "1px solid rgba(239,68,68,0.2)",
              }}
            >
              <AlertCircle size={10} />
              Ollama not connected. Install from ollama.com and run `ollama
              pull llama3.1`
            </div>
          )}
        </>
      )}
    </div>
  );
}
