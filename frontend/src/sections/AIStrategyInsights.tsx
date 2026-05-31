import { useState } from "react";
import { useGenerateStrategy } from "@/hooks/api";
import {
  Sparkles,
  Loader,
  TrendingUp,
  Shield,
  AlertTriangle,
  CheckCircle,
  Lightbulb,
  Target,
  ChevronRight,
  BrainCircuit,
} from "lucide-react";

interface Props {
  results: any;
}

interface Insight {
  id: string;
  type: "analysis" | "optimization" | "risk" | "recommendation";
  title: string;
  content: string;
  icon: typeof Sparkles;
  color: string;
}

export default function AIStrategyInsights({ results }: Props) {
  const [generated, setGenerated] = useState(false);
  const [insights, setInsights] = useState<Insight[]>([]);

  const generateMutation = useGenerateStrategy({
    onSuccess: (data) => {
      // Parse the AI-generated analysis into structured insights
      const raw = data.analysis;
      const parsed: Insight[] = [];

      // Extract STRATEGY NAME section
      const nameMatch = raw.match(/STRATEGY NAME:\s*(.+)/i);
      if (nameMatch) {
        parsed.push({
          id: "name",
          type: "recommendation",
          title: "AI-Recommended Strategy",
          content: nameMatch[1].trim(),
          icon: Target,
          color: "var(--accent-blue)",
        });
      }

      // Extract ENTRY CONDITIONS
      const entryMatch = raw.match(/ENTRY CONDITIONS:([\s\S]*?)(?=EXIT RULES:|RATIONALE:|$)/i);
      if (entryMatch) {
        parsed.push({
          id: "entry",
          type: "analysis",
          title: "Optimal Entry Conditions",
          content: entryMatch[1].trim().replace(/^\d+\.\s*/gm, "").replace(/\n/g, ", "),
          icon: BrainCircuit,
          color: "var(--accent-purple)",
        });
      }

      // Extract RATIONALE
      const rationaleMatch = raw.match(/RATIONALE:\s*(.+)/i);
      if (rationaleMatch) {
        parsed.push({
          id: "rationale",
          type: "recommendation",
          title: "Why This Strategy Works",
          content: rationaleMatch[1].trim(),
          icon: Lightbulb,
          color: "var(--accent-yellow)",
        });
      }

      // Extract RISK WARNING
      const riskMatch = raw.match(/RISK WARNING:\s*(.+)/i);
      if (riskMatch) {
        parsed.push({
          id: "risk",
          type: "risk",
          title: "Risk Warning",
          content: riskMatch[1].trim(),
          icon: AlertTriangle,
          color: "var(--accent-red)",
        });
      }

      // Add raw analysis as fallback
      if (parsed.length === 0) {
        parsed.push({
          id: "raw",
          type: "analysis",
          title: "AI Strategy Analysis",
          content: raw.length > 500 ? raw.substring(0, 500) + "..." : raw,
          icon: Sparkles,
          color: "var(--accent-cyan)",
        });
      }

      setInsights(parsed);
      setGenerated(true);
    },
  });

  const generateInsights = () => {
    const best = (results.strategies || [])
      .filter((s: any) => s.isViable)
      .sort((a: any, b: any) => b.metrics.expectancy - a.metrics.expectancy)[0];

    if (!best) return;

    const features = best.conditions
      ? best.conditions.map((c: any) => `${c.feature} ${c.operator} ${c.threshold.toFixed(4)}`)
      : ["4h_trend >= 1", "ltf_rvol_20 < 0.002", "vol_ratio > 1.5"];

    generateMutation.mutate({
      symbol: results.symbol,
      rr: results.config?.rr || "2:1",
      features,
      metrics: {
        winRate: best.metrics.winRate,
        profitFactor: best.metrics.profitFactor,
        expectancy: best.metrics.expectancy,
        tradeCount: best.metrics.tradeCount,
        maxDrawdown: best.metrics.maxDrawdown,
        avgBarsHeld: best.metrics.avgBarsHeld,
        sharpe: best.metrics.sharpe,
      },
      model: "qwen3.5:4b",
      sessionId: `insights_${Date.now()}`,
    });
  };

  return (
    <div className="space-y-4">
      {/* Generate Button */}
      {!generated && (
        <button
          onClick={generateInsights}
          disabled={generateMutation.isPending}
          className="w-full metric-card flex items-center justify-center gap-3 py-4 cursor-pointer transition-all hover:opacity-90"
          style={{
            background:
              "linear-gradient(135deg, rgba(168,85,247,0.1), rgba(59,130,246,0.1))",
            border: "1px dashed var(--accent-purple)",
          }}
        >
          {generateMutation.isPending ? (
            <Loader size={18} className="animate-spin" style={{ color: "var(--accent-purple)" }} />
          ) : (
            <Sparkles size={18} style={{ color: "var(--accent-purple)" }} />
          )}
          <span className="text-sm font-medium" style={{ color: "var(--accent-purple)" }}>
            {generateMutation.isPending
              ? "AI Analyzing Best Strategy..."
              : "Generate AI Strategy Insights"}
          </span>
          <ChevronRight size={14} style={{ color: "var(--accent-purple)" }} />
        </button>
      )}

      {/* Generated Insights */}
      {generated && (
        <div className="space-y-3 animate-fade-in">
          <div className="flex items-center gap-2 mb-2">
            <Sparkles size={14} style={{ color: "var(--accent-purple)" }} />
            <span className="text-xs font-semibold" style={{ color: "var(--accent-purple)" }}>
              AI-Generated Strategy Analysis
            </span>
          </div>

          {insights.map((insight) => (
            <div
              key={insight.id}
              className="metric-card"
              style={{ borderLeft: `3px solid ${insight.color}` }}
            >
              <div className="flex items-start gap-3">
                <div
                  className="w-8 h-8 rounded-lg flex-shrink-0 flex items-center justify-center"
                  style={{ background: `${insight.color}15` }}
                >
                  <insight.icon size={14} style={{ color: insight.color }} />
                </div>
                <div className="flex-1 min-w-0">
                  <h4
                    className="text-xs font-semibold mb-1"
                    style={{ color: insight.color }}
                  >
                    {insight.title}
                  </h4>
                  <p
                    className="text-xs leading-relaxed"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {insight.content}
                  </p>
                </div>
              </div>
            </div>
          ))}

          {/* Raw analysis expand */}
          <details className="metric-card">
            <summary
              className="text-xs font-medium cursor-pointer flex items-center gap-2"
              style={{ color: "var(--text-muted)" }}
            >
              <ChevronRight size={12} />
              View Full AI Analysis
            </summary>
            <pre
              className="mt-3 text-[11px] mono leading-relaxed p-3 rounded overflow-auto max-h-[300px]"
              style={{
                background: "var(--bg-primary)",
                color: "var(--text-secondary)",
                border: "1px solid var(--border)",
              }}
            >
              {generateMutation.data?.analysis || ""}
            </pre>
          </details>

          <button
            onClick={generateInsights}
            disabled={generateMutation.isPending}
            className="text-[11px] px-3 py-1.5 rounded-md transition-all hover:opacity-80"
            style={{
              color: "var(--accent-purple)",
              background: "rgba(168,85,247,0.1)",
              border: "1px solid rgba(168,85,247,0.2)",
            }}
          >
            {generateMutation.isPending ? (
              <span className="flex items-center gap-1">
                <Loader size={10} className="animate-spin" /> Regenerating...
              </span>
            ) : (
              "Regenerate Analysis"
            )}
          </button>
        </div>
      )}

      {/* Static AI Tips */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {[
          {
            icon: TrendingUp,
            color: "var(--accent-green)",
            title: "High Win Rate Tip",
            text: "For 1:2 and 1:3 R:R, focus on mean-reversion features: Bollinger Band position, RSI extremes, and volume spikes on rejection candles. These catch short-term reversals with higher probability.",
          },
          {
            icon: Shield,
            color: "var(--accent-blue)",
            title: "Risk Management",
            text: "At 10x leverage, a 1% adverse move = 10% loss. Always set hard liquidation buffer at 2x your stop distance. Use isolated margin, never cross-margin for directional trades.",
          },
          {
            icon: Lightbulb,
            color: "var(--accent-yellow)",
            title: "Feature Engineering",
            text: "The most predictive features for crypto: (1) HTF trend alignment, (2) Volume ratio vs 20-period mean, (3) ATR percentile, (4) Taker buy/sell imbalance. Combine 3+ for robust signals.",
          },
          {
            icon: CheckCircle,
            color: "var(--accent-cyan)",
            title: "Validation Checklist",
            text: "Before trading any strategy: (1) Min 50 trades in backtest, (2) PF > 1.2, (3) Positive expectancy after 0.09% fees, (4) Walk-forward AUC > 0.52 on all folds, (5) Stability > 60%.",
          },
        ].map((tip, i) => (
          <div
            key={i}
            className="metric-card"
            style={{ borderLeft: `3px solid ${tip.color}` }}
          >
            <div className="flex items-start gap-2">
              <tip.icon size={12} style={{ color: tip.color, marginTop: 2 }} />
              <div>
                <h5
                  className="text-[11px] font-semibold mb-1"
                  style={{ color: tip.color }}
                >
                  {tip.title}
                </h5>
                <p
                  className="text-[11px] leading-relaxed"
                  style={{ color: "var(--text-muted)" }}
                >
                  {tip.text}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
