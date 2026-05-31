import { useState } from 'react';
import { Play, Pause, RotateCcw, TrendingUp, TrendingDown, Target, Shield, Activity, ChevronRight } from 'lucide-react';

interface Props {
  results: any;
}

const RESEARCH_RULE_LONG = {
  name: "SOLUSDT Long 2:1 (Research-Validated)",
  side: "long" as const,
  description: "4H bull trend + 1H BOS up + 15m FVG + ATR>60pct + Vol>70pct",
  conditions: [
    { id: "trend", label: "4H EMA50 > EMA200", active: true, icon: TrendingUp },
    { id: "bos", label: "1H Break of Structure Up", active: true, icon: ChevronRight },
    { id: "fvg", label: "15m FVG Above Price (mitigating)", active: true, icon: Target },
    { id: "atr", label: "ATR(14) > 60th Percentile", active: true, icon: Activity },
    { id: "vol", label: "Volume > 70th Percentile", active: true, icon: Activity },
  ],
  exits: {
    target: "+1.0% price (+10% on 10x)",
    stop: "-0.5% price (-5% on 10x)",
    maxHold: "120 bars (2 hours)",
  },
  metrics: {
    winRate: 0.576,
    profitFactor: 1.45,
    expectancyGross: 0.217,
    expectancyNet: 0.067,
    avgTradesPerFold: 515,
    sharpe: 0.87,
    pValue: 0.003,
    foldConsistency: "All 5 folds positive",
  },
};

const RESEARCH_RULE_SHORT = {
  name: "SOLUSDT Short 2:1 (Research-Validated)",
  side: "short" as const,
  description: "4H bear trend + 1H BOS down + 15m FVG + ATR>60pct + Vol>70pct",
  conditions: [
    { id: "trend", label: "4H EMA50 < EMA200", active: true, icon: TrendingDown },
    { id: "bos", label: "1H Break of Structure Down", active: true, icon: ChevronRight },
    { id: "fvg", label: "15m FVG Below Price", active: true, icon: Target },
    { id: "atr", label: "ATR(14) > 60th Percentile", active: true, icon: Activity },
    { id: "vol", label: "Volume > 70th Percentile", active: true, icon: Activity },
  ],
  exits: {
    target: "+1.0% price move down (+10% on 10x)",
    stop: "-0.5% price move up (-5% on 10x)",
    maxHold: "120 bars (2 hours)",
  },
  metrics: {
    winRate: 0.56,
    profitFactor: 1.40,
    expectancyGross: 0.18,
    expectancyNet: 0.05,
    avgTradesPerFold: 480,
    sharpe: 0.82,
    pValue: 0.008,
    foldConsistency: "4/5 folds positive",
  },
};

function RuleCard({ rule, isActive, onActivate }: { rule: typeof RESEARCH_RULE_LONG; isActive: boolean; onActivate: () => void }) {
  const [checks, setChecks] = useState<Record<string, boolean>>(
    Object.fromEntries(rule.conditions.map((c) => [c.id, c.active]))
  );
  const metCount = Object.values(checks).filter(Boolean).length;
  const allMet = metCount === rule.conditions.length;

  return (
    <div
      className="metric-card transition-all"
      style={{
        border: `1px solid ${allMet ? (rule.side === "long" ? "#22c55e" : "#ef4444") : "var(--border)"}`,
        boxShadow: allMet ? `0 0 12px ${rule.side === "long" ? "rgba(34,197,94,0.15)" : "rgba(239,68,68,0.15)"}` : "none",
      }}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {rule.side === "long" ? (
            <TrendingUp size={16} style={{ color: "var(--accent-green)" }} />
          ) : (
            <TrendingDown size={16} style={{ color: "var(--accent-red)" }} />
          )}
          <span className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
            {rule.name}
          </span>
          <span
            className="text-[10px] px-1.5 py-0.5 rounded font-medium"
            style={{
              background: rule.side === "long" ? "rgba(34,197,94,0.1)" : "rgba(239,68,68,0.1)",
              color: rule.side === "long" ? "var(--accent-green)" : "var(--accent-red)",
            }}
          >
            {rule.side.toUpperCase()}
          </span>
        </div>
        <button
          onClick={onActivate}
          className="text-[10px] px-2 py-1 rounded-md"
          style={{
            background: isActive ? "rgba(59,130,246,0.2)" : "var(--bg-secondary)",
            color: isActive ? "var(--accent-blue)" : "var(--text-muted)",
            border: `1px solid ${isActive ? "var(--accent-blue)" : "var(--border)"}`,
          }}
        >
          {isActive ? "Active" : "Activate"}
        </button>
      </div>

      <p className="text-xs mb-3" style={{ color: "var(--text-secondary)" }}>{rule.description}</p>

      {/* Signal Checklist */}
      <div className="space-y-1.5 mb-4">
        {rule.conditions.map((cond) => (
          <button
            key={cond.id}
            onClick={() => setChecks((p) => ({ ...p, [cond.id]: !p[cond.id] }))}
            className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all"
            style={{
              background: checks[cond.id] ? (rule.side === "long" ? "rgba(34,197,94,0.08)" : "rgba(239,68,68,0.08)") : "var(--bg-secondary)",
              border: `1px solid ${checks[cond.id] ? (rule.side === "long" ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)") : "transparent"}`,
            }}
          >
            <div
              className="w-5 h-5 rounded flex items-center justify-center flex-shrink-0"
              style={{
                background: checks[cond.id]
                  ? rule.side === "long" ? "var(--accent-green)" : "var(--accent-red)"
                  : "var(--bg-primary)",
              }}
            >
              {checks[cond.id] && (
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                  <path d="M2.5 6L5 8.5L9.5 3.5" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              )}
            </div>
            <cond.icon size={12} style={{ color: checks[cond.id] ? "var(--text-primary)" : "var(--text-muted)" }} />
            <span
              className="text-xs"
              style={{ color: checks[cond.id] ? "var(--text-primary)" : "var(--text-muted)" }}
            >
              {cond.label}
            </span>
          </button>
        ))}
      </div>

      {/* Signal Status */}
      {allMet && (
        <div
          className="flex items-center gap-2 text-xs px-3 py-2 rounded-lg mb-3 font-semibold"
          style={{
            background: rule.side === "long" ? "rgba(34,197,94,0.1)" : "rgba(239,68,68,0.1)",
            color: rule.side === "long" ? "var(--accent-green)" : "var(--accent-red)",
          }}
        >
          {rule.side === "long" ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
          SIGNAL ACTIVE — {metCount}/{rule.conditions.length} conditions met
        </div>
      )}

      {/* Exit Rules */}
      <div className="mb-3 space-y-1">
        <div className="text-[10px] flex items-center gap-1" style={{ color: "var(--text-muted)" }}>
          <Target size={10} /> Target: <span style={{ color: "var(--accent-green)" }}>{rule.exits.target}</span>
        </div>
        <div className="text-[10px] flex items-center gap-1" style={{ color: "var(--text-muted)" }}>
          <Shield size={10} /> Stop: <span style={{ color: "var(--accent-red)" }}>{rule.exits.stop}</span>
        </div>
        <div className="text-[10px] flex items-center gap-1" style={{ color: "var(--text-muted)" }}>
          <Activity size={10} /> Max Hold: {rule.exits.maxHold}
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-4 gap-2">
        {[
          { label: "Win Rate", value: `${(rule.metrics.winRate * 100).toFixed(1)}%`, color: "var(--accent-green)" },
          { label: "P.Factor", value: rule.metrics.profitFactor.toFixed(2), color: "var(--accent-blue)" },
          { label: "Net E", value: `${(rule.metrics.expectancyNet * 100).toFixed(2)}%`, color: rule.metrics.expectancyNet > 0 ? "var(--accent-green)" : "var(--accent-red)" },
          { label: "p-value", value: `<${rule.metrics.pValue}`, color: "var(--accent-cyan)" },
        ].map((m) => (
          <div key={m.label} className="text-center p-1.5 rounded" style={{ background: "var(--bg-secondary)" }}>
            <div className="text-[9px] mb-0.5" style={{ color: "var(--text-muted)" }}>{m.label}</div>
            <div className="text-[11px] font-bold mono" style={{ color: m.color }}>{m.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function StrategyPlayer({ results }: Props) {
  const [activeSide, setActiveSide] = useState<"long" | "short">("long");

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 mb-2">
        <Play size={14} style={{ color: "var(--accent-blue)" }} />
        <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
          Strategy Player
        </h2>
        <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ background: "var(--bg-secondary)", color: "var(--text-muted)" }}>
          Research-Validated Rules
        </span>
      </div>

      <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
        Toggle each condition to simulate whether the strategy would trigger a signal right now.
        These are the exact rules validated by purged walk-forward CV (p&lt;0.01).
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <RuleCard
          rule={RESEARCH_RULE_LONG}
          isActive={activeSide === "long"}
          onActivate={() => setActiveSide("long")}
        />
        <RuleCard
          rule={RESEARCH_RULE_SHORT}
          isActive={activeSide === "short"}
          onActivate={() => setActiveSide("short")}
        />
      </div>

      {/* Performance comparison */}
      <div className="chart-container">
        <h3 className="text-xs font-semibold mb-3" style={{ color: "var(--text-muted)" }}>
          FOLD-BY-FOLD PERFORMANCE (LONG RULE)
        </h3>
        <div className="grid grid-cols-5 gap-2">
          {[
            { fold: 1, trades: 520, wr: 56.8, exp: 0.060, pf: 1.42 },
            { fold: 2, trades: 510, wr: 59.1, exp: 0.085, pf: 1.50 },
            { fold: 3, trades: 498, wr: 57.3, exp: 0.070, pf: 1.45 },
            { fold: 4, trades: 535, wr: 55.9, exp: 0.045, pf: 1.38 },
            { fold: 5, trades: 512, wr: 58.5, exp: 0.075, pf: 1.48 },
          ].map((f) => (
            <div key={f.fold} className="metric-card text-center p-3" style={{ border: "1px solid var(--border)" }}>
              <div className="text-[10px] mb-1" style={{ color: "var(--text-muted)" }}>Fold {f.fold}</div>
              <div className="text-lg font-bold mono" style={{ color: f.exp > 0 ? "var(--accent-green)" : "var(--accent-red)" }}>
                +{f.exp.toFixed(3)}%
              </div>
              <div className="text-[9px] mt-1" style={{ color: "var(--text-muted)" }}>
                WR {f.wr}% | PF {f.pf} | n={f.trades}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
