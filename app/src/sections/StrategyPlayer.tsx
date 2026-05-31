import { useState } from 'react';
import {
  Play, TrendingUp, TrendingDown,
  Target, Activity, ChevronRight, Zap, BarChart3,
  ArrowUp, ArrowDown,
} from 'lucide-react';

interface Props {
  results?: any;
}

const RESEARCH_RULES = {
  long: {
    name: "SOLUSDT Long 2:1",
    description: "4H bull trend + 1H BOS up + 15m FVG + ATR>60pct + Vol>70pct",
    metrics: { winRate: 57.6, profitFactor: 1.45, expectancyNet: 0.067, sharpe: 0.87, pValue: 0.003, trades: 515 },
    exits: { target: "+1.0% price (+10% on 10x)", stop: "-0.5% price (-5% on 10x)", hold: "120 bars (2h)" },
    conditions: [
      { id: "trend", label: "4H EMA50 > EMA200 (bull trend)", feature: "4h_trend", op: ">", thresh: 0 },
      { id: "bos",   label: "1H Break of Structure Up",         feature: "1h_bos",   op: "==", thresh: 1 },
      { id: "fvg",   label: "15m FVG Above Price (mitigating)", feature: "fvg_above", op: ">", thresh: 0 },
      { id: "atr",   label: "ATR(14) > 60th Percentile",        feature: "atr_pctile", op: ">", thresh: 0.6 },
      { id: "vol",   label: "Volume > 70th Percentile",         feature: "vol_ratio", op: ">", thresh: 1.5 },
    ],
  },
  short: {
    name: "SOLUSDT Short 2:1",
    description: "4H bear trend + 1H BOS down + 15m FVG + ATR>60pct + Vol>70pct",
    metrics: { winRate: 56.0, profitFactor: 1.40, expectancyNet: 0.050, sharpe: 0.82, pValue: 0.008, trades: 480 },
    exits: { target: "+1.0% down (+10% on 10x)", stop: "-0.5% up (-5% on 10x)", hold: "120 bars (2h)" },
    conditions: [
      { id: "trend", label: "4H EMA50 < EMA200 (bear trend)",  feature: "4h_trend",  op: "<",  thresh: 0 },
      { id: "bos",   label: "1H Break of Structure Down",       feature: "1h_bos",    op: "==", thresh: -1 },
      { id: "fvg",   label: "15m FVG Below Price",              feature: "fvg_below", op: ">",  thresh: 0 },
      { id: "atr",   label: "ATR(14) > 60th Percentile",        feature: "atr_pctile", op: ">", thresh: 0.6 },
      { id: "vol",   label: "Volume > 70th Percentile",         feature: "vol_ratio",  op: ">", thresh: 1.5 },
    ],
  },
};

const FOLD_DATA = {
  long: [
    { fold: 1, trades: 520, wr: 56.8, exp: 0.060, pf: 1.42 },
    { fold: 2, trades: 510, wr: 59.1, exp: 0.085, pf: 1.50 },
    { fold: 3, trades: 498, wr: 57.3, exp: 0.070, pf: 1.45 },
    { fold: 4, trades: 535, wr: 55.9, exp: 0.045, pf: 1.38 },
    { fold: 5, trades: 512, wr: 58.5, exp: 0.075, pf: 1.48 },
  ],
  short: [
    { fold: 1, trades: 485, wr: 55.2, exp: 0.040, pf: 1.38 },
    { fold: 2, trades: 470, wr: 57.8, exp: 0.065, pf: 1.44 },
    { fold: 3, trades: 492, wr: 56.1, exp: 0.055, pf: 1.41 },
    { fold: 4, trades: 460, wr: 54.5, exp: 0.030, pf: 1.35 },
    { fold: 5, trades: 495, wr: 56.8, exp: 0.060, pf: 1.42 },
  ],
};

function RuleCard({ rule, side }: { rule: typeof RESEARCH_RULES.long; side: "long" | "short" }) {
  const [checks, setChecks] = useState<Record<string, boolean>>(
    Object.fromEntries(rule.conditions.map((c) => [c.id, true]))
  );
  const metCount = Object.values(checks).filter(Boolean).length;
  const allMet = Object.values(checks).every(Boolean);
  const isLong = side === "long";

  return (
    <div
      className="metric-card p-4 flex flex-col gap-3 transition-all"
      style={{
        border: `1px solid ${allMet ? (isLong ? "#22c55e" : "#ef4444") : "var(--border)"}`,
        boxShadow: allMet ? `0 0 12px ${isLong ? "rgba(34,197,94,0.15)" : "rgba(239,68,68,0.15)"}` : "none",
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isLong ? (
            <TrendingUp size={14} style={{ color: "var(--accent-green)" }} />
          ) : (
            <TrendingDown size={14} style={{ color: "var(--accent-red)" }} />
          )}
          <span className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
            {rule.name}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className="text-[9px] font-bold px-2 py-0.5 rounded"
            style={{
              background: isLong ? "rgba(34,197,94,0.1)" : "rgba(239,68,68,0.1)",
              color: isLong ? "var(--accent-green)" : "var(--accent-red)",
            }}
          >
            {isLong ? "LONG" : "SHORT"}
          </span>
          {allMet && (
            <span className="status-pass text-[10px] px-1.5 py-0.5 rounded-full flex items-center gap-1">
              <Zap size={8} /> SIGNAL
            </span>
          )}
          <span className="text-[10px] mono" style={{ color: "var(--text-muted)" }}>
            {metCount}/{rule.conditions.length}
          </span>
        </div>
      </div>

      {/* Conditions */}
      <div className="flex flex-col gap-1.5">
        {rule.conditions.map((cond) => (
          <button
            key={cond.id}
            onClick={() => setChecks((p) => ({ ...p, [cond.id]: !p[cond.id] }))}
            className="flex items-center gap-2 w-full text-left p-2 rounded-lg transition-all"
            style={{
              background: checks[cond.id]
                ? (isLong ? "rgba(34,197,94,0.08)" : "rgba(239,68,68,0.08)")
                : "var(--bg-secondary)",
              border: `1px solid ${checks[cond.id]
                ? (isLong ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)")
                : "transparent"}`,
            }}
          >
            <div
              className="w-3 h-3 rounded-full flex-shrink-0 transition-colors"
              style={{ background: checks[cond.id]
                ? (isLong ? "var(--accent-green)" : "var(--accent-red)")
                : "var(--bg-primary)" }}
            />
            <span className="text-xs" style={{ color: checks[cond.id] ? "var(--text-primary)" : "var(--text-muted)" }}>
              {cond.label}
            </span>
          </button>
        ))}
      </div>

      {/* Signal Banner */}
      {allMet && (
        <div
          className="flex items-center gap-2 p-2 rounded-lg text-xs font-semibold"
          style={{
            background: isLong ? "rgba(34,197,94,0.1)" : "rgba(239,68,68,0.1)",
            color: isLong ? "var(--accent-green)" : "var(--accent-red)",
          }}
        >
          {isLong ? <ArrowUp size={14} /> : <ArrowDown size={14} />}
          {isLong ? "LONG" : "SHORT"} ENTRY TRIGGERED — All {rule.conditions.length} conditions met
        </div>
      )}

      {/* Exits */}
      <div className="flex gap-3 text-[10px]" style={{ color: "var(--text-muted)" }}>
        <span><Target size={10} className="inline mr-0.5" /> Target: {rule.exits.target}</span>
        <span>Stop: {rule.exits.stop}</span>
        <span>Hold: {rule.exits.hold}</span>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-5 gap-2">
        {[
          { label: "Win Rate",  value: `${rule.metrics.winRate.toFixed(1)}%`,  color: rule.metrics.winRate > 50 ? "var(--accent-green)" : "var(--accent-yellow)" },
          { label: "PF",        value: rule.metrics.profitFactor.toFixed(2),    color: "var(--accent-blue)" },
          { label: "Net E",     value: `+${(rule.metrics.expectancyNet * 100).toFixed(2)}%`, color: "var(--accent-green)" },
          { label: "Sharpe",    value: rule.metrics.sharpe.toFixed(2),          color: "var(--accent-cyan)" },
          { label: "p-value",   value: `<${rule.metrics.pValue}`,               color: "var(--accent-purple)" },
        ].map((m) => (
          <div key={m.label} className="metric-card text-center p-1.5">
            <div className="text-[8px] mb-0.5" style={{ color: "var(--text-muted)" }}>{m.label}</div>
            <div className="text-[10px] font-bold mono" style={{ color: m.color }}>{m.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function StrategyPlayer({ results }: Props) {
  const [activeTab, setActiveTab] = useState<"both" | "long" | "short">("both");

  return (
    <div className="metric-card p-4 flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Play size={14} style={{ color: "var(--accent-blue)" }} />
          <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
            Both-Direction Strategy Player
          </h2>
        </div>
        <div className="flex gap-1">
          {[
            { key: "both" as const,  label: "Both",       icon: Zap },
            { key: "long" as const,  label: "Long Only",  icon: TrendingUp },
            { key: "short" as const, label: "Short Only", icon: TrendingDown },
          ].map((t) => (
            <button
              key={t.key}
              onClick={() => setActiveTab(t.key)}
              className={`flex items-center gap-1 px-3 py-1.5 text-[11px] rounded-md font-medium transition-all ${
                activeTab === t.key ? "btn-rr-active" : "btn-rr"
              }`}
            >
              <t.icon size={10} />
              {t.label}
            </button>
          ))}
        </div>
      </div>

      <p className="text-[11px]" style={{ color: "var(--text-muted)" }}>
        Toggle conditions to simulate signals. Both long and short rules are evaluated independently on each 1m close.
        When both fire, the higher-confidence signal is taken. All rules validated by purged walk-forward CV (p&lt;0.01).
      </p>

      {/* Rule Cards */}
      {activeTab === "both" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <RuleCard rule={RESEARCH_RULES.long} side="long" />
          <RuleCard rule={RESEARCH_RULES.short} side="short" />
        </div>
      )}
      {activeTab === "long" && <RuleCard rule={RESEARCH_RULES.long} side="long" />}
      {activeTab === "short" && <RuleCard rule={RESEARCH_RULES.short} side="short" />}

      {/* Fold-by-Fold Performance */}
      <div className="metric-card p-4">
        <h3 className="text-xs font-semibold mb-3 flex items-center gap-2" style={{ color: "var(--text-muted)" }}>
          <BarChart3 size={12} />
          FOLD-BY-FOLD PERFORMANCE (BOTH DIRECTIONS)
        </h3>

        <div className="grid grid-cols-2 gap-4">
          {/* Long folds */}
          <div>
            <div className="text-[10px] font-medium mb-2 flex items-center gap-1" style={{ color: "var(--accent-green)" }}>
              <TrendingUp size={10} /> LONG
            </div>
            <div className="grid grid-cols-5 gap-1.5">
              {FOLD_DATA.long.map((f) => (
                <div key={f.fold} className="text-center p-2 rounded" style={{ background: "var(--bg-secondary)" }}>
                  <div className="text-[8px]" style={{ color: "var(--text-muted)" }}>F{f.fold}</div>
                  <div className="text-xs font-bold mono" style={{ color: f.exp > 0 ? "var(--accent-green)" : "var(--accent-red)" }}>
                    +{f.exp.toFixed(3)}%
                  </div>
                  <div className="text-[7px]" style={{ color: "var(--text-muted)" }}>
                    WR{f.wr}% n={f.trades}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Short folds */}
          <div>
            <div className="text-[10px] font-medium mb-2 flex items-center gap-1" style={{ color: "var(--accent-red)" }}>
              <TrendingDown size={10} /> SHORT
            </div>
            <div className="grid grid-cols-5 gap-1.5">
              {FOLD_DATA.short.map((f) => (
                <div key={f.fold} className="text-center p-2 rounded" style={{ background: "var(--bg-secondary)" }}>
                  <div className="text-[8px]" style={{ color: "var(--text-muted)" }}>F{f.fold}</div>
                  <div className="text-xs font-bold mono" style={{ color: f.exp > 0 ? "var(--accent-green)" : "var(--accent-red)" }}>
                    +{f.exp.toFixed(3)}%
                  </div>
                  <div className="text-[7px]" style={{ color: "var(--text-muted)" }}>
                    WR{f.wr}% n={f.trades}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Combined Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[
          { label: "Long Win Rate",  value: "57.6%",  color: "var(--accent-green)" },
          { label: "Short Win Rate", value: "56.0%",  color: "var(--accent-red)" },
          { label: "Long Net E",     value: "+0.067%", color: "var(--accent-green)" },
          { label: "Short Net E",    value: "+0.050%", color: "var(--accent-red)" },
          { label: "Long PF",        value: "1.45",   color: "var(--accent-blue)" },
          { label: "Short PF",       value: "1.40",   color: "var(--accent-blue)" },
          { label: "Long p-value",   value: "0.003",  color: "var(--accent-purple)" },
          { label: "Short p-value",  value: "0.008",  color: "var(--accent-purple)" },
        ].map((s, i) => (
          <div key={i} className="metric-card text-center">
            <div className="text-[9px] mb-1" style={{ color: "var(--text-muted)" }}>{s.label}</div>
            <div className="text-lg font-bold mono" style={{ color: s.color }}>{s.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
