import { useState } from 'react';
import {
  Play, TrendingUp, TrendingDown,
  Target, Activity, Zap, BarChart3,
  ArrowUp, ArrowDown, AlertTriangle, ShieldCheck, FlaskConical,
} from 'lucide-react';
import type { PlayerSide, ResearchResult } from '@/types';

interface Props {
  results?: ResearchResult | null;
}

const pct = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined ? '—' : `${(v * 100).toFixed(d)}%`;
const num = (v: number | null | undefined, d = 2) =>
  v === null || v === undefined ? '—' : v.toFixed(d);

/** One side's real, backtested rule card — all numbers come from `side`. */
function RuleCard({ side, dir }: { side: PlayerSide; dir: 'long' | 'short' }) {
  const conditions = side.conditions ?? [];
  const [checks, setChecks] = useState<Record<string, boolean>>(
    Object.fromEntries(conditions.map((c) => [c.id, true]))
  );
  const isLong = dir === 'long';
  const m = side.metrics;

  // No real backtest for this side (too few signals) -> honest empty state.
  if (!m) {
    return (
      <div className="metric-card p-4 flex flex-col gap-2" style={{ border: '1px solid var(--border)' }}>
        <div className="flex items-center gap-2">
          {isLong ? <TrendingUp size={14} style={{ color: 'var(--accent-green)' }} />
                  : <TrendingDown size={14} style={{ color: 'var(--accent-red)' }} />}
          <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            {dir.toUpperCase()} rule
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
          <AlertTriangle size={12} style={{ color: 'var(--accent-yellow)' }} />
          {side.error || 'Not enough signals to backtest this rule on the selected window.'}
        </div>
      </div>
    );
  }

  const allMet = conditions.length > 0 && Object.values(checks).every(Boolean);
  const metCount = Object.values(checks).filter(Boolean).length;
  const robust = side.isRobust;

  return (
    <div
      className="metric-card p-4 flex flex-col gap-3 transition-all"
      style={{
        border: `1px solid ${allMet ? (isLong ? '#22c55e' : '#ef4444') : 'var(--border)'}`,
        boxShadow: allMet ? `0 0 12px ${isLong ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)'}` : 'none',
      }}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isLong ? <TrendingUp size={14} style={{ color: 'var(--accent-green)' }} />
                  : <TrendingDown size={14} style={{ color: 'var(--accent-red)' }} />}
          <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            {side.name || `${dir.toUpperCase()} rule`}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className="text-[9px] font-bold px-2 py-0.5 rounded"
            style={{ background: isLong ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)',
                     color: isLong ? 'var(--accent-green)' : 'var(--accent-red)' }}
          >
            {dir.toUpperCase()}
          </span>
          <span
            className="text-[9px] font-bold px-2 py-0.5 rounded flex items-center gap-1"
            style={{ background: robust ? 'rgba(34,197,94,0.12)' : 'rgba(234,179,8,0.12)',
                     color: robust ? 'var(--accent-green)' : 'var(--accent-yellow)' }}
            title={robust ? 'Passes out-of-sample robustness gate' : (side.warnings || []).join('; ')}
          >
            {robust ? <ShieldCheck size={9} /> : <AlertTriangle size={9} />}
            {robust ? 'ROBUST' : 'NOT ROBUST'}
          </span>
          <span className="text-[10px] mono" style={{ color: 'var(--text-muted)' }}>
            {metCount}/{conditions.length}
          </span>
        </div>
      </div>

      {/* Conditions (toggle to simulate which fired) */}
      <div className="flex flex-col gap-1.5">
        {conditions.map((cond) => (
          <button
            key={cond.id}
            onClick={() => setChecks((p) => ({ ...p, [cond.id]: !p[cond.id] }))}
            className="flex items-center gap-2 w-full text-left p-2 rounded-lg transition-all"
            style={{
              background: checks[cond.id]
                ? (isLong ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)')
                : 'var(--bg-secondary)',
              border: `1px solid ${checks[cond.id]
                ? (isLong ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)') : 'transparent'}`,
            }}
          >
            <div className="w-3 h-3 rounded-full flex-shrink-0 transition-colors"
              style={{ background: checks[cond.id] ? (isLong ? 'var(--accent-green)' : 'var(--accent-red)') : 'var(--bg-primary)' }} />
            <span className="text-xs" style={{ color: checks[cond.id] ? 'var(--text-primary)' : 'var(--text-muted)' }}>
              {cond.label}
            </span>
          </button>
        ))}
      </div>

      {allMet && (
        <div className="flex items-center gap-2 p-2 rounded-lg text-xs font-semibold"
          style={{ background: isLong ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)',
                   color: isLong ? 'var(--accent-green)' : 'var(--accent-red)' }}>
          {isLong ? <ArrowUp size={14} /> : <ArrowDown size={14} />}
          {dir.toUpperCase()} ENTRY — all {conditions.length} conditions met
        </div>
      )}

      {/* Exits */}
      {side.exits && (
        <div className="flex gap-3 text-[10px]" style={{ color: 'var(--text-muted)' }}>
          <span><Target size={10} className="inline mr-0.5" /> Target: {pct(side.exits.targetPct)}</span>
          <span>Stop: {pct(side.exits.stopPct)}</span>
          <span>Hold: {side.exits.timeStopBars} bars</span>
        </div>
      )}

      {/* Real metrics */}
      <div className="grid grid-cols-6 gap-2">
        {[
          { label: 'Trades', value: `${m.tradeCount}`, color: 'var(--text-primary)' },
          { label: 'Win Rate', value: pct(m.winRate), color: m.winRate > 0.5 ? 'var(--accent-green)' : 'var(--accent-yellow)' },
          { label: 'PF', value: num(m.profitFactor), color: 'var(--accent-blue)' },
          { label: 'Net E', value: pct(m.expectancy, 3), color: m.expectancy > 0 ? 'var(--accent-green)' : 'var(--accent-red)' },
          { label: 'Sharpe', value: num(m.sharpe), color: 'var(--accent-cyan)' },
          { label: 'p-value', value: num(m.pValue, 3), color: m.pValue < 0.05 ? 'var(--accent-purple)' : 'var(--accent-yellow)' },
        ].map((s) => (
          <div key={s.label} className="metric-card text-center p-1.5">
            <div className="text-[8px] mb-0.5" style={{ color: 'var(--text-muted)' }}>{s.label}</div>
            <div className="text-[10px] font-bold mono" style={{ color: s.color }}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* Edge vs random baseline */}
      <div className="text-[10px] flex items-center gap-1.5" style={{ color: 'var(--text-muted)' }}>
        <Activity size={10} />
        Edge vs random entry: WR {(side.edgeWinRate ?? 0) >= 0 ? '+' : ''}{pct(side.edgeWinRate)},
        {' '}E {(side.edgeExpectancy ?? 0) >= 0 ? '+' : ''}{pct(side.edgeExpectancy, 3)}
      </div>

      {/* Honest warnings */}
      {!robust && (side.warnings || []).length > 0 && (
        <div className="text-[10px] flex flex-col gap-0.5" style={{ color: 'var(--accent-yellow)' }}>
          {(side.warnings || []).map((w, i) => (
            <span key={i} className="flex items-center gap-1"><AlertTriangle size={9} /> {w}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function FoldStrip({ side }: { side?: PlayerSide }) {
  const folds = side?.folds ?? [];
  if (!side?.metrics || folds.length === 0)
    return <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>No fold data</div>;
  return (
    <div className="grid grid-cols-5 gap-1.5">
      {folds.map((f) => (
        <div key={f.fold} className="text-center p-2 rounded" style={{ background: 'var(--bg-secondary)' }}>
          <div className="text-[8px]" style={{ color: 'var(--text-muted)' }}>F{f.fold}</div>
          <div className="text-xs font-bold mono"
            style={{ color: f.expectancy === null ? 'var(--text-muted)' : (f.expectancy > 0 ? 'var(--accent-green)' : 'var(--accent-red)') }}>
            {f.expectancy === null ? '—' : `${f.expectancy >= 0 ? '+' : ''}${(f.expectancy * 100).toFixed(3)}%`}
          </div>
          <div className="text-[7px]" style={{ color: 'var(--text-muted)' }}>
            WR{f.winRate === null ? '—' : (f.winRate * 100).toFixed(0)}% n={f.trades}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function StrategyPlayer({ results }: Props) {
  const [activeTab, setActiveTab] = useState<'both' | 'long' | 'short'>('both');
  const player = results?.player;

  // Honest empty state — no fabricated numbers when there's no run.
  if (!player || player.error || (!player.long && !player.short)) {
    return (
      <div className="metric-card p-8 flex flex-col items-center gap-3 text-center">
        <Play size={28} style={{ color: 'var(--accent-blue)' }} />
        <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          Strategy Player
        </h2>
        <p className="text-xs max-w-md" style={{ color: 'var(--text-muted)' }}>
          {player?.error
            ? `Rule strategy unavailable: ${player.error}`
            : 'Run research to backtest the 5-condition rule and populate live-validated long/short metrics here. No placeholder numbers are shown.'}
        </p>
      </div>
    );
  }

  const long = player.long;
  const short = player.short;
  const synthetic = results?.dataSource === 'synthetic';

  return (
    <div className="metric-card p-4 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Play size={14} style={{ color: 'var(--accent-blue)' }} />
          <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            Both-Direction Strategy Player
          </h2>
          <span className="text-[10px] mono px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-secondary)', color: 'var(--text-muted)' }}>
            {results?.symbol} · {player.rr}
          </span>
        </div>
        <div className="flex gap-1">
          {[
            { key: 'both' as const, label: 'Both', icon: Zap },
            { key: 'long' as const, label: 'Long Only', icon: TrendingUp },
            { key: 'short' as const, label: 'Short Only', icon: TrendingDown },
          ].map((t) => (
            <button key={t.key} onClick={() => setActiveTab(t.key)}
              className={`flex items-center gap-1 px-3 py-1.5 text-[11px] rounded-md font-medium transition-all ${
                activeTab === t.key ? 'btn-rr-active' : 'btn-rr'}`}>
              <t.icon size={10} />{t.label}
            </button>
          ))}
        </div>
      </div>

      {synthetic && (
        <div className="flex items-center gap-2 p-2 rounded-lg text-[11px] font-medium"
          style={{ background: 'rgba(234,179,8,0.1)', color: 'var(--accent-yellow)' }}>
          <FlaskConical size={12} />
          Synthetic demo data — live Binance klines were unavailable, so these are
          deterministic generated bars, not real market history.
        </div>
      )}

      <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
        Both rules are backtested out-of-sample with time folds, a bootstrap p-value, and a
        random-entry baseline. A rule is only marked ROBUST when it is profitable, consistent
        across folds (stability ≥ 0.6), significant (p &lt; 0.05), and beats the baseline.
      </p>

      {activeTab === 'both' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {long && <RuleCard side={long} dir="long" />}
          {short && <RuleCard side={short} dir="short" />}
        </div>
      )}
      {activeTab === 'long' && long && <RuleCard side={long} dir="long" />}
      {activeTab === 'short' && short && <RuleCard side={short} dir="short" />}

      {/* Fold-by-fold (real) */}
      <div className="metric-card p-4">
        <h3 className="text-xs font-semibold mb-3 flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
          <BarChart3 size={12} /> FOLD-BY-FOLD OUT-OF-SAMPLE EXPECTANCY
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-[10px] font-medium mb-2 flex items-center gap-1" style={{ color: 'var(--accent-green)' }}>
              <TrendingUp size={10} /> LONG
            </div>
            <FoldStrip side={long} />
          </div>
          <div>
            <div className="text-[10px] font-medium mb-2 flex items-center gap-1" style={{ color: 'var(--accent-red)' }}>
              <TrendingDown size={10} /> SHORT
            </div>
            <FoldStrip side={short} />
          </div>
        </div>
      </div>
    </div>
  );
}
