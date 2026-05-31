import {
  Play, TrendingUp, TrendingDown, Target, Zap, BarChart3,
  ShieldCheck, AlertTriangle, Repeat, ArrowRight,
} from 'lucide-react';
import type { ResearchResult, RuleCell } from '../types';

interface Props {
  results?: ResearchResult | null;
}

const pct = (x: number | null | undefined, d = 1) =>
  x === null || x === undefined ? '—' : `${(x * 100).toFixed(d)}%`;
const signedPct = (x: number | null | undefined, d = 2) =>
  x === null || x === undefined ? '—' : `${x >= 0 ? '+' : ''}${(x * 100).toFixed(d)}%`;

function verdict(cell: RuleCell) {
  if (cell.is_robust) return { label: 'DEPLOYABLE', color: 'var(--accent-green)', icon: ShieldCheck };
  const hasEdge = (cell.edge_win_rate ?? 0) > 0 && cell.metrics.expectancy_net > 0;
  if (hasEdge) return { label: 'EDGE, NOT ROBUST', color: 'var(--accent-yellow)', icon: AlertTriangle };
  return { label: 'NO EDGE', color: 'var(--accent-red)', icon: AlertTriangle };
}

function CellCard({ cell }: { cell: RuleCell }) {
  const m = cell.metrics;
  if (!m || !m.trade_count) {
    return (
      <div className="metric-card p-3 opacity-60">
        <div className="text-xs font-semibold mb-1" style={{ color: 'var(--text-muted)' }}>
          {cell.side.toUpperCase()} · {cell.mode}
        </div>
        <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>No trades generated</div>
      </div>
    );
  }
  const v = verdict(cell);
  const isRev = cell.mode === 'reversion';
  const tradeLong = cell.trade_direction === 'long';
  const edgeWR = cell.edge_win_rate ?? 0;

  return (
    <div
      className="metric-card p-3 flex flex-col gap-2 transition-all"
      style={{
        border: `1px solid ${cell.is_robust ? 'var(--accent-green)' : 'var(--border)'}`,
        boxShadow: cell.is_robust ? '0 0 12px rgba(34,197,94,0.15)' : 'none',
      }}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          {isRev ? <Repeat size={13} style={{ color: 'var(--accent-cyan)' }} />
                 : (tradeLong ? <TrendingUp size={13} style={{ color: 'var(--accent-green)' }} />
                              : <TrendingDown size={13} style={{ color: 'var(--accent-red)' }} />)}
          <span className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>
            {cell.side.toUpperCase()} signal · {cell.mode}
          </span>
        </div>
        <span className="text-[9px] font-bold px-2 py-0.5 rounded flex items-center gap-1"
          style={{ background: 'var(--bg-secondary)', color: v.color }}>
          <v.icon size={9} /> {v.label}
        </span>
      </div>

      {/* what it actually trades */}
      <div className="flex items-center gap-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>
        {cell.side.toUpperCase()} conditions fire <ArrowRight size={9} />
        <span style={{ color: tradeLong ? 'var(--accent-green)' : 'var(--accent-red)', fontWeight: 600 }}>
          enter {cell.trade_direction.toUpperCase()}
        </span>
        {isRev && <span>(fade)</span>}
      </div>

      <div className="grid grid-cols-3 gap-1.5">
        {[
          { label: 'Win Rate', value: pct(m.win_rate), sub: `base ${pct(cell.baseline.win_rate)}`,
            color: m.win_rate > cell.baseline.win_rate ? 'var(--accent-green)' : 'var(--accent-red)' },
          { label: 'Edge WR', value: signedPct(edgeWR, 1), sub: 'vs random',
            color: edgeWR > 0 ? 'var(--accent-green)' : 'var(--accent-red)' },
          { label: 'Net E/trade', value: signedPct(m.expectancy_net, 3), sub: 'after fees',
            color: m.expectancy_net > 0 ? 'var(--accent-green)' : 'var(--accent-red)' },
          { label: 'PF', value: m.profit_factor === null ? '∞' : m.profit_factor.toFixed(2),
            sub: '', color: 'var(--accent-blue)' },
          { label: 'p-value', value: m.p_value.toFixed(3), sub: m.p_value < 0.05 ? 'sig' : 'not sig',
            color: m.p_value < 0.05 ? 'var(--accent-green)' : 'var(--accent-yellow)' },
          { label: 'Robust', value: `${cell.robustness_score.toFixed(0)}`, sub: '/100',
            color: 'var(--accent-purple)' },
        ].map((s) => (
          <div key={s.label} className="metric-card text-center p-1.5">
            <div className="text-[8px]" style={{ color: 'var(--text-muted)' }}>{s.label}</div>
            <div className="text-[11px] font-bold mono" style={{ color: s.color }}>{s.value}</div>
            <div className="text-[7px]" style={{ color: 'var(--text-muted)' }}>{s.sub}</div>
          </div>
        ))}
      </div>

      <div className="flex gap-3 text-[9px]" style={{ color: 'var(--text-muted)' }}>
        <span><Target size={9} className="inline mr-0.5" /> n={m.trade_count}</span>
        <span>stability {pct(cell.walk_forward.stability, 0)}</span>
        <span>hold {m.avg_bars_held}b</span>
      </div>

      {cell.warnings.length > 0 && (
        <div className="text-[9px] flex flex-wrap gap-1">
          {cell.warnings.map((w, i) => (
            <span key={i} className="px-1.5 py-0.5 rounded"
              style={{ background: 'rgba(234,179,8,0.1)', color: 'var(--accent-yellow)' }}>{w}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function FoldStrip({ cell }: { cell: RuleCell }) {
  const folds = cell.walk_forward.folds || [];
  if (!folds.length) return null;
  return (
    <div className="metric-card p-3">
      <h3 className="text-[11px] font-semibold mb-2 flex items-center gap-1.5" style={{ color: 'var(--text-muted)' }}>
        <BarChart3 size={11} /> OUT-OF-TIME FOLDS — best: {cell.side.toUpperCase()} {cell.mode}
      </h3>
      <div className="grid grid-cols-5 gap-1.5">
        {folds.map((f) => (
          <div key={f.fold} className="text-center p-2 rounded" style={{ background: 'var(--bg-secondary)' }}>
            <div className="text-[8px]" style={{ color: 'var(--text-muted)' }}>F{f.fold}</div>
            <div className="text-[11px] font-bold mono"
              style={{ color: f.expectancy === null ? 'var(--text-muted)'
                : f.expectancy > 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
              {f.expectancy === null ? '—' : signedPct(f.expectancy, 2)}
            </div>
            <div className="text-[7px]" style={{ color: 'var(--text-muted)' }}>
              {f.win_rate === null ? `n=${f.trades}` : `WR${(f.win_rate * 100).toFixed(0)} n=${f.trades}`}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function StrategyPlayer({ results }: Props) {
  const player = results?.player;

  if (!player || player.error || (!player.long && !player.short)) {
    return (
      <div className="metric-card p-6 flex flex-col items-center gap-2 text-center">
        <Play size={20} style={{ color: 'var(--accent-blue)' }} />
        <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          5-Condition Rule Strategy
        </h2>
        <p className="text-[12px]" style={{ color: 'var(--text-muted)' }}>
          {player?.error
            ? `Rule backtest unavailable: ${player.error}`
            : 'Run research to backtest the 5-condition rule and see its real, validated edge.'}
        </p>
      </div>
    );
  }

  const cells: RuleCell[] = [
    player.long?.momentum, player.long?.reversion,
    player.short?.momentum, player.short?.reversion,
  ].filter(Boolean) as RuleCell[];
  const best = player.best || null;

  return (
    <div className="metric-card p-4 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Play size={14} style={{ color: 'var(--accent-blue)' }} />
          <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            5-Condition Rule — momentum vs mean-reversion
          </h2>
        </div>
        <span className="text-[10px] mono" style={{ color: 'var(--text-muted)' }}>
          {player.rr} · {player.leverage}x · fee {((player.cost ?? 0) * 100).toFixed(2)}%
        </span>
      </div>

      <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
        Real backtest, leakage-safe, out-of-time folds + bootstrap significance vs a random-entry
        baseline. <b>Momentum</b> trades the signal direction; <b>reversion</b> fades it. A card is
        marked DEPLOYABLE only if it beats random, is significant (p&lt;0.05), and stays positive
        across folds.
      </p>

      {/* headline verdict */}
      {best && (
        <div className="flex items-center gap-2 p-2.5 rounded-lg text-xs"
          style={{
            background: player.has_edge ? 'rgba(34,197,94,0.1)' : 'rgba(234,179,8,0.08)',
            color: player.has_edge ? 'var(--accent-green)' : 'var(--accent-yellow)',
          }}>
          {player.has_edge ? <ShieldCheck size={15} /> : <AlertTriangle size={15} />}
          <span className="font-semibold">
            {player.has_edge
              ? `Deployable edge: ${best.side.toUpperCase()} ${best.mode} → enter ${best.trade_direction.toUpperCase()}, `
              : `Best candidate (not yet robust): ${best.side.toUpperCase()} ${best.mode} → enter ${best.trade_direction.toUpperCase()}, `}
            WR {pct(best.metrics.win_rate)} ({signedPct(best.edge_win_rate, 1)} vs random),
            net {signedPct(best.metrics.expectancy_net, 3)}/trade, p={best.metrics.p_value.toFixed(3)}
          </span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {cells.map((c, i) => <CellCard key={i} cell={c} />)}
      </div>

      {best && best.metrics.trade_count > 0 && <FoldStrip cell={best} />}

      <div className="flex items-center gap-1.5 text-[10px]" style={{ color: 'var(--text-muted)' }}>
        <Zap size={10} />
        Conditions: {(player.conditions || []).map((c) => c.label).join(' · ')}
      </div>
    </div>
  );
}
