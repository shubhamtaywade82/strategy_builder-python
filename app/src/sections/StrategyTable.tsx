import { TrendingUp, ChevronRight, CheckCircle } from 'lucide-react';

interface Props {
  results: any;
}

export default function StrategyTable({ results }: Props) {
  const strategies = (results.strategies || []).sort(
    (a: any, b: any) => b.metrics.expectancy - a.metrics.expectancy
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold flex items-center gap-2" style={{ color: 'var(--text-primary)' }}>
          <TrendingUp size={14} style={{ color: 'var(--accent-blue)' }} />
          Strategy Comparison
        </h2>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{strategies.length} strategies found</span>
      </div>

      <div className="overflow-x-auto">
        <table className="data-table">
          <thead>
            <tr>
              <th>Status</th>
              <th>Strategy</th>
              <th>Side</th>
              <th>Trades</th>
              <th>Win Rate</th>
              <th>P.Factor</th>
              <th>Expectancy</th>
              <th>Net PnL</th>
              <th>Avg Win</th>
              <th>Avg Loss</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {strategies.map((s: any) => (
              <tr key={s.id}>
                <td>
                  {s.isViable ? (
                    <span className="status-pass text-[10px] px-1.5 py-0.5 rounded">PASS</span>
                  ) : (
                    <span className="status-fail text-[10px] px-1.5 py-0.5 rounded">FAIL</span>
                  )}
                </td>
                <td>
                  <div className="font-medium text-xs" style={{ color: 'var(--text-primary)' }}>{s.name}</div>
                  <div className="text-[10px] truncate max-w-[200px]" style={{ color: 'var(--text-muted)' }}>{s.description}</div>
                </td>
                <td>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${s.side === 'long' ? 'status-pass' : 'status-fail'}`}>
                    {s.side.toUpperCase()}
                  </span>
                </td>
                <td className="mono text-xs">{s.metrics.tradeCount}</td>
                <td className="mono text-xs font-semibold" style={{ color: s.metrics.winRate > 0.5 ? 'var(--accent-green)' : s.metrics.winRate > 0.35 ? 'var(--accent-yellow)' : 'var(--accent-red)' }}>
                  {(s.metrics.winRate * 100).toFixed(1)}%
                </td>
                <td className="mono text-xs" style={{ color: s.metrics.profitFactor > 1.2 ? 'var(--accent-green)' : s.metrics.profitFactor > 1.0 ? 'var(--accent-yellow)' : 'var(--accent-red)' }}>
                  {s.metrics.profitFactor.toFixed(2)}
                </td>
                <td className="mono text-xs font-semibold" style={{ color: s.metrics.expectancy > 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                  {(s.metrics.expectancy * 100).toFixed(2)}%
                </td>
                <td className="mono text-xs" style={{ color: s.metrics.netPnl > 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                  {s.metrics.netPnl > 0 ? '+' : ''}{(s.metrics.netPnl * 100).toFixed(1)}%
                </td>
                <td className="mono text-xs" style={{ color: 'var(--accent-green)' }}>+{(s.metrics.avgWin * 100).toFixed(2)}%</td>
                <td className="mono text-xs" style={{ color: 'var(--accent-red)' }}>{(s.metrics.avgLoss * 100).toFixed(2)}%</td>
                <td>
                  <button className="p-1 rounded hover:opacity-70" style={{ color: 'var(--accent-blue)' }}>
                    <ChevronRight size={14} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Strategy Detail Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {strategies.filter((s: any) => s.isViable).slice(0, 4).map((s: any) => (
          <div key={s.id} className="metric-card" style={{ animationDelay: '0.1s' }}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <CheckCircle size={12} style={{ color: 'var(--accent-green)' }} />
                <span className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>{s.name}</span>
              </div>
              <span className={`text-[10px] px-1.5 py-0.5 rounded ${s.side === 'long' ? 'status-pass' : 'status-fail'}`}>
                {s.side.toUpperCase()}
              </span>
            </div>
            <p className="text-[11px] mb-3 leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
              {s.description}
            </p>
            <div className="space-y-1.5">
              {s.conditions.slice(0, 4).map((c: any, i: number) => (
                <div key={i} className="flex items-center justify-between text-[11px] px-2 py-1 rounded" style={{ background: 'var(--bg-secondary)' }}>
                  <div className="flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 rounded-full" style={{ background: `rgba(59,130,246,${0.3 + c.importance * 3})` }} />
                    <span style={{ color: 'var(--text-secondary)' }}>{c.feature}</span>
                    <span style={{ color: 'var(--text-muted)' }}>{c.operator}</span>
                    <span className="mono" style={{ color: 'var(--accent-cyan)' }}>{c.threshold.toFixed(4)}</span>
                  </div>
                  <div className="w-12 h-1 rounded-full overflow-hidden" style={{ background: 'var(--bg-primary)' }}>
                    <div className="h-full rounded-full" style={{ width: `${c.importance * 500}%`, background: 'var(--accent-blue)' }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
