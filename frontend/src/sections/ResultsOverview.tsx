import { TrendingUp, Target, Award, CheckCircle } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell } from 'recharts';

interface Props {
  results: any;
}

export default function ResultsOverview({ results }: Props) {
  const allStrats = results.strategies || [];
  const viable = allStrats.filter((s: any) => s.isViable);
  const best = viable.length > 0 ? viable.reduce((a: any, b: any) => a.metrics.expectancy > b.metrics.expectancy ? a : b) : null;

  const rrData = Object.entries(results.results || {}).map(([rr, data]: [string, any]) => {
    const strats = data.strategies || [];
    const v = strats.filter((s: any) => s.isViable);
    const top = v.length > 0 ? v.reduce((a: any, b: any) => a.metrics.winRate > b.metrics.winRate ? a : b) : null;
    return {
      rr,
      label: `${rr} ${top ? `(${top.side})` : ''}`,
      winRate: top ? top.metrics.winRate * 100 : 0,
      pf: top ? top.metrics.profitFactor : 0,
      expectancy: top ? top.metrics.expectancy * 100 : 0,
      trades: top ? top.metrics.tradeCount : 0,
      viable: v.length,
    };
  });

  const statCards = [
    { label: 'Total Strategies', value: allStrats.length, icon: Target, color: 'var(--accent-blue)', suffix: '' },
    { label: 'Viable Strategies', value: viable.length, icon: CheckCircle, color: 'var(--accent-green)', suffix: '' },
    { label: 'Best Win Rate', value: best ? best.metrics.winRate * 100 : 0, icon: Award, color: 'var(--accent-yellow)', suffix: '%' },
    { label: 'Best Profit Factor', value: best ? best.metrics.profitFactor : 0, icon: TrendingUp, color: 'var(--accent-cyan)', suffix: '' },
  ];

  return (
    <div className="space-y-4">
      {/* Stat Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {statCards.map((card, i) => (
          <div key={i} className="metric-card" style={{ animationDelay: `${i * 0.05}s` }}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>{card.label}</span>
              <card.icon size={14} style={{ color: card.color }} />
            </div>
            <div className="text-2xl font-bold mono" style={{ color: card.color }}>
              {typeof card.value === 'number' ? card.value.toFixed(card.suffix === '%' ? 1 : 2) : card.value}{card.suffix}
            </div>
          </div>
        ))}
      </div>

      {/* Best Strategy */}
      {best && (
        <div className="metric-card gradient-border glow-green">
          <div className="flex items-center gap-2 mb-3">
            <Award size={16} style={{ color: 'var(--accent-green)' }} />
            <span className="text-sm font-semibold" style={{ color: 'var(--accent-green)' }}>Best Strategy</span>
            <span className="status-pass text-[10px] px-2 py-0.5 rounded-full ml-2">TRADEABLE</span>
          </div>
          <div className="font-mono text-sm mb-2" style={{ color: 'var(--text-primary)' }}>
            {best.name}
          </div>
          <div className="text-xs mb-3" style={{ color: 'var(--text-secondary)' }}>
            {best.description}
          </div>
          <div className="flex gap-4 flex-wrap text-xs">
            <span><span style={{ color: 'var(--text-muted)' }}>Win Rate:</span> <strong style={{ color: 'var(--accent-green)' }}>{(best.metrics.winRate * 100).toFixed(1)}%</strong></span>
            <span><span style={{ color: 'var(--text-muted)' }}>Profit Factor:</span> <strong>{best.metrics.profitFactor.toFixed(2)}</strong></span>
            <span><span style={{ color: 'var(--text-muted)' }}>Expectancy:</span> <strong>{(best.metrics.expectancy * 100).toFixed(2)}%</strong></span>
            <span><span style={{ color: 'var(--text-muted)' }}>Trades:</span> <strong>{best.metrics.tradeCount}</strong></span>
            <span><span style={{ color: 'var(--text-muted)' }}>Max DD:</span> <strong style={{ color: 'var(--accent-red)' }}>{(best.metrics.maxDrawdown * 100).toFixed(1)}%</strong></span>
          </div>
        </div>
      )}

      {/* RR Comparison Chart */}
      {rrData.length > 0 && (
        <div className="chart-container">
          <h3 className="text-xs font-semibold mb-4" style={{ color: 'var(--text-muted)' }}>WIN RATE BY RISK:REWARD</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={rrData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3142" />
              <XAxis dataKey="rr" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} domain={[0, 100]} />
              <Tooltip
                contentStyle={{ background: '#1a1f2e', border: '1px solid #2a3142', borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: '#94a3b8' }}
              />
              <Bar dataKey="winRate" radius={[4, 4, 0, 0]}>
                {rrData.map((entry, index) => (
                  <Cell key={index} fill={entry.winRate > 50 ? '#22c55e' : entry.winRate > 40 ? '#eab308' : '#ef4444'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
