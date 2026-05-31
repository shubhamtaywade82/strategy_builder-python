import { PieChart, Brain, TrendingUp, TrendingDown, BarChart3 } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

interface Props {
  results: any;
}

export default function FeaturePanel({ results }: Props) {
  const features = results.topFeatures || [];

  const impData = features.map((f: any) => ({
    name: f.feature.replace(/_/g, ' '),
    importance: f.importance * 100,
    winRateAbove: f.winRateAbove * 100,
    winRateBelow: f.winRateBelow * 100,
    shapValue: f.shapValue,
    direction: f.direction,
  }));

  const radarData = features.map((f: any) => ({
    feature: f.feature.length > 10 ? f.feature.slice(0, 10) + '..' : f.feature,
    importance: f.importance * 100,
    winRateDiff: (f.winRateAbove - f.winRateBelow) * 100,
    shapAbs: Math.abs(f.shapValue) * 100,
  }));

  return (
    <div className="space-y-4">
      {/* Feature Importance Chart */}
      <div className="chart-container">
        <h3 className="text-xs font-semibold mb-4 flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
          <BarChart3 size={12} />
          FEATURE IMPORTANCE (XGBoost)
        </h3>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={[...impData].reverse()} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#2a3142" />
            <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis dataKey="name" type="category" tick={{ fill: '#94a3b8', fontSize: 10 }} width={100} />
            <Tooltip contentStyle={{ background: '#1a1f2e', border: '1px solid #2a3142', borderRadius: 8, fontSize: 12 }} />
            <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Radar Chart */}
      <div className="chart-container">
        <h3 className="text-xs font-semibold mb-4 flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
          <PieChart size={12} />
          FEATURE PROFILE
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart data={radarData}>
            <PolarGrid stroke="#2a3142" />
            <PolarAngleAxis dataKey="feature" tick={{ fill: '#94a3b8', fontSize: 10 }} />
            <PolarRadiusAxis tick={{ fill: '#64748b', fontSize: 10 }} />
            <Radar name="Importance" dataKey="importance" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.2} strokeWidth={2} />
            <Radar name="Win Rate Impact" dataKey="winRateDiff" stroke="#22c55e" fill="#22c55e" fillOpacity={0.1} strokeWidth={1.5} />
            <Tooltip contentStyle={{ background: '#1a1f2e', border: '1px solid #2a3142', borderRadius: 8, fontSize: 12 }} />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Feature Insights Table */}
      <div className="metric-card">
        <h3 className="text-xs font-semibold mb-3 flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
          <Brain size={12} />
          FEATURE INSIGHTS
        </h3>
        <div className="space-y-2">
          {features.map((f: any, i: number) => (
            <div key={i} className="flex items-center gap-3 p-2 rounded-lg" style={{ background: 'var(--bg-secondary)' }}>
              <div className="w-6 h-6 rounded flex items-center justify-center text-[10px] font-bold" style={{
                background: f.direction === 'high'
                  ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)',
                color: f.direction === 'high' ? 'var(--accent-green)' : 'var(--accent-red)'
              }}>
                {f.direction === 'high' ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>{f.feature}</span>
                  <span className="text-[10px] px-1 py-0.5 rounded" style={{
                    background: f.direction === 'high' ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)',
                    color: f.direction === 'high' ? 'var(--accent-green)' : 'var(--accent-red)'
                  }}>
                    {f.direction.toUpperCase()}
                  </span>
                </div>
                <div className="flex items-center gap-3 mt-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                  <span>Threshold: <span className="mono" style={{ color: 'var(--accent-cyan)' }}>{f.threshold.toFixed(4)}</span></span>
                  <span>WR Above: <span className="mono" style={{ color: 'var(--accent-green)' }}>{(f.winRateAbove * 100).toFixed(1)}%</span></span>
                  <span>WR Below: <span className="mono" style={{ color: 'var(--accent-red)' }}>{(f.winRateBelow * 100).toFixed(1)}%</span></span>
                  <span>SHAP: <span className="mono" style={{ color: f.shapValue > 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>{f.shapValue > 0 ? '+' : ''}{f.shapValue.toFixed(3)}</span></span>
                </div>
              </div>
              <div className="w-16">
                <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-primary)' }}>
                  <div className="h-full rounded-full transition-all" style={{
                    width: `${Math.min(f.importance * 500, 100)}%`,
                    background: `linear-gradient(90deg, var(--accent-blue), var(--accent-purple))`
                  }} />
                </div>
                <div className="text-[9px] text-right mt-0.5 mono" style={{ color: 'var(--text-muted)' }}>
                  {(f.importance * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
