import { AlertTriangle, CheckCircle, XCircle, BarChart3 } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, BarChart, Bar, ReferenceLine } from 'recharts';

interface Props {
  results: any;
}

export default function ValidationPanel({ results }: Props) {
  // Collect all walk-forward and shuffle test data across RRs
  const allWf = Object.entries(results.results || {})
    .filter(([_, d]: [string, any]) => d.walk_forward?.foldResults)
    .map(([rr, d]: [string, any]) => ({
      rr,
      ...d.walk_forward,
      shuffle: d.shuffle_test,
    }));

  const foldChartData = allWf.flatMap((w: any) =>
    (w.foldResults || []).map((f: any) => ({
      name: `${w.rr}-F${f.fold}`,
      fold: f.fold,
      rr: w.rr,
      trainAuc: f.train_auc,
      testAuc: f.test_auc,
      precision: f.test_precision,
      recall: f.test_recall,
    }))
  );

  const wfSummary = allWf.map((w: any) => ({
    rr: w.rr,
    avgAuc: w.avg_test_auc,
    minAuc: w.min_test_auc,
    stability: w.stability,
    degradation: w.degradation,
    isValid: w.isValid,
    shuffledAuc: w.shuffle?.shuffled_auc_mean,
    realAuc: w.shuffle?.real_auc,
    pValue: w.shuffle?.p_value,
    isSig: w.shuffle?.is_significant,
  }));

  return (
    <div className="space-y-4">
      {/* Validation Status */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        {allWf.map((w: any) => (
          <div key={w.rr} className={`metric-card ${w.isValid ? 'glow-green' : 'glow-red'}`}>
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>RR {w.rr}</span>
              {w.isValid ? (
                <span className="status-pass text-[10px] px-2 py-0.5 rounded-full flex items-center gap-1">
                  <CheckCircle size={10} /> VALID
                </span>
              ) : (
                <span className="status-fail text-[10px] px-2 py-0.5 rounded-full flex items-center gap-1">
                  <XCircle size={10} /> INVALID
                </span>
              )}
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span style={{ color: 'var(--text-muted)' }}>Avg Test AUC</span>
                <span className="mono font-semibold" style={{ color: w.avg_test_auc > 0.52 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                  {w.avg_test_auc?.toFixed(3)}
                </span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: 'var(--text-muted)' }}>Min Test AUC</span>
                <span className="mono">{w.min_test_auc?.toFixed(3)}</span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: 'var(--text-muted)' }}>Stability</span>
                <span className="mono font-semibold" style={{ color: w.stability > 0.6 ? 'var(--accent-green)' : 'var(--accent-yellow)' }}>
                  {(w.stability * 100)?.toFixed(0)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: 'var(--text-muted)' }}>Degradation</span>
                <span className="mono" style={{ color: w.degradation > -0.1 ? 'var(--accent-green)' : 'var(--accent-yellow)' }}>
                  {w.degradation?.toFixed(3)}
                </span>
              </div>
              {w.shuffle && (
                <>
                  <div className="border-t pt-2 mt-2" style={{ borderColor: 'var(--border)' }}>
                    <div className="flex justify-between">
                      <span style={{ color: 'var(--text-muted)' }}>Real AUC</span>
                      <span className="mono" style={{ color: 'var(--accent-blue)' }}>{w.shuffle.real_auc?.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span style={{ color: 'var(--text-muted)' }}>Shuffled AUC</span>
                      <span className="mono">{w.shuffle.shuffled_auc_mean?.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span style={{ color: 'var(--text-muted)' }}>P-Value</span>
                      <span className="mono font-semibold" style={{ color: w.shuffle.is_significant ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                        {w.shuffle.p_value?.toFixed(3)} {w.shuffle.is_significant ? '(sig)' : '(ns)'}
                      </span>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Fold-by-fold Chart */}
      {foldChartData.length > 0 && (
        <div className="chart-container">
          <h3 className="text-xs font-semibold mb-4 flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
            <BarChart3 size={12} />
            FOLD-BY-FOLD AUC COMPARISON
          </h3>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={foldChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3142" />
              <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 10 }} angle={-45} textAnchor="end" height={60} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} domain={[0.4, 0.8]} />
              <Tooltip
                contentStyle={{ background: '#1a1f2e', border: '1px solid #2a3142', borderRadius: 8, fontSize: 12 }}
              />
              <ReferenceLine y={0.5} stroke="#ef4444" strokeDasharray="5 5" label={{ value: 'Random', fill: '#ef4444', fontSize: 10 }} />
              <ReferenceLine y={0.52} stroke="#eab308" strokeDasharray="5 5" label={{ value: 'Min Edge', fill: '#eab308', fontSize: 10 }} />
              <Line type="monotone" dataKey="trainAuc" stroke="#3b82f6" strokeWidth={1.5} dot={{ r: 3 }} name="Train AUC" />
              <Line type="monotone" dataKey="testAuc" stroke="#22c55e" strokeWidth={2} dot={{ r: 4 }} name="Test AUC" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Shuffle Test */}
      {wfSummary.length > 0 && (
        <div className="chart-container">
          <h3 className="text-xs font-semibold mb-4 flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
            <AlertTriangle size={12} />
            SHUFFLE TEST (LEAKAGE DETECTION)
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={wfSummary}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3142" />
              <XAxis dataKey="rr" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} domain={[0.4, 0.7]} />
              <Tooltip contentStyle={{ background: '#1a1f2e', border: '1px solid #2a3142', borderRadius: 8, fontSize: 12 }} />
              <ReferenceLine y={0.5} stroke="#ef4444" strokeDasharray="5 5" />
              <Bar dataKey="realAuc" fill="#22c55e" radius={[4, 4, 0, 0]} name="Real AUC" />
              <Bar dataKey="shuffledAuc" fill="#64748b" radius={[4, 4, 0, 0]} name="Shuffled AUC" />
            </BarChart>
          </ResponsiveContainer>
          <div className="flex gap-4 mt-3 text-[11px]" style={{ color: 'var(--text-muted)' }}>
            <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-sm" style={{ background: '#22c55e' }} /> Real AUC</span>
            <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-sm" style={{ background: '#64748b' }} /> Shuffled AUC</span>
            <span className="flex items-center gap-1"><div className="w-4 h-0 border-t border-dashed" style={{ borderColor: '#ef4444' }} /> Random (0.5)</span>
          </div>
        </div>
      )}
    </div>
  );
}
