import { Target, Layers, Brain, Shield, Code } from 'lucide-react';

const steps = [
  {
    icon: Target,
    title: '1. Triple-Barrier Labeling',
    color: 'var(--accent-red)',
    desc: 'Every 1m bar is labeled: "Did price hit +target% BEFORE -stop% within horizon bars?" Labels both long and short independently. If both barriers fall in same bar, stop hits first (worst case).',
    code: `label = 1 if high[j] >= entry * 1.01 before low[j] <= entry * 0.995 else 0`,
  },
  {
    icon: Layers,
    title: '2. Leakage-Safe MTF Features',
    color: 'var(--accent-blue)',
    desc: 'Features from 1m/15m/1h/4h/1d using ONLY fully-closed HTF bars. merge_asof(backward) ensures a 10:30 bar never sees the in-progress 10:00 HTF candle.',
    code: `pd.merge_asof(ltf, htf, direction='backward')  # close_time <= open_time`,
  },
  {
    icon: Brain,
    title: '3. XGBoost + SHAP Discovery',
    color: 'var(--accent-purple)',
    desc: 'Train classifier to predict barrier outcomes. Extract feature importance + SHAP values to identify which conditions (trend, volatility, volume) predict wins.',
    code: `model.fit(X, y)  # features -> triple_barrier_label
shap_values = explainer(model)`,
  },
  {
    icon: Shield,
    title: '4. Purged Walk-Forward CV',
    color: 'var(--accent-green)',
    desc: 'Train on past, test on future. Embargo gap prevents label overlap. Strategy rejected unless it passes ALL folds with AUC > 0.52 and stability >= 60%.',
    code: `for fold in walk_forward(data, embargo=120):
    train -> test  # no overlap, no leakage`,
  },
];

const rrTable = [
  { rr: '3:1', target: '1.5%', stop: '0.5%', leverage: '10x', marginR: '15%', marginL: '-5%', edge: 'Trend following' },
  { rr: '2:1', target: '1.0%', stop: '0.5%', leverage: '10x', marginR: '10%', marginL: '-5%', edge: 'Balanced' },
  { rr: '1:1', target: '1.0%', stop: '1.0%', leverage: '10x', marginR: '10%', marginL: '-10%', edge: 'Mean reversion' },
  { rr: '1:2', target: '0.5%', stop: '1.0%', leverage: '10x', marginR: '5%', marginL: '-10%', edge: 'Scalping' },
  { rr: '1:3', target: '0.5%', stop: '1.5%', leverage: '10x', marginR: '5%', marginL: '-15%', edge: 'High WR scalping' },
];

export default function HowItWorks() {
  return (
    <div className="space-y-6">
      {/* Pipeline Steps */}
      <div className="space-y-3">
        {steps.map((step, i) => (
          <div key={i} className="metric-card animate-slide-in" style={{ animationDelay: `${i * 0.1}s` }}>
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 rounded-lg flex-shrink-0 flex items-center justify-center" style={{ background: `${step.color}15` }}>
                <step.icon size={18} style={{ color: step.color }} />
              </div>
              <div className="flex-1">
                <h3 className="text-sm font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>{step.title}</h3>
                <p className="text-xs leading-relaxed mb-2" style={{ color: 'var(--text-secondary)' }}>{step.desc}</p>
                <code className="block text-[11px] mono p-2 rounded" style={{ background: 'var(--bg-primary)', color: step.color, border: '1px solid var(--border)' }}>
                  {step.code}
                </code>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* RR Configuration Table */}
      <div className="metric-card">
        <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Multi-RR Search Space</h3>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>R:R</th>
                <th>Price Target</th>
                <th>Stop Loss</th>
                <th>Margin Win</th>
                <th>Margin Loss</th>
                <th>Expected Edge</th>
              </tr>
            </thead>
            <tbody>
              {rrTable.map((row) => (
                <tr key={row.rr}>
                  <td className="font-semibold" style={{ color: 'var(--accent-blue)' }}>{row.rr}</td>
                  <td className="mono" style={{ color: 'var(--accent-green)' }}>+{row.target}</td>
                  <td className="mono" style={{ color: 'var(--accent-red)' }}>-{row.stop}</td>
                  <td className="mono" style={{ color: 'var(--accent-green)' }}>+{row.marginR}</td>
                  <td className="mono" style={{ color: 'var(--accent-red)' }}>{row.marginL}</td>
                  <td className="text-xs" style={{ color: 'var(--text-secondary)' }}>{row.edge}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Research Spec */}
      <div className="metric-card">
        <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>Research Spec (Hand to AI Agent)</h3>
        <div className="space-y-2">
          <div className="text-xs p-3 rounded" style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            <p className="mb-2"><strong style={{ color: 'var(--text-primary)' }}>OBJECTIVE:</strong> Find feature conditions on Binance USD-M {'{SYMBOL}'} that precede a net +1.0% favorable move (after 0.09% round-trip cost) before a 0.5% stop, within 120x1m bars. Long-only first; repeat for short.</p>
            <p className="mb-2"><strong style={{ color: 'var(--text-primary)' }}>DATA:</strong> 1m/15m/1h/4h/1d klines from /fapi/v1/klines. HTF features aligned to closed bars only (merge_asof backward on close_time). Entry = open[t+1].</p>
            <p className="mb-2"><strong style={{ color: 'var(--text-primary)' }}>TASK:</strong></p>
            <ol className="list-decimal ml-4 space-y-1">
              <li>Use provided triple_barrier_labels target. Do NOT redefine.</li>
              <li>Engineer features ONLY from data {'<='} decision bar t.</li>
              <li>Fit classifier; report fold-by-fold fee-adjusted expectancy under purged walk-forward (embargo{'>'}=120).</li>
              <li>Report shuffle-test result.</li>
              <li>Output human-readable decision rule with out-of-sample expectancy.</li>
            </ol>
            <p className="mt-2"><strong style={{ color: 'var(--text-primary)' }}>CONSTRAINTS:</strong> No metric without purged CV. No rule if expectancy {'<'} 0 at cost=0.0015. No rule accepted if it relies on one fold only.</p>
          </div>
        </div>
      </div>

      {/* Python Engine Usage */}
      <div className="metric-card">
        <h3 className="text-sm font-semibold mb-3 flex items-center gap-2" style={{ color: 'var(--text-primary)' }}>
          <Code size={14} style={{ color: 'var(--accent-cyan)' }} />
          Python Engine Usage
        </h3>
        <div className="space-y-2">
          <code className="block text-[11px] mono p-3 rounded" style={{ background: 'var(--bg-primary)', color: 'var(--text-secondary)', border: '1px solid var(--border)', lineHeight: 1.8 }}>
            <span style={{ color: 'var(--text-muted)' }}># Install dependencies</span><br/>
            pip install pandas numpy requests xgboost scikit-learn<br/><br/>
            <span style={{ color: 'var(--text-muted)' }}># Run research</span><br/>
            cd engine<br/>
            python main.py --symbol SOLUSDT --days 60 --output result.json<br/><br/>
            <span style={{ color: 'var(--text-muted)' }}># Or in Python</span><br/>
            <span style={{ color: 'var(--accent-purple)' }}>from</span> grid_search <span style={{ color: 'var(--accent-purple)' }}>import</span> run_grid_search<br/>
            <span style={{ color: 'var(--accent-purple)' }}>from</span> data_fetcher <span style={{ color: 'var(--accent-purple)' }}>import</span> fetch_symbol_mtf<br/><br/>
            data = fetch_symbol_mtf(<span style={{ color: 'var(--accent-green)' }}>"SOLUSDT"</span>, days=<span style={{ color: 'var(--accent-yellow)' }}>60</span>)<br/>
            result = run_grid_search(data, <span style={{ color: 'var(--accent-green)' }}>"SOLUSDT"</span>, output_path=<span style={{ color: 'var(--accent-green)' }}>"sol_result.json"</span>)
          </code>
        </div>
      </div>
    </div>
  );
}
