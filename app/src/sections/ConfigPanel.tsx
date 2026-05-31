import { Settings, Crosshair, Calendar, Gauge } from 'lucide-react';
import { type RRRatio, RR_CONFIGS } from '../types';

interface Props {
  symbol: string;
  setSymbol: (s: string) => void;
  selectedRRs: Set<RRRatio>;
  toggleRR: (rr: RRRatio) => void;
  leverage: number;
  setLeverage: (l: number) => void;
  days: number;
  setDays: (d: number) => void;
}

const SYMBOLS = ['SOLUSDT', 'BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'LINKUSDT'];
const LEVERAGES = [5, 10, 20, 50];
const DAY_OPTIONS = [30, 60, 90, 120, 180, 365];

export default function ConfigPanel({ symbol, setSymbol, selectedRRs, toggleRR, leverage, setLeverage, days, setDays }: Props) {
  return (
    <div className="animate-fade-in metric-card">
      <div className="flex items-center gap-2 mb-4">
        <Settings size={14} style={{ color: 'var(--accent-blue)' }} />
        <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Research Configuration</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Symbol */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium flex items-center gap-1" style={{ color: 'var(--text-muted)' }}>
            <Crosshair size={10} /> Symbol
          </label>
          <select value={symbol} onChange={e => setSymbol(e.target.value)} className="input-dark w-full">
            {SYMBOLS.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>

        {/* Days */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium flex items-center gap-1" style={{ color: 'var(--text-muted)' }}>
            <Calendar size={10} /> Lookback Period
          </label>
          <select value={days} onChange={e => setDays(Number(e.target.value))} className="input-dark w-full">
            {DAY_OPTIONS.map(d => <option key={d} value={d}>{d} days</option>)}
          </select>
        </div>

        {/* Leverage */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium flex items-center gap-1" style={{ color: 'var(--text-muted)' }}>
            <Gauge size={10} /> Leverage
          </label>
          <div className="flex gap-1">
            {LEVERAGES.map(l => (
              <button
                key={l}
                onClick={() => setLeverage(l)}
                className={`flex-1 py-1.5 text-xs rounded-md font-medium transition-all ${leverage === l ? 'btn-rr-active' : 'btn-rr'}`}
              >
                {l}x
              </button>
            ))}
          </div>
        </div>

        {/* RR Ratios */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Risk:Reward Targets</label>
          <div className="flex gap-1 flex-wrap">
            {(Object.keys(RR_CONFIGS) as RRRatio[]).map(rr => (
              <button
                key={rr}
                onClick={() => toggleRR(rr)}
                className={`px-2.5 py-1.5 text-xs rounded-md font-medium transition-all ${selectedRRs.has(rr) ? 'btn-rr-active' : 'btn-rr'}`}
                title={RR_CONFIGS[rr].label}
              >
                {rr}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* RR Details */}
      {selectedRRs.size > 0 && (
        <div className="mt-4 flex gap-2 flex-wrap">
          {Array.from(selectedRRs).map(rr => (
            <div key={rr} className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-md" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
              <span className="font-semibold" style={{ color: 'var(--accent-blue)' }}>{rr}</span>
              <span>{RR_CONFIGS[rr].label}</span>
              <span style={{ color: 'var(--text-muted)' }}>|</span>
              <span>Target: +{(RR_CONFIGS[rr].upPct * 100).toFixed(1)}%</span>
              <span>Stop: -{(RR_CONFIGS[rr].dnPct * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
