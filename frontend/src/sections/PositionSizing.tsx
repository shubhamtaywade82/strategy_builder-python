import { useState } from 'react';
import { Calculator, TrendingUp, AlertTriangle, Shield, BarChart3 } from 'lucide-react';

interface Props {
  results: any;
}

export default function PositionSizing({ results }: Props) {
  const [accountSize, setAccountSize] = useState(10000);
  const [riskPct, setRiskPct] = useState(2);
  const [entryPrice, setEntryPrice] = useState(150);
  const [stopPrice, setStopPrice] = useState(149.25);
  const [leverage, setLeverage] = useState(10);
  const [winRate, setWinRate] = useState(57.6);
  const [avgWin, setAvgWin] = useState(0.95);
  const [avgLoss, setAvgLoss] = useState(0.59);

  const priceRisk = Math.abs(entryPrice - stopPrice) / entryPrice;
  const riskAmount = accountSize * (riskPct / 100);
  const notional = priceRisk > 0 ? riskAmount / priceRisk : 0;
  const marginRequired = notional / leverage;
  const marginPctOfAccount = (marginRequired / accountSize) * 100;

  // Kelly
  const w = winRate / 100;
  const r = avgWin / avgLoss;
  const kelly = w - ((1 - w) / r);
  const halfKelly = kelly * 0.5;
  const quarterKelly = kelly * 0.25;
  const edge = w * (avgWin / 100) - (1 - w) * (avgLoss / 100);

  // Expectancy over 100 trades
  const nTrades = 100;
  const expectedReturn = nTrades * edge * (accountSize / 100) * leverage;

  // Risk of ruin (simplified)
  const riskOfRuin = Math.pow((1 - w) / (1 - (w * r / (1 + r))), nTrades / 10) * 100;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 mb-2">
        <Calculator size={14} style={{ color: "var(--accent-blue)" }} />
        <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
          Position Sizing Calculator
        </h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Inputs */}
        <div className="metric-card space-y-3">
          <h3 className="text-xs font-semibold" style={{ color: "var(--text-muted)" }}>
            TRADE PARAMETERS
          </h3>

          {[
            { label: "Account Size ($)", value: accountSize, set: setAccountSize, min: 100, max: 1000000, step: 100 },
            { label: "Risk per Trade (%)", value: riskPct, set: setRiskPct, min: 0.1, max: 20, step: 0.1 },
            { label: "Entry Price ($)", value: entryPrice, set: setEntryPrice, min: 0.01, max: 100000, step: 0.01 },
            { label: "Stop Price ($)", value: stopPrice, set: setStopPrice, min: 0.01, max: 100000, step: 0.01 },
          ].map((field) => (
            <div key={field.label} className="space-y-1">
              <label className="text-[10px] font-medium" style={{ color: "var(--text-muted)" }}>
                {field.label}
              </label>
              <input
                type="number"
                value={field.value}
                onChange={(e) => field.set(Number(e.target.value))}
                min={field.min}
                max={field.max}
                step={field.step}
                className="input-dark w-full mono text-xs"
              />
            </div>
          ))}

          <div className="space-y-1">
            <label className="text-[10px] font-medium" style={{ color: "var(--text-muted)" }}>Leverage</label>
            <div className="flex gap-1">
              {[5, 10, 20, 50].map((l) => (
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
        </div>

        {/* Results */}
        <div className="space-y-3">
          {/* Fixed Fractional */}
          <div className="metric-card">
            <h3 className="text-xs font-semibold mb-2 flex items-center gap-1" style={{ color: "var(--text-muted)" }}>
              <Shield size={10} /> FIXED FRACTIONAL RESULT
            </h3>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span style={{ color: "var(--text-muted)" }}>Price Risk</span>
                <span className="mono font-semibold">{(priceRisk * 100).toFixed(2)}%</span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: "var(--text-muted)" }}>Risk Amount</span>
                <span className="mono font-semibold" style={{ color: "var(--accent-red)" }}>
                  ${riskAmount.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: "var(--text-muted)" }}>Position Size</span>
                <span className="mono font-semibold" style={{ color: "var(--accent-blue)" }}>
                  ${notional.toFixed(0)}
                </span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: "var(--text-muted)" }}>Margin Required</span>
                <span className="mono font-semibold">
                  ${marginRequired.toFixed(2)} ({marginPctOfAccount.toFixed(1)}% of account)
                </span>
              </div>
            </div>
          </div>

          {/* Kelly */}
          <div className="metric-card">
            <h3 className="text-xs font-semibold mb-2 flex items-center gap-1" style={{ color: "var(--text-muted)" }}>
              <TrendingUp size={10} /> KELLY CRITERION
            </h3>
            <div className="space-y-2">
              <div className="flex gap-2 text-xs">
                <div className="flex-1 space-y-1">
                  <label className="text-[10px]" style={{ color: "var(--text-muted)" }}>Win Rate %</label>
                  <input type="number" value={winRate} onChange={(e) => setWinRate(Number(e.target.value))} className="input-dark w-full mono" />
                </div>
                <div className="flex-1 space-y-1">
                  <label className="text-[10px]" style={{ color: "var(--text-muted)" }}>Avg Win %</label>
                  <input type="number" value={avgWin} onChange={(e) => setAvgWin(Number(e.target.value))} className="input-dark w-full mono" step={0.01} />
                </div>
                <div className="flex-1 space-y-1">
                  <label className="text-[10px]" style={{ color: "var(--text-muted)" }}>Avg Loss %</label>
                  <input type="number" value={avgLoss} onChange={(e) => setAvgLoss(Number(e.target.value))} className="input-dark w-full mono" step={0.01} />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2 mt-2">
                {[
                  { label: "Full Kelly", value: kelly * 100, color: kelly > 0 ? "var(--accent-yellow)" : "var(--accent-red)" },
                  { label: "Half Kelly", value: halfKelly * 100, color: halfKelly > 0 ? "var(--accent-green)" : "var(--accent-red)" },
                  { label: "Quarter Kelly", value: quarterKelly * 100, color: quarterKelly > 0 ? "var(--accent-blue)" : "var(--accent-red)" },
                ].map((k) => (
                  <div key={k.label} className="text-center p-2 rounded" style={{ background: "var(--bg-secondary)" }}>
                    <div className="text-[9px]" style={{ color: "var(--text-muted)" }}>{k.label}</div>
                    <div className="text-sm font-bold mono" style={{ color: k.color }}>
                      {k.value.toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>

              <div className="text-[10px] flex items-center gap-1 mt-1" style={{ color: "var(--text-muted)" }}>
                <AlertTriangle size={10} />
                Edge: {edge > 0 ? "+" : ""}{(edge * 100).toFixed(3)}% per trade
              </div>
            </div>
          </div>

          {/* Projections */}
          <div className="metric-card">
            <h3 className="text-xs font-semibold mb-2 flex items-center gap-1" style={{ color: "var(--text-muted)" }}>
              <BarChart3 size={10} /> 100-TRADE PROJECTION
            </h3>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span style={{ color: "var(--text-muted)" }}>Expected Return</span>
                <span className="mono font-semibold" style={{ color: expectedReturn > 0 ? "var(--accent-green)" : "var(--accent-red)" }}>
                  {expectedReturn > 0 ? "+" : ""}${expectedReturn.toFixed(0)}
                </span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: "var(--text-muted)" }}>Expected Return %</span>
                <span className="mono font-semibold" style={{ color: expectedReturn > 0 ? "var(--accent-green)" : "var(--accent-red)" }}>
                  {expectedReturn > 0 ? "+" : ""}{((expectedReturn / accountSize) * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
