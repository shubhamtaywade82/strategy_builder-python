import { useState } from 'react';
import './App.css';
import {
  Activity, Target, Shield, Zap, Brain,
  ChevronRight, Download, Play, BarChart,
  PieChart, MessageSquare, Clock, Search,
  Sparkles, X, Gamepad2, Calculator, AlertCircle,
} from 'lucide-react';
import { type RRRatio, type ResearchResult, RR_CONFIGS } from './types';
import Header from './sections/Header';
import ConfigPanel from './sections/ConfigPanel';
import ResultsOverview from './sections/ResultsOverview';
import StrategyTable from './sections/StrategyTable';
import ValidationPanel from './sections/ValidationPanel';
import FeaturePanel from './sections/FeaturePanel';
import HowItWorks from './sections/HowItWorks';
import AIChat from './sections/AIChat';
import AIStrategyInsights from './sections/AIStrategyInsights';
import StrategyPlayer from './sections/StrategyPlayer';
import PositionSizing from './sections/PositionSizing';
import { trpc } from './providers/trpc';

function App() {
  const [symbol, setSymbol] = useState('SOLUSDT');
  const [selectedRRs, setSelectedRRs] = useState<Set<RRRatio>>(new Set(['2:1', '1:1']));
  const [leverage, setLeverage] = useState(10);
  const [days, setDays] = useState(60);
  const [activeTab, setActiveTab] = useState<'results' | 'player' | 'sizing' | 'validation' | 'features' | 'ai' | 'about'>('results');
  const [results, setResults] = useState<ResearchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAIChat, setShowAIChat] = useState(false);

  const runMutation = trpc.research.run.useMutation({
    onSuccess: (data) => {
      setResults(data as ResearchResult);
      setError(null);
    },
    onError: (err) => {
      setError(err.message);
    },
  });

  const isRunning = runMutation.isPending;

  const toggleRR = (rr: RRRatio) => {
    const next = new Set(selectedRRs);
    if (next.has(rr)) next.delete(rr); else next.add(rr);
    setSelectedRRs(next);
  };

  const runResearch = () => {
    setError(null);
    runMutation.mutate({
      symbol,
      days,
      leverage,
      horizon: 120,
      rrs: Array.from(selectedRRs),
    });
  };

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg-primary)' }}>
      <Header />

      <main className="max-w-[1440px] mx-auto px-4 py-6 space-y-6">
        {/* Hero Section */}
        <div className="animate-fade-in">
          <div className="gradient-border p-6" style={{ background: 'var(--bg-card)' }}>
            <div className="flex items-start justify-between flex-wrap gap-4">
              <div className="space-y-2">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg, rgba(59,130,246,0.2), rgba(168,85,247,0.15))' }}>
                    <Brain size={20} style={{ color: 'var(--accent-blue)' }} />
                  </div>
                  <div>
                    <h1 className="text-xl font-bold" style={{ color: 'var(--text-primary)' }}>
                      AI Strategy Research Lab
                    </h1>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      Multi-RR grid search with purged walk-forward validation + LLM-powered analysis
                    </p>
                  </div>
                </div>
                <div className="flex gap-4 text-xs" style={{ color: 'var(--text-muted)' }}>
                  <span className="flex items-center gap-1"><Target size={12}/> Triple-Barrier Labels</span>
                  <span className="flex items-center gap-1"><Shield size={12}/> Leakage-Safe Features</span>
                  <span className="flex items-center gap-1"><Activity size={12}/> Purged CV</span>
                  <span className="flex items-center gap-1"><Zap size={12}/> XGBoost + SHAP</span>
                  <span className="flex items-center gap-1"><Sparkles size={12}/> Ollama AI</span>
                </div>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={() => setShowAIChat(!showAIChat)}
                  className="btn-rr flex items-center gap-2"
                  style={{ borderColor: 'var(--accent-purple)', color: 'var(--accent-purple)', background: 'rgba(168,85,247,0.1)' }}
                >
                  <MessageSquare size={14}/> AI Chat
                </button>
                <button
                  onClick={runResearch}
                  disabled={isRunning || selectedRRs.size === 0}
                  className="btn-primary flex items-center gap-2"
                  style={{ opacity: isRunning || selectedRRs.size === 0 ? 0.6 : 1 }}
                >
                  {isRunning ? (
                    <><Clock size={14} className="animate-spin"/> Running...</>
                  ) : (
                    <><Play size={14}/> Run Research</>
                  )}
                </button>
                <a href="/engine/main.py" download="strategy_engine.py" className="btn-rr flex items-center gap-1" style={{ textDecoration: 'none' }}>
                  <Download size={14}/> Engine
                </a>
              </div>
            </div>
          </div>
        </div>

        {/* Configuration Panel */}
        <ConfigPanel
          symbol={symbol}
          setSymbol={setSymbol}
          selectedRRs={selectedRRs}
          toggleRR={toggleRR}
          leverage={leverage}
          setLeverage={setLeverage}
          days={days}
          setDays={setDays}
        />

        {/* AI Chat Overlay */}
        {showAIChat && (
          <div className="fixed bottom-4 right-4 w-[440px] max-w-[calc(100vw-2rem)] z-50 animate-fade-in">
            <div className="rounded-xl overflow-hidden shadow-2xl" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
              <div className="flex items-center justify-between px-4 py-2" style={{ background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border)' }}>
                <div className="flex items-center gap-2">
                  <Sparkles size={14} style={{ color: 'var(--accent-purple)' }} />
                  <span className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>AI Strategy Assistant</span>
                </div>
                <button onClick={() => setShowAIChat(false)} className="p-1 rounded hover:opacity-70" style={{ color: 'var(--text-muted)' }}>
                  <X size={14} />
                </button>
              </div>
              <div className="p-3" style={{ maxHeight: '60vh', overflowY: 'auto' }}>
                <AIChat />
              </div>
            </div>
          </div>
        )}

        {/* Error Banner */}
        {error && (
          <div className="animate-fade-in flex items-start gap-3 p-4 rounded-xl" style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)', color: 'var(--text-primary)' }}>
            <AlertCircle size={16} style={{ color: '#ef4444', flexShrink: 0, marginTop: 2 }} />
            <div className="text-sm">
              <div className="font-semibold mb-1" style={{ color: '#ef4444' }}>Research failed</div>
              <div style={{ color: 'var(--text-muted)' }}>{error}</div>
            </div>
          </div>
        )}

        {/* Results */}
        {results && (
          <div className="animate-fade-in space-y-6">
            {/* Tab Navigation */}
            <div className="flex gap-1 border-b" style={{ borderColor: 'var(--border)' }}>
              {[
                { key: 'results' as const, label: 'Strategy Results', icon: BarChart },
                { key: 'player' as const, label: 'Strategy Player', icon: Gamepad2 },
                { key: 'sizing' as const, label: 'Position Sizing', icon: Calculator },
                { key: 'validation' as const, label: 'Validation', icon: Shield },
                { key: 'features' as const, label: 'Feature Analysis', icon: PieChart },
                { key: 'ai' as const, label: 'AI Insights', icon: Sparkles },
                { key: 'about' as const, label: 'How It Works', icon: Brain },
              ].map(tab => (
                <button
                  key={tab.key}
                  onClick={() => setActiveTab(tab.key)}
                  className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium transition-all ${activeTab === tab.key ? 'tab-active' : 'tab-inactive'}`}
                >
                  <tab.icon size={14}/>
                  {tab.label}
                </button>
              ))}
            </div>

            {activeTab === 'results' && (
              <>
                <ResultsOverview results={results} />
                <StrategyTable results={results} />
                <AIStrategyInsights results={results} />
              </>
            )}

            {activeTab === 'player' && (
              <StrategyPlayer results={results} />
            )}

            {activeTab === 'sizing' && (
              <PositionSizing results={results} />
            )}

            {activeTab === 'validation' && (
              <ValidationPanel results={results} />
            )}

            {activeTab === 'features' && (
              <FeaturePanel results={results} />
            )}

            {activeTab === 'ai' && (
              <>
                <AIStrategyInsights results={results} />
                <AIChat />
              </>
            )}

            {activeTab === 'about' && (
              <HowItWorks />
            )}
          </div>
        )}

        {/* Empty State */}
        {!results && !isRunning && (
          <div className="animate-fade-in text-center py-16">
            <div className="w-16 h-16 rounded-2xl mx-auto mb-4 flex items-center justify-center" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
              <Search size={28} style={{ color: 'var(--text-muted)' }} />
            </div>
            <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--text-secondary)' }}>
              Configure & Run Research
            </h3>
            <p className="text-sm max-w-md mx-auto" style={{ color: 'var(--text-muted)' }}>
              Select your risk-reward ratios, configure parameters, and click "Run Research"
              to discover high win-rate strategies for {symbol}.
            </p>
            <div className="mt-6 flex justify-center gap-2">
              {(['3:1','2:1','1:1','1:2','1:3'] as RRRatio[]).map(rr => (
                <button
                  key={rr}
                  onClick={() => toggleRR(rr)}
                  className={`btn-rr ${selectedRRs.has(rr) ? 'btn-rr-active' : ''}`}
                >
                  {rr}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Running State */}
        {isRunning && (
          <div className="animate-fade-in text-center py-16">
            <div className="w-16 h-16 rounded-2xl mx-auto mb-4 flex items-center justify-center animate-pulse" style={{ background: 'var(--bg-card)', border: '1px solid var(--accent-blue)' }}>
              <Activity size={28} style={{ color: 'var(--accent-blue)' }} />
            </div>
            <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--text-secondary)' }}>
              Running Multi-RR Grid Search...
            </h3>
            <div className="max-w-md mx-auto space-y-1">
              {Array.from(selectedRRs).map((rr, i) => (
                <div key={rr} className="flex items-center gap-2 text-sm animate-slide-in" style={{ color: 'var(--text-muted)', animationDelay: `${i * 0.2}s` }}>
                  <ChevronRight size={14} style={{ color: 'var(--accent-blue)' }} />
                  <span>Searching RR {rr} — {RR_CONFIGS[rr].label}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
