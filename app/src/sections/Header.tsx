import { Brain, Github, ExternalLink } from 'lucide-react';

export default function Header() {
  return (
    <header style={{ background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border)' }}>
      <div className="max-w-[1440px] mx-auto px-4 h-14 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #3b82f6, #a855f7)' }}>
            <Brain size={16} className="text-white" />
          </div>
          <div className="flex items-center gap-2">
            <span className="font-bold text-sm" style={{ color: 'var(--text-primary)' }}>Strategy Research Lab</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-primary)', color: 'var(--accent-blue)', border: '1px solid var(--border)' }}>v1.0</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <a
            href="https://github.com/shubhamtaywade82/strategy_builder-python"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md transition-all hover:opacity-80"
            style={{ background: 'var(--bg-card)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
          >
            <Github size={12} />
            strategy_builder-python
            <ExternalLink size={10} />
          </a>
          <a
            href="https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/Kline-Candlestick-Data"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md transition-all hover:opacity-80"
            style={{ background: 'var(--bg-card)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
          >
            <ExternalLink size={12} />
            Binance API
          </a>
        </div>
      </div>
    </header>
  );
}
