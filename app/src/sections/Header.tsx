import { Brain, Github, ExternalLink, Wifi, WifiOff, TrendingUp, TrendingDown } from 'lucide-react';
import { useBinanceTicker } from '@/hooks/useBinanceTicker';

interface HeaderProps {
  symbol?: string;
}

function fmt(n: number, decimals = 2): string {
  if (!n) return '—';
  return n.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

function fmtVolume(n: number): string {
  if (!n) return '—';
  if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
  return `$${n.toFixed(0)}`;
}

export default function Header({ symbol = 'SOLUSDT' }: HeaderProps) {
  const ticker = useBinanceTicker(symbol);
  const isUp = ticker.change24h >= 0;
  const fundingPct = (ticker.fundingRate * 100).toFixed(4);

  return (
    <header style={{ background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border)' }}>
      <div className="max-w-[1440px] mx-auto px-4 h-14 flex items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-3 flex-shrink-0">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #3b82f6, #a855f7)' }}>
            <Brain size={16} className="text-white" />
          </div>
          <div className="flex items-center gap-2">
            <span className="font-bold text-sm" style={{ color: 'var(--text-primary)' }}>Strategy Research Lab</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-primary)', color: 'var(--accent-blue)', border: '1px solid var(--border)' }}>v2.0</span>
          </div>
        </div>

        {/* Live Ticker */}
        <div className="flex items-center gap-5 text-xs flex-1 justify-center flex-wrap">
          {/* Symbol + Price */}
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              {ticker.connected
                ? <Wifi size={10} style={{ color: '#22c55e' }} />
                : <WifiOff size={10} style={{ color: '#6b7280' }} />}
              <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>{symbol}</span>
            </div>
            <span className="font-bold text-base" style={{ color: 'var(--text-primary)' }}>
              ${ticker.markPrice ? fmt(ticker.markPrice, ticker.markPrice > 100 ? 2 : 4) : '—'}
            </span>
            {ticker.change24h !== 0 && (
              <span className="flex items-center gap-0.5 font-semibold" style={{ color: isUp ? '#22c55e' : '#ef4444' }}>
                {isUp ? <TrendingUp size={11} /> : <TrendingDown size={11} />}
                {isUp ? '+' : ''}{fmt(ticker.change24h, 2)}%
              </span>
            )}
          </div>

          {/* Stat pills */}
          <div className="flex items-center gap-3" style={{ color: 'var(--text-muted)' }}>
            <span>H: <span style={{ color: 'var(--text-secondary)' }}>${fmt(ticker.high24h, ticker.high24h > 100 ? 2 : 4)}</span></span>
            <span>L: <span style={{ color: 'var(--text-secondary)' }}>${fmt(ticker.low24h, ticker.low24h > 100 ? 2 : 4)}</span></span>
            <span>Vol: <span style={{ color: 'var(--text-secondary)' }}>{fmtVolume(ticker.volume24h)}</span></span>
            <span>
              Funding:{' '}
              <span style={{ color: ticker.fundingRate >= 0 ? '#f59e0b' : '#22c55e' }}>
                {ticker.fundingRate !== 0 ? `${ticker.fundingRate >= 0 ? '+' : ''}${fundingPct}%` : '—'}
              </span>
            </span>
            {ticker.markPrice > 0 && ticker.indexPrice > 0 && (
              <span>
                Basis:{' '}
                <span style={{ color: 'var(--text-secondary)' }}>
                  {((ticker.markPrice / ticker.indexPrice - 1) * 100).toFixed(3)}%
                </span>
              </span>
            )}
          </div>
        </div>

        {/* Links */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <a
            href="https://github.com/shubhamtaywade82/strategy_builder-python"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md transition-all hover:opacity-80"
            style={{ background: 'var(--bg-card)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
          >
            <Github size={12} />
            GitHub
          </a>
          <a
            href="https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data"
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
