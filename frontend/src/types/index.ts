export interface StrategyConfig {
  symbol: string;
  rr: string;
  upPct: number;
  dnPct: number;
  leverage: number;
  horizon: number;
  side: 'long' | 'short' | 'both';
}

export interface Trade {
  entryTime: string;
  entryPrice: number;
  side: number;
  exitTime: string;
  exitPrice: number;
  exitReason: string;
  netPnl: number;
  marginPnl: number;
  barsHeld: number;
}

export interface BacktestMetrics {
  tradeCount: number;
  winCount: number;
  lossCount: number;
  winRate: number;
  profitFactor: number;
  expectancy: number;
  netPnl: number;
  avgWin: number;
  avgLoss: number;
  maxDrawdown: number;
  sharpe: number;
  avgBarsHeld: number;
  targetHitRate: number;
  stopHitRate: number;
}

export interface StrategyRule {
  id: string;
  name: string;
  description: string;
  side: string;
  conditions: RuleCondition[];
  metrics: BacktestMetrics;
  isViable: boolean;
  rejectReason?: string;
}

export interface RuleCondition {
  feature: string;
  operator: string;
  threshold: number;
  importance: number;
}

export interface FoldResult {
  fold: number;
  trainAuc: number;
  testAuc: number;
  testPrecision: number;
  testRecall: number;
  nTrain: number;
  nTest: number;
}

export interface WalkForwardSummary {
  folds: number;
  avgTestAuc: number;
  minTestAuc: number;
  stability: number;
  degradation: number;
  isValid: boolean;
  foldResults: FoldResult[];
}

export interface FeatureInsight {
  feature: string;
  direction: 'high' | 'low';
  threshold: number;
  importance: number;
  winRateAbove: number;
  winRateBelow: number;
  shapValue: number;
}

export interface ResearchResult {
  symbol: string;
  config: StrategyConfig;
  strategies: StrategyRule[];
  walkForward: WalkForwardSummary;
  topFeatures: FeatureInsight[];
  labels: {
    longWins: number;
    shortWins: number;
    noTrade: number;
    total: number;
  };
  player?: PlayerResult;
}

export interface RuleCell {
  side: 'long' | 'short';
  mode: 'momentum' | 'reversion';
  trade_direction: 'long' | 'short';
  n_signals: number;
  baseline: { win_rate: number; expectancy: number };
  edge_win_rate?: number;
  edge_expectancy?: number;
  metrics: {
    trade_count: number;
    win_rate: number;
    profit_factor: number | null;
    expectancy_net: number;
    avg_win: number;
    avg_loss: number;
    sharpe: number;
    target_hit_rate: number;
    avg_bars_held: number;
    p_value: number;
  };
  walk_forward: { folds: Array<{ fold: number; trades: number; expectancy: number | null; win_rate: number | null }>; stability: number };
  robustness_score: number;
  is_robust: boolean;
  warnings: string[];
}

export interface PlayerResult {
  error?: string;
  rr?: string;
  cost?: number;
  leverage?: number;
  horizon_bars?: number;
  conditions?: Array<{ id: string; label: string }>;
  long?: { momentum: RuleCell; reversion: RuleCell };
  short?: { momentum: RuleCell; reversion: RuleCell };
  best?: RuleCell | null;
  has_edge?: boolean;
}

export type RRRatio = '3:1' | '2:1' | '1:1' | '1:2' | '1:3';

export const RR_CONFIGS: Record<RRRatio, { upPct: number; dnPct: number; label: string }> = {
  '3:1': { upPct: 0.015, dnPct: 0.005, label: '3:1 (1.5% / 0.5%)' },
  '2:1': { upPct: 0.01, dnPct: 0.005, label: '2:1 (1.0% / 0.5%)' },
  '1:1': { upPct: 0.01, dnPct: 0.01, label: '1:1 (1.0% / 1.0%)' },
  '1:2': { upPct: 0.005, dnPct: 0.01, label: '1:2 (0.5% / 1.0%)' },
  '1:3': { upPct: 0.005, dnPct: 0.015, label: '1:3 (0.5% / 1.5%)' },
};
